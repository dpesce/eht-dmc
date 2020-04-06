#######################################################
# imports
#######################################################

from __future__ import division
from __future__ import print_function
from builtins import list
from builtins import len
from builtins import range
from builtins import enumerate

import numpy as np
import ehtim as eh
import pymc3 as pm
import dmc.data_utils as du
import dmc.model_utils as mu

#######################################################
# constants
#######################################################

MAX_TREEDEPTH = 10
EARLY_MAX_TREEDEPTH = 10

#######################################################
# functions
#######################################################

def image(obs,nx,ny,xmin,xmax,ymin,ymax,total_flux_estimate=None,loose_change=False,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,smooth=False,
          n_start=25,n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,**kwargs):
    """ Fit a Stokes I image to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           nx (int): number of image pixels in the x-direction
           ny (int): number of image pixels in the y-direction
           xmin(float): minimum x pixel value (uas)
           xmax(float): maximum x pixel value (uas)
           ymin(float): minimum y pixel value (uas)
           ymax(float): maximum x pixel value (uas)
           
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           offset_window (float): width of square offset window (uas)
    
           loose_change (bool): flag to use the "loose change" noise prescription
           fit_total_flux (bool): flag to fit for the total flux
           allow_offset (bool): flag to permit image centroid to be a free parameter
           smooth (bool): flag to fit for a Gaussian smoothing kernel
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)

    ###################################################
    # data bookkeeping

    # first, make sure we're using a Stokes representation
    if obs.polrep is not 'stokes':
        obs = obs.switch_polrep('stokes')

    # (u,v) coordinate info
    u = obs.data['u']
    v = obs.data['v']
    rho = np.sqrt((u**2.0) + (v**2.0))

    # get array of stations
    ant1 = obs.data['t1']
    ant2 = obs.data['t2']
    stations = np.unique(np.concatenate((ant1,ant2)))

    # read in the real and imaginary visibilities
    I_real = np.real(obs.data['vis'])
    I_imag = np.imag(obs.data['vis'])
    I_real_err = obs.data['sigma']
    I_imag_err = obs.data['sigma']

    # construct mask to remove missing data
    mask = np.where(np.isfinite(obs.data['vis']))

    # construct design matrices for gain terms
    T_gains, A_gains = du.gain_account(obs)
    gain_design_mat_1, gain_design_mat_2 = du.gain_design_mats(obs)

    # if there's no input total flux estimate, estimate it here
    if total_flux_estimate is None:
        total_flux_estimate = du.estimate_total_flux(obs)

    ###################################################
    # organizing image information

    # total number of pixels
    npix = nx*ny

    # one-dimensional pixel vectors
    x_1d = np.linspace(xmin,xmax,nx)
    y_1d = np.linspace(ymin,ymax,ny)

    # two-dimensional pixel vectors
    x2d, y2d = np.meshgrid(x_1d,y_1d,indexing='ij')

    # convert from microarcseconds to radians
    x = eh.RADPERUAS*x2d.ravel()
    y = eh.RADPERUAS*y2d.ravel()

    # constructing Fourier transform matrix
    A = np.zeros((len(u),len(x)),dtype='complex')
    for i in range(len(u)):
        A[i,:] = np.exp(-2.0*np.pi*(1j)*((u[i]*x) + (v[i]*y)))

    # Taking a complex conjugate to account for eht-imaging internal FT convention
    A_real = np.real(A)
    A_imag = -np.imag(A)

    ###################################################
    # organizing prior information

    # prior info for log gain amplitudes
    loggainamp_mean, loggainamp_std = mu.gain_logamp_prior(obs)
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,ref_station=ref_station)
    
    # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
    dirichlet_weights = 1.0*np.ones_like(x)

    ###################################################
    # setting up the model

    # number of gains
    N_gains = len(loggainamp_mean)

    model = pm.Model()

    with model:

        ###############################################
        # set the priors for the image parameters
        
        # total flux prior
        if fit_total_flux:
            # set to be normal around the correct value, but bounded positive
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
            F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        # Impose a Dirichlet prior on the pixel intensities,
        # with summation constraint equal to the total flux
        pix = pm.Dirichlet('pix',dirichlet_weights)
        I = pm.Deterministic('I',pix*F)

        # systematic noise prescription
        if loose_change:
            # set the prior on the multiplicative systematic error term to be uniform on [0,1]
            multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=1.0)

            # set the prior on the additive systematic error term to be uniform on [0,100] mJy
            additive = 0.1*pm.Uniform('additive',lower=0.0,upper=0.1)
        else:
            # set the prior on the systematic error term to be uniform on [0,1]
            f = pm.Uniform('f',lower=0.0,upper=1.0)

        # permit a centroid shift in the image
        if allow_offset:
            x0 = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0))
            y0 = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0))
        else:
            x0 = 0.0
            y0 = 0.0

        # Gaussian smoothing kernel parameters
        if smooth:
            # smoothing width
            sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=20.0)

        ###############################################
        # set the priors for the gain parameters

        # set the gain amplitude priors to be log-normal around the specified inputs
        logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
        g = pm.Deterministic('gain_amps',pm.math.exp(logg))
        
        # set the gain phase priors to be periodic uniform on (-pi,pi)
        theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal_pregain_preshift_presmooth = pm.math.dot(A_real,I)
        Iimag_pregain_preshift_presmooth = pm.math.dot(A_imag,I)

        ###############################################
        # smooth with the Gaussian kernel
        
        if smooth:
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        else:
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0) + (v*y0))
        Ireal_pregain = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # compute the corruption terms

        gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
        gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

        gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
        gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        ###############################################
        # apply corruptions to the model visibilities

        Ireal_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Ireal_pregain) - (pm.math.sin(gainphase_1 - gainphase_2)*Iimag_pregain))
        Iimag_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Iimag_pregain) + (pm.math.sin(gainphase_1 - gainphase_2)*Ireal_pregain))

        ###############################################
        # add in the systematic noise component

        if loose_change:
            Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
            Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + ((multiplicative*Itot_model)**2.0) + (additive**2.0))
            Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + ((multiplicative*Itot_model)**2.0) + (additive**2.0))
        else:
            Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + (f*F)**2.0)
            Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + (f*F)**2.0)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # NOTE: the current tuning scheme is rather arbitrary
    # and could likely benefit from systematization

    # set up tuning windows
    windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        start = None
        burnin_trace = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH,regularize=regularize)
            burnin_trace = pm.sample(start=start, tune=steps, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            start = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=start, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'image',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'xmin': xmin,
                 'xmax': xmax,
                 'ymin': ymin,
                 'ymax': ymax,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains
                 }

    return modelinfo

def polimage(obs,nx,ny,xmin,xmax,ymin,ymax,total_flux_estimate=None,RLequal=False,
          fit_StokesV=True,fit_total_flux=False,n_start=25,n_burn=500,n_tune=5000,
          ntuning=2000,ntrials=10000,**kwargs):
    """ Fit a polarimetric image to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           nx (int): number of image pixels in the x-direction
           ny (int): number of image pixels in the y-direction
           xmin(float): minimum x pixel value (uas)
           xmax(float): maximum x pixel value (uas)
           ymin(float): minimum y pixel value (uas)
           ymax(float): maximum x pixel value (uas)
           
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           
           RLequal (bool): flag to fix right and left gain terms to be equal
           fit_StokesV (bool): flag to fit for Stokes V; set to False to fix V = 0
           fit_total_flux (bool): flag to fit for the total flux
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)

    ###################################################
    # data bookkeeping

    # first, make sure we're using a circular representation
    if obs.polrep is not 'circ':
        obs = obs.switch_polrep('circ')

    # (u,v) coordinate info
    u = obs.data['u']
    v = obs.data['v']
    rho = np.sqrt((u**2.0) + (v**2.0))

    # read in the real and imaginary parts for each data product
    RR_real = np.real(obs.data['rrvis'])
    RR_imag = np.imag(obs.data['rrvis'])
    RR_real_err = obs.data['rrsigma']
    RR_imag_err = obs.data['rrsigma']

    LL_real = np.real(obs.data['llvis'])
    LL_imag = np.imag(obs.data['llvis'])
    LL_real_err = obs.data['llsigma']
    LL_imag_err = obs.data['llsigma']

    RL_real = np.real(obs.data['rlvis'])
    RL_imag = np.imag(obs.data['rlvis'])
    RL_real_err = obs.data['rlsigma']
    RL_imag_err = obs.data['rlsigma']

    LR_real = np.real(obs.data['lrvis'])
    LR_imag = np.imag(obs.data['lrvis'])
    LR_real_err = obs.data['lrsigma']
    LR_imag_err = obs.data['lrsigma']

    # mask out any blank data by giving it enormous uncertainties
    mask = ~np.isfinite(obs.data['rrvis'])
    RR_real[mask] = 0.0
    RR_imag[mask] = 0.0
    RR_real_err[mask] = 1000.0
    RR_imag_err[mask] = 1000.0
    mask_RR = np.where(np.isfinite(obs.data['rrvis']))

    mask = ~np.isfinite(obs.data['llvis'])
    LL_real[mask] = 0.0
    LL_imag[mask] = 0.0
    LL_real_err[mask] = 1000.0
    LL_imag_err[mask] = 1000.0
    mask_LL = np.where(np.isfinite(obs.data['llvis']))

    mask = ~np.isfinite(obs.data['rlvis'])
    RL_real[mask] = 0.0
    RL_imag[mask] = 0.0
    RL_real_err[mask] = 1000.0
    RL_imag_err[mask] = 1000.0
    mask_RL = np.where(np.isfinite(obs.data['rlvis']))

    mask = ~np.isfinite(obs.data['lrvis'])
    LR_real[mask] = 0.0
    LR_imag[mask] = 0.0
    LR_real_err[mask] = 1000.0
    LR_imag_err[mask] = 1000.0
    mask_LR = np.where(np.isfinite(obs.data['lrvis']))

    # construct design matrices for gain terms
    T_gains, A_gains = du.gain_account(obs)
    gain_design_mat_1, gain_design_mat_2 = du.gain_design_mats(obs)

    # construct design matrices for leakage terms
    stations, dterm_design_mat_1, dterm_design_mat_2 = du.dterm_design_mats(obs)

    # construct vectors of field rotation corrections
    FR1, FR2 = du.FRvec(obs,ehtim_convention=ehtim_convention)

    # if there's no input total flux estimate, estimate it here
    if total_flux_estimate is None:
        total_flux_estimate = du.estimate_total_flux(obs)

    ###################################################
    # organizing image information

    # total number of pixels
    npix = nx*ny

    # one-dimensional pixel vectors
    x_1d = np.linspace(xmin,xmax,nx)
    y_1d = np.linspace(ymin,ymax,ny)

    # two-dimensional pixel vectors
    x2d, y2d = np.meshgrid(x_1d,y_1d,indexing='ij')

    # convert from microarcseconds to radians
    x = eh.RADPERUAS*x2d.ravel()
    y = eh.RADPERUAS*y2d.ravel()

    # constructing Fourier transform matrix
    A = np.zeros((len(u),len(x)),dtype='complex')
    for i in range(len(u)):
        A[i,:] = np.exp(-2.0*np.pi*(1j)*((u[i]*x) + (v[i]*y)))

    # Taking a complex conjugate to account for eht-imaging internal FT convention
    A_real = np.real(A)
    A_imag = -np.imag(A)

    ###################################################
    # organizing prior information

    # prior info for log gain amplitudes
    loggainamp_mean, loggainamp_std = mu.gain_logamp_prior(obs)

    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,ref_station=ref_station)

    # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
    dirichlet_weights = 1.0*np.ones_like(x)

    ###################################################
    # setting up the model

    # number of gains and dterms
    N_Dterms = dterm_design_mat_1.shape[1]
    N_gains = len(loggainamp_mean)

    model = pm.Model()

    with model:

        ###############################################
        # set the priors for the image parameters
        
        # total flux prior
        if fit_total_flux:
            # set to be normal around the correct value, but bounded positive
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
            F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        # Impose a Dirichlet prior on the pixel Stokes I intensities,
        # with summation constraint equal to the total flux
        pix = pm.Dirichlet('pix',dirichlet_weights)
        I = pm.Deterministic('I',pix*F)
        
        # sample the polarization fraction uniformly on [0,1]
        p = pm.Uniform('p',lower=0.0,upper=1.0,shape=npix)

        # sample alpha from a periodic uniform distribution
        alpha = pm.VonMises('alpha',mu=0.0,kappa=0.0001,shape=npix)
        EVPA = pm.Deterministic('EVPA',alpha/2.0)

        # circular polarization angle
        if fit_StokesV:
            # sample cos(beta) uniformly on [-1,1]
            cosbeta = pm.Uniform('cosbeta',lower=-1.0,upper=1.0,shape=npix)
        else:
            # fix to be zero
            cosbeta = 0.0
        sinbeta = pm.math.sqrt(1.0 - (cosbeta**2.0))

        # set the prior on the systematic error term to be uniform on [0,1]
        f = pm.Uniform('f',lower=0.0,upper=1.0)

        ###############################################
        # set the priors for the gain parameters

        # set the gain amplitude priors to be log-normal around the specified inputs
        logg_R = pm.Normal('right_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
        g_R = pm.Deterministic('right_gain_amps',pm.math.exp(logg_R))

        if RLequal:
            logg_L = pm.Deterministic('left_logg',logg_R)
            g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
        else:
            logg_L = pm.Normal('left_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
            g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
        
        # set the gain phase priors to be periodic uniform on (-pi,pi)
        theta_R = pm.VonMises('right_gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)
        
        if RLequal:
            theta_L = pm.Deterministic('left_gain_phases',theta_R)
        else:
            theta_L = pm.VonMises('left_gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # set the priors for the leakage parameters
        
        # set the D term amplitude priors to be uniform on [0,1]
        Damp_R = pm.Uniform('right_Dterm_amps',lower=0.0,upper=1.0,shape=N_Dterms,testval=0.01)
        logDamp_R = pm.math.log(Damp_R)

        Damp_L = pm.Uniform('left_Dterm_amps',lower=0.0,upper=1.0,shape=N_Dterms,testval=0.01)
        logDamp_L = pm.math.log(Damp_L)

        # set the D term phase priors to be periodic uniform on (-pi,pi)
        delta_R = pm.VonMises('right_Dterm_phases',mu=0.0,kappa=0.0001,shape=N_Dterms)
        delta_L = pm.VonMises('left_Dterm_phases',mu=0.0,kappa=0.0001,shape=N_Dterms)

        # save the real and imaginary parts for output diagnostics
        D_R_real = pm.Deterministic('right_Dterm_reals',Damp_R*pm.math.cos(delta_R))
        D_R_imag = pm.Deterministic('right_Dterm_imags',Damp_R*pm.math.sin(delta_R))
        D_L_real = pm.Deterministic('left_Dterm_reals',Damp_L*pm.math.cos(delta_L))
        D_L_imag = pm.Deterministic('left_Dterm_imags',Damp_L*pm.math.sin(delta_L))

        ###############################################
        # compute the polarized Stokes parameters

        Q = pm.Deterministic('Q',I*p*pm.math.cos(alpha)*sinbeta)
        U = pm.Deterministic('U',I*p*pm.math.sin(alpha)*sinbeta)
        V = pm.Deterministic('V',I*p*cosbeta)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal = pm.math.dot(A_real,I)
        Iimag = pm.math.dot(A_imag,I)
        
        Qreal = pm.math.dot(A_real,Q)
        Qimag = pm.math.dot(A_imag,Q)
        
        Ureal = pm.math.dot(A_real,U)
        Uimag = pm.math.dot(A_imag,U)
        
        Vreal = pm.math.dot(A_real,V)
        Vimag = pm.math.dot(A_imag,V)

        ###############################################
        # construct the pre-corrupted circular basis model visibilities

        RR_real_pregain = Ireal + Vreal
        RR_imag_pregain = Iimag + Vimag

        LL_real_pregain = Ireal - Vreal
        LL_imag_pregain = Iimag - Vimag

        RL_real_pregain = Qreal - Uimag
        RL_imag_pregain = Qimag + Ureal

        LR_real_pregain = Qreal + Uimag
        LR_imag_pregain = Qimag - Ureal

        ###############################################
        # compute the corruption terms
        
        gainamp_R1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg_R))
        gainamp_R2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg_R))
        gainamp_L1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg_L))
        gainamp_L2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg_L))
        
        gainphase_R1 = pm.math.dot(gain_design_mat_1,theta_R)
        gainphase_R2 = pm.math.dot(gain_design_mat_2,theta_R)
        gainphase_L1 = pm.math.dot(gain_design_mat_1,theta_L)
        gainphase_L2 = pm.math.dot(gain_design_mat_2,theta_L)
        
        Damp_R1 = pm.math.exp(pm.math.dot(dterm_design_mat_1,logDamp_R))
        Damp_R2 = pm.math.exp(pm.math.dot(dterm_design_mat_2,logDamp_R))
        Damp_L1 = pm.math.exp(pm.math.dot(dterm_design_mat_1,logDamp_L))
        Damp_L2 = pm.math.exp(pm.math.dot(dterm_design_mat_2,logDamp_L))
        
        Dphase_R1_preFR = pm.math.dot(dterm_design_mat_1,delta_R)
        Dphase_R2_preFR = pm.math.dot(dterm_design_mat_2,delta_R)
        Dphase_L1_preFR = pm.math.dot(dterm_design_mat_1,delta_L)
        Dphase_L2_preFR = pm.math.dot(dterm_design_mat_2,delta_L)
        
        Dphase_R1 = Dphase_R1_preFR + FR1
        Dphase_R2 = Dphase_R2_preFR + FR2
        Dphase_L1 = Dphase_L1_preFR - FR1
        Dphase_L2 = Dphase_L2_preFR - FR2
        
        ###############################################
        # apply corruptions to the model visibilities

        RR_real_model = gainamp_R1*gainamp_R2*((pm.math.cos(gainphase_R1 - gainphase_R2)*RR_real_pregain)
                        - (pm.math.sin(gainphase_R1 - gainphase_R2)*RR_imag_pregain)
                        + (Damp_R1*Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_real_pregain)
                        - (Damp_R1*Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_imag_pregain)
                        + (Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_real_pregain)
                        - (Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_imag_pregain)
                        + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_real_pregain)
                        - (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_imag_pregain))

        RR_imag_model = gainamp_R1*gainamp_R2*((pm.math.sin(gainphase_R1 - gainphase_R2)*RR_real_pregain)
                        + (pm.math.cos(gainphase_R1 - gainphase_R2)*RR_imag_pregain)
                        + (Damp_R1*Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_real_pregain)
                        + (Damp_R1*Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_imag_pregain)
                        + (Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_real_pregain)
                        + (Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_imag_pregain)
                        + (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_real_pregain)
                        + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_imag_pregain))

        LL_real_model = gainamp_L1*gainamp_L2*((Damp_L1*Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_real_pregain)
                        - (Damp_L1*Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_imag_pregain)
                        + (pm.math.cos(gainphase_L1 - gainphase_L2)*LL_real_pregain)
                        - (pm.math.sin(gainphase_L1 - gainphase_L2)*LL_imag_pregain)
                        + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_real_pregain)
                        - (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_imag_pregain)
                        + (Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_real_pregain)
                        - (Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_imag_pregain))

        LL_imag_model = gainamp_L1*gainamp_L2*((Damp_L1*Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_real_pregain)
                        + (Damp_L1*Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_imag_pregain)
                        + (pm.math.sin(gainphase_L1 - gainphase_L2)*LL_real_pregain)
                        + (pm.math.cos(gainphase_L1 - gainphase_L2)*LL_imag_pregain)
                        + (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_real_pregain)
                        + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_imag_pregain)
                        + (Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_real_pregain)
                        + (Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_imag_pregain))

        RL_real_model = gainamp_R1*gainamp_L2*((Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_real_pregain)
                        - (Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_imag_pregain)
                        + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_real_pregain)
                        - (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_imag_pregain)
                        + (pm.math.cos(gainphase_R1 - gainphase_L2)*RL_real_pregain)
                        - (pm.math.sin(gainphase_R1 - gainphase_L2)*RL_imag_pregain)
                        + (Damp_R1*Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_real_pregain)
                        - (Damp_R1*Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_imag_pregain))

        RL_imag_model = gainamp_R1*gainamp_L2*((Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_real_pregain)
                        + (Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_imag_pregain)
                        + (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_real_pregain)
                        + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_imag_pregain)
                        + (pm.math.sin(gainphase_R1 - gainphase_L2)*RL_real_pregain)
                        + (pm.math.cos(gainphase_R1 - gainphase_L2)*RL_imag_pregain)
                        + (Damp_R1*Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_real_pregain)
                        + (Damp_R1*Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_imag_pregain))

        LR_real_model = gainamp_L1*gainamp_R2*((Damp_L1*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_real_pregain)
                        - (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_imag_pregain)
                        + (Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_real_pregain)
                        - (Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_imag_pregain)
                        + (Damp_L1*Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_real_pregain)
                        - (Damp_L1*Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_imag_pregain)
                        + (pm.math.cos(gainphase_L1 - gainphase_R2)*LR_real_pregain)
                        - (pm.math.sin(gainphase_L1 - gainphase_R2)*LR_imag_pregain))

        LR_imag_model = gainamp_L1*gainamp_R2*((Damp_L1*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_real_pregain)
                        + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_imag_pregain)
                        + (Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_real_pregain)
                        + (Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_imag_pregain)
                        + (Damp_L1*Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_real_pregain)
                        + (Damp_L1*Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_imag_pregain)
                        + (pm.math.sin(gainphase_L1 - gainphase_R2)*LR_real_pregain)
                        + (pm.math.cos(gainphase_L1 - gainphase_R2)*LR_imag_pregain))

        ###############################################
        # add in the systematic noise component

        RR_real_err_model = pm.math.sqrt((RR_real_err**2.0) + (f*F)**2.0)
        RR_imag_err_model = pm.math.sqrt((RR_imag_err**2.0) + (f*F)**2.0)

        LL_real_err_model = pm.math.sqrt((LL_real_err**2.0) + (f*F)**2.0)
        LL_imag_err_model = pm.math.sqrt((LL_imag_err**2.0) + (f*F)**2.0)

        RL_real_err_model = pm.math.sqrt((RL_real_err**2.0) + (f*F)**2.0)
        RL_imag_err_model = pm.math.sqrt((RL_imag_err**2.0) + (f*F)**2.0)

        LR_real_err_model = pm.math.sqrt((LR_real_err**2.0) + (f*F)**2.0)
        LR_imag_err_model = pm.math.sqrt((LR_imag_err**2.0) + (f*F)**2.0)

        ###############################################
        # define the likelihood

        L_real_RR = pm.Normal('L_real_RR',mu=RR_real_model[mask_RR],sd=RR_real_err_model[mask_RR],observed=RR_real[mask_RR])
        L_imag_RR = pm.Normal('L_imag_RR',mu=RR_imag_model[mask_RR],sd=RR_imag_err_model[mask_RR],observed=RR_imag[mask_RR])

        L_real_LL = pm.Normal('L_real_LL',mu=LL_real_model[mask_LL],sd=LL_real_err_model[mask_LL],observed=LL_real[mask_LL])
        L_imag_LL = pm.Normal('L_imag_LL',mu=LL_imag_model[mask_LL],sd=LL_imag_err_model[mask_LL],observed=LL_imag[mask_LL])

        L_real_RL = pm.Normal('L_real_RL',mu=RL_real_model[mask_RL],sd=RL_real_err_model[mask_RL],observed=RL_real[mask_RL])
        L_imag_RL = pm.Normal('L_imag_RL',mu=RL_imag_model[mask_RL],sd=RL_imag_err_model[mask_RL],observed=RL_imag[mask_RL])

        L_real_LR = pm.Normal('L_real_LR',mu=LR_real_model[mask_LR],sd=LR_real_err_model[mask_LR],observed=LR_real[mask_LR])
        L_imag_LR = pm.Normal('L_imag_LR',mu=LR_imag_model[mask_LR],sd=LR_imag_err_model[mask_LR],observed=LR_imag[mask_LR])
        
        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr_RR = pm.Deterministic('ssr_RR',pm.math.sum((((RR_real_model[mask_RR]-RR_real[mask_RR])/RR_real_err_model[mask_RR])**2.0) + (((RR_imag_model[mask_RR]-RR_imag[mask_RR])/RR_imag_err_model[mask_RR])**2.0)))
        ssr_LL = pm.Deterministic('ssr_LL',pm.math.sum((((LL_real_model[mask_LL]-LL_real[mask_LL])/LL_real_err_model[mask_LL])**2.0) + (((LL_imag_model[mask_LL]-LL_imag[mask_LL])/LL_imag_err_model[mask_LL])**2.0)))
        ssr_RL = pm.Deterministic('ssr_RL',pm.math.sum((((RL_real_model[mask_RL]-RL_real[mask_RL])/RL_real_err_model[mask_RL])**2.0) + (((RL_imag_model[mask_RL]-RL_imag[mask_RL])/RL_imag_err_model[mask_RL])**2.0)))
        ssr_LR = pm.Deterministic('ssr_LR',pm.math.sum((((LR_real_model[mask_LR]-LR_real[mask_LR])/LR_real_err_model[mask_LR])**2.0) + (((LR_imag_model[mask_LR]-LR_imag[mask_LR])/LR_imag_err_model[mask_LR])**2.0)))
        
    ###################################################
    # fit the model

    # NOTE: the current tuning scheme is rather arbitrary
    # and could likely benefit from systematization

    # set up tuning windows
    windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        start = None
        burnin_trace = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH,regularize=regularize)
            burnin_trace = pm.sample(start=start, tune=steps, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            start = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=start, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'polimage',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'xmin': xmin,
                 'xmax': xmax,
                 'ymin': ymin,
                 'ymax': ymax,
                 'fit_total_flux': fit_total_flux,
                 'total_flux_estimate': total_flux_estimate,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'RLequal': RLequal,
                 'fit_StokesV': fit_StokesV,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains
                 }

    return modelinfo

def point(obs,total_flux_estimate=None,fit_total_flux=True,
          allow_offset=False,offset_window=200.0,n_start=25,
          n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,**kwargs):
    """ Fit a point source model to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data           
           total_flux_estimate (float): estimate of total Stokes I flux (Jy)
           
           fit_total_flux (bool): flag to fit for the total flux
           allow_offset (bool): flag to permit image centroid to be a free parameter
           offset_window (float): width of square offset window (uas)
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)

    ###################################################
    # data bookkeeping

    # first, make sure we're using a Stokes representation
    if obs.polrep is not 'stokes':
        obs = obs.switch_polrep('stokes')

    # (u,v) coordinate info
    u = obs.data['u']
    v = obs.data['v']
    rho = np.sqrt((u**2.0) + (v**2.0))

    # get array of stations
    ant1 = obs.data['t1']
    ant2 = obs.data['t2']
    stations = np.unique(np.concatenate((ant1,ant2)))

    # read in the real and imaginary visibilities
    I_real = np.real(obs.data['vis'])
    I_imag = np.imag(obs.data['vis'])
    I_real_err = obs.data['sigma']
    I_imag_err = obs.data['sigma']

    # construct mask to remove missing data
    mask = np.where(np.isfinite(obs.data['vis']))

    # construct design matrices for gain terms
    T_gains, A_gains = du.gain_account(obs)
    gain_design_mat_1, gain_design_mat_2 = du.gain_design_mats(obs)

    # if there's no input total flux estimate, estimate it here
    if total_flux_estimate is None:
        total_flux_estimate = du.estimate_total_flux(obs)

    ###################################################
    # organizing prior information

    # prior info for log gain amplitudes
    loggainamp_mean, loggainamp_std = mu.gain_logamp_prior(obs)

    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,ref_station=ref_station)

    ###################################################
    # setting up the model

    # number of gains
    N_gains = len(loggainamp_mean)

    model = pm.Model()

    with model:

        ###############################################
        # set the priors for the point source parameters
        
        # Stokes I is equal to the total flux
        if fit_total_flux:
            # set to be normal around the correct value, but bounded positive
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
            I = BoundedNormal('I',mu=total_flux_estimate,sd=0.1*total_flux_estimate)
        else:
            # fix at input value
            I = total_flux_estimate

        # permit a centroid shift
        if allow_offset:
            x0 = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0))
            y0 = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0))
        else:
            x0 = 0.0
            y0 = 0.0

        # set the prior on the systematic error term to be uniform on [0,1]
        f = pm.Uniform('f',lower=0.0,upper=1.0)

        ###############################################
        # set the priors for the gain parameters

        # set the gain amplitude priors to be log-normal around the specified inputs
        logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
        g = pm.Deterministic('gain_amps',pm.math.exp(logg))
        
        # set the gain phase priors to be periodic uniform on (-pi,pi)
        theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal_pregain_preshift = I
        Iimag_pregain_preshift = 0.0

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0) + (v*y0))
        Ireal_pregain = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # compute the corruption terms

        gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
        gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

        gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
        gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        ###############################################
        # apply corruptions to the model visibilities

        Ireal_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Ireal_pregain) - (pm.math.sin(gainphase_1 - gainphase_2)*Iimag_pregain))
        Iimag_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Iimag_pregain) + (pm.math.sin(gainphase_1 - gainphase_2)*Ireal_pregain))

        ###############################################
        # add in the systematic noise component

        Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + (f*I)**2.0)
        Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + (f*I)**2.0)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))
        
    ###################################################
    # fit the model

    # NOTE: the current tuning scheme is rather arbitrary
    # and could likely benefit from systematization

    # set up tuning windows
    windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        start = None
        burnin_trace = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH,regularize=regularize)
            burnin_trace = pm.sample(start=start, tune=steps, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            start = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=start, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'point',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'fit_total_flux': fit_total_flux,
                 'total_flux_estimate': total_flux_estimate,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains
                 }

    return modelinfo

def polpoint(obs,total_flux_estimate=None,RLequal=False,fit_StokesV=True,
             fit_total_flux=False,allow_offset=False,offset_window=200.0,
             n_start=25,n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,**kwargs):
    """ Fit a polarized point source model to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data           
           total_flux_estimate (float): estimate of total Stokes I flux (Jy)
           
           RLequal (bool): flag to fix right and left gain terms to be equal
           fit_StokesV (bool): flag to fit for Stokes V; set to False to fix V = 0
           fit_total_flux (bool): flag to fit for the total flux
           allow_offset (bool): flag to permit image centroid to be a free parameter
           offset_window (float): width of square offset window (uas)
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)

    ###################################################
    # data bookkeeping

    # first, make sure we're using a circular representation
    if obs.polrep is not 'circ':
        obs = obs.switch_polrep('circ')

    # (u,v) coordinate info
    u = obs.data['u']
    v = obs.data['v']
    rho = np.sqrt((u**2.0) + (v**2.0))

    # read in the real and imaginary parts for each data product
    RR_real = np.real(obs.data['rrvis'])
    RR_imag = np.imag(obs.data['rrvis'])
    RR_real_err = obs.data['rrsigma']
    RR_imag_err = obs.data['rrsigma']

    LL_real = np.real(obs.data['llvis'])
    LL_imag = np.imag(obs.data['llvis'])
    LL_real_err = obs.data['llsigma']
    LL_imag_err = obs.data['llsigma']

    RL_real = np.real(obs.data['rlvis'])
    RL_imag = np.imag(obs.data['rlvis'])
    RL_real_err = obs.data['rlsigma']
    RL_imag_err = obs.data['rlsigma']

    LR_real = np.real(obs.data['lrvis'])
    LR_imag = np.imag(obs.data['lrvis'])
    LR_real_err = obs.data['lrsigma']
    LR_imag_err = obs.data['lrsigma']

    # mask out any blank data by giving it enormous uncertainties
    mask = ~np.isfinite(obs.data['rrvis'])
    RR_real[mask] = 0.0
    RR_imag[mask] = 0.0
    RR_real_err[mask] = 1000.0
    RR_imag_err[mask] = 1000.0
    mask_RR = np.where(np.isfinite(obs.data['rrvis']))

    mask = ~np.isfinite(obs.data['llvis'])
    LL_real[mask] = 0.0
    LL_imag[mask] = 0.0
    LL_real_err[mask] = 1000.0
    LL_imag_err[mask] = 1000.0
    mask_LL = np.where(np.isfinite(obs.data['llvis']))

    mask = ~np.isfinite(obs.data['rlvis'])
    RL_real[mask] = 0.0
    RL_imag[mask] = 0.0
    RL_real_err[mask] = 1000.0
    RL_imag_err[mask] = 1000.0
    mask_RL = np.where(np.isfinite(obs.data['rlvis']))

    mask = ~np.isfinite(obs.data['lrvis'])
    LR_real[mask] = 0.0
    LR_imag[mask] = 0.0
    LR_real_err[mask] = 1000.0
    LR_imag_err[mask] = 1000.0
    mask_LR = np.where(np.isfinite(obs.data['lrvis']))

    # construct design matrices for gain terms
    T_gains, A_gains = du.gain_account(obs)
    gain_design_mat_1, gain_design_mat_2 = du.gain_design_mats(obs)

    # construct design matrices for leakage terms
    stations, dterm_design_mat_1, dterm_design_mat_2 = du.dterm_design_mats(obs)

    # construct vectors of field rotation corrections
    FR1, FR2 = du.FRvec(obs,ehtim_convention=ehtim_convention)

    # if there's no input total flux estimate, estimate it here
    if total_flux_estimate is None:
        total_flux_estimate = du.estimate_total_flux(obs)

    ###################################################
    # organizing prior information

    # prior info for log gain amplitudes
    loggainamp_mean, loggainamp_std = mu.gain_logamp_prior(obs)
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,ref_station=ref_station)

    ###################################################
    # setting up the model

    # number of gains and dterms
    N_Dterms = dterm_design_mat_1.shape[1]
    N_gains = len(loggainamp_mean)

    model = pm.Model()

    with model:

        ###############################################
        # set the priors for the point source parameters
        
        # Stokes I is equal to the total flux
        if fit_total_flux:
            # set to be normal around the correct value, but bounded positive
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
            I = BoundedNormal('I',mu=total_flux_estimate,sd=0.1*total_flux_estimate)
        else:
            # fix at input value
            I = total_flux_estimate
        
        # sample the polarization fraction uniformly on [0,1]
        p = pm.Uniform('p',lower=0.0,upper=1.0)

        # sample alpha from a periodic uniform distribution
        alpha = pm.VonMises('alpha',mu=0.0,kappa=0.0001)
        EVPA = pm.Deterministic('EVPA',alpha/2.0)

        # circular polarization angle
        if fit_StokesV:
            # sample cos(beta) uniformly on [-1,1]
            cosbeta = pm.Uniform('cosbeta',lower=-1.0,upper=1.0)
        else:
            # fix to be zero
            cosbeta = 0.0
        sinbeta = pm.math.sqrt(1.0 - (cosbeta**2.0))

        # permit a centroid shift
        if allow_offset:
            x0 = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0))
            y0 = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0))
        else:
            x0 = 0.0
            y0 = 0.0

        # set the prior on the systematic error term to be uniform on [0,1]
        f = pm.Uniform('f',lower=0.0,upper=1.0)

        ###############################################
        # set the priors for the gain parameters

        # set the gain amplitude priors to be log-normal around the specified inputs
        logg_R = pm.Normal('right_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
        g_R = pm.Deterministic('right_gain_amps',pm.math.exp(logg_R))

        if RLequal:
            logg_L = pm.Deterministic('left_logg',logg_R)
            g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
        else:
            logg_L = pm.Normal('left_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
            g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
        
        # set the gain phase priors to be periodic uniform on (-pi,pi)
        theta_R = pm.VonMises('right_gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)
        
        if RLequal:
            theta_L = pm.Deterministic('left_gain_phases',theta_R)
        else:
            theta_L = pm.VonMises('left_gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # set the priors for the leakage parameters
        
        # set the D term amplitude priors to be uniform on [0,1]
        Damp_R = pm.Uniform('right_Dterm_amps',lower=0.0,upper=1.0,shape=N_Dterms,testval=0.01)
        logDamp_R = pm.math.log(Damp_R)

        Damp_L = pm.Uniform('left_Dterm_amps',lower=0.0,upper=1.0,shape=N_Dterms,testval=0.01)
        logDamp_L = pm.math.log(Damp_L)

        # set the D term phase priors to be periodic uniform on (-pi,pi)
        delta_R = pm.VonMises('right_Dterm_phases',mu=0.0,kappa=0.0001,shape=N_Dterms)
        delta_L = pm.VonMises('left_Dterm_phases',mu=0.0,kappa=0.0001,shape=N_Dterms)

        # save the real and imaginary parts for output diagnostics
        D_R_real = pm.Deterministic('right_Dterm_reals',Damp_R*pm.math.cos(delta_R))
        D_R_imag = pm.Deterministic('right_Dterm_imags',Damp_R*pm.math.sin(delta_R))
        D_L_real = pm.Deterministic('left_Dterm_reals',Damp_L*pm.math.cos(delta_L))
        D_L_imag = pm.Deterministic('left_Dterm_imags',Damp_L*pm.math.sin(delta_L))

        ###############################################
        # compute the polarized Stokes parameters

        Q = pm.Deterministic('Q',I*p*pm.math.cos(alpha)*sinbeta)
        U = pm.Deterministic('U',I*p*pm.math.sin(alpha)*sinbeta)
        V = pm.Deterministic('V',I*p*cosbeta)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal_preshift = I
        Iimag_preshift = 0.0
        
        Qreal_preshift = Q
        Qimag_preshift = 0.0
        
        Ureal_preshift = U
        Uimag_preshift = 0.0
        
        Vreal_preshift = V
        Vimag_preshift = 0.0

        ###############################################
        # shift centroid
        
        shift_term = 2.0*np.pi*((u*x0) + (v*y0))
        
        Ireal = (Ireal_preshift*pm.math.cos(shift_term)) + (Iimag_preshift*pm.math.sin(shift_term))
        Iimag = (Iimag_preshift*pm.math.cos(shift_term)) - (Ireal_preshift*pm.math.sin(shift_term))
        
        Qreal = (Qreal_preshift*pm.math.cos(shift_term)) + (Qimag_preshift*pm.math.sin(shift_term))
        Qimag = (Qimag_preshift*pm.math.cos(shift_term)) - (Qreal_preshift*pm.math.sin(shift_term))
        
        Ureal = (Ureal_preshift*pm.math.cos(shift_term)) + (Uimag_preshift*pm.math.sin(shift_term))
        Uimag = (Uimag_preshift*pm.math.cos(shift_term)) - (Ureal_preshift*pm.math.sin(shift_term))
        
        Vreal = (Vreal_preshift*pm.math.cos(shift_term)) + (Vimag_preshift*pm.math.sin(shift_term))
        Vimag = (Vimag_preshift*pm.math.cos(shift_term)) - (Vreal_preshift*pm.math.sin(shift_term))
        
        ###############################################
        # construct the pre-corrupted circular basis model visibilities

        RR_real_pregain = Ireal + Vreal
        RR_imag_pregain = Iimag + Vimag

        LL_real_pregain = Ireal - Vreal
        LL_imag_pregain = Iimag - Vimag

        RL_real_pregain = Qreal - Uimag
        RL_imag_pregain = Qimag + Ureal

        LR_real_pregain = Qreal + Uimag
        LR_imag_pregain = Qimag - Ureal

        ###############################################
        # compute the corruption terms
        
        gainamp_R1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg_R))
        gainamp_R2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg_R))
        gainamp_L1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg_L))
        gainamp_L2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg_L))
        
        gainphase_R1 = pm.math.dot(gain_design_mat_1,theta_R)
        gainphase_R2 = pm.math.dot(gain_design_mat_2,theta_R)
        gainphase_L1 = pm.math.dot(gain_design_mat_1,theta_L)
        gainphase_L2 = pm.math.dot(gain_design_mat_2,theta_L)
        
        Damp_R1 = pm.math.exp(pm.math.dot(dterm_design_mat_1,logDamp_R))
        Damp_R2 = pm.math.exp(pm.math.dot(dterm_design_mat_2,logDamp_R))
        Damp_L1 = pm.math.exp(pm.math.dot(dterm_design_mat_1,logDamp_L))
        Damp_L2 = pm.math.exp(pm.math.dot(dterm_design_mat_2,logDamp_L))
        
        Dphase_R1_preFR = pm.math.dot(dterm_design_mat_1,delta_R)
        Dphase_R2_preFR = pm.math.dot(dterm_design_mat_2,delta_R)
        Dphase_L1_preFR = pm.math.dot(dterm_design_mat_1,delta_L)
        Dphase_L2_preFR = pm.math.dot(dterm_design_mat_2,delta_L)
        
        Dphase_R1 = Dphase_R1_preFR + FR1
        Dphase_R2 = Dphase_R2_preFR + FR2
        Dphase_L1 = Dphase_L1_preFR - FR1
        Dphase_L2 = Dphase_L2_preFR - FR2
        
        ###############################################
        # apply corruptions to the model visibilities

        RR_real_model = gainamp_R1*gainamp_R2*((pm.math.cos(gainphase_R1 - gainphase_R2)*RR_real_pregain)
                        - (pm.math.sin(gainphase_R1 - gainphase_R2)*RR_imag_pregain)
                        + (Damp_R1*Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_real_pregain)
                        - (Damp_R1*Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_imag_pregain)
                        + (Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_real_pregain)
                        - (Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_imag_pregain)
                        + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_real_pregain)
                        - (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_imag_pregain))

        RR_imag_model = gainamp_R1*gainamp_R2*((pm.math.sin(gainphase_R1 - gainphase_R2)*RR_real_pregain)
                        + (pm.math.cos(gainphase_R1 - gainphase_R2)*RR_imag_pregain)
                        + (Damp_R1*Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_real_pregain)
                        + (Damp_R1*Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_imag_pregain)
                        + (Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_real_pregain)
                        + (Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_imag_pregain)
                        + (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_real_pregain)
                        + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_imag_pregain))

        LL_real_model = gainamp_L1*gainamp_L2*((Damp_L1*Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_real_pregain)
                        - (Damp_L1*Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_imag_pregain)
                        + (pm.math.cos(gainphase_L1 - gainphase_L2)*LL_real_pregain)
                        - (pm.math.sin(gainphase_L1 - gainphase_L2)*LL_imag_pregain)
                        + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_real_pregain)
                        - (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_imag_pregain)
                        + (Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_real_pregain)
                        - (Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_imag_pregain))

        LL_imag_model = gainamp_L1*gainamp_L2*((Damp_L1*Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_real_pregain)
                        + (Damp_L1*Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_imag_pregain)
                        + (pm.math.sin(gainphase_L1 - gainphase_L2)*LL_real_pregain)
                        + (pm.math.cos(gainphase_L1 - gainphase_L2)*LL_imag_pregain)
                        + (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_real_pregain)
                        + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_imag_pregain)
                        + (Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_real_pregain)
                        + (Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_imag_pregain))

        RL_real_model = gainamp_R1*gainamp_L2*((Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_real_pregain)
                        - (Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_imag_pregain)
                        + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_real_pregain)
                        - (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_imag_pregain)
                        + (pm.math.cos(gainphase_R1 - gainphase_L2)*RL_real_pregain)
                        - (pm.math.sin(gainphase_R1 - gainphase_L2)*RL_imag_pregain)
                        + (Damp_R1*Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_real_pregain)
                        - (Damp_R1*Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_imag_pregain))

        RL_imag_model = gainamp_R1*gainamp_L2*((Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_real_pregain)
                        + (Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_imag_pregain)
                        + (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_real_pregain)
                        + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_imag_pregain)
                        + (pm.math.sin(gainphase_R1 - gainphase_L2)*RL_real_pregain)
                        + (pm.math.cos(gainphase_R1 - gainphase_L2)*RL_imag_pregain)
                        + (Damp_R1*Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_real_pregain)
                        + (Damp_R1*Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_imag_pregain))

        LR_real_model = gainamp_L1*gainamp_R2*((Damp_L1*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_real_pregain)
                        - (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_imag_pregain)
                        + (Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_real_pregain)
                        - (Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_imag_pregain)
                        + (Damp_L1*Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_real_pregain)
                        - (Damp_L1*Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_imag_pregain)
                        + (pm.math.cos(gainphase_L1 - gainphase_R2)*LR_real_pregain)
                        - (pm.math.sin(gainphase_L1 - gainphase_R2)*LR_imag_pregain))

        LR_imag_model = gainamp_L1*gainamp_R2*((Damp_L1*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_real_pregain)
                        + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_imag_pregain)
                        + (Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_real_pregain)
                        + (Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_imag_pregain)
                        + (Damp_L1*Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_real_pregain)
                        + (Damp_L1*Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_imag_pregain)
                        + (pm.math.sin(gainphase_L1 - gainphase_R2)*LR_real_pregain)
                        + (pm.math.cos(gainphase_L1 - gainphase_R2)*LR_imag_pregain))

        ###############################################
        # add in the systematic noise component

        RR_real_err_model = pm.math.sqrt((RR_real_err**2.0) + (f*I)**2.0)
        RR_imag_err_model = pm.math.sqrt((RR_imag_err**2.0) + (f*I)**2.0)

        LL_real_err_model = pm.math.sqrt((LL_real_err**2.0) + (f*I)**2.0)
        LL_imag_err_model = pm.math.sqrt((LL_imag_err**2.0) + (f*I)**2.0)

        RL_real_err_model = pm.math.sqrt((RL_real_err**2.0) + (f*I)**2.0)
        RL_imag_err_model = pm.math.sqrt((RL_imag_err**2.0) + (f*I)**2.0)

        LR_real_err_model = pm.math.sqrt((LR_real_err**2.0) + (f*I)**2.0)
        LR_imag_err_model = pm.math.sqrt((LR_imag_err**2.0) + (f*I)**2.0)

        ###############################################
        # define the likelihood

        L_real_RR = pm.Normal('L_real_RR',mu=RR_real_model[mask_RR],sd=RR_real_err_model[mask_RR],observed=RR_real[mask_RR])
        L_imag_RR = pm.Normal('L_imag_RR',mu=RR_imag_model[mask_RR],sd=RR_imag_err_model[mask_RR],observed=RR_imag[mask_RR])

        L_real_LL = pm.Normal('L_real_LL',mu=LL_real_model[mask_LL],sd=LL_real_err_model[mask_LL],observed=LL_real[mask_LL])
        L_imag_LL = pm.Normal('L_imag_LL',mu=LL_imag_model[mask_LL],sd=LL_imag_err_model[mask_LL],observed=LL_imag[mask_LL])

        L_real_RL = pm.Normal('L_real_RL',mu=RL_real_model[mask_RL],sd=RL_real_err_model[mask_RL],observed=RL_real[mask_RL])
        L_imag_RL = pm.Normal('L_imag_RL',mu=RL_imag_model[mask_RL],sd=RL_imag_err_model[mask_RL],observed=RL_imag[mask_RL])

        L_real_LR = pm.Normal('L_real_LR',mu=LR_real_model[mask_LR],sd=LR_real_err_model[mask_LR],observed=LR_real[mask_LR])
        L_imag_LR = pm.Normal('L_imag_LR',mu=LR_imag_model[mask_LR],sd=LR_imag_err_model[mask_LR],observed=LR_imag[mask_LR])
        
        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr_RR = pm.Deterministic('ssr_RR',pm.math.sum((((RR_real_model[mask_RR]-RR_real[mask_RR])/RR_real_err_model[mask_RR])**2.0) + (((RR_imag_model[mask_RR]-RR_imag[mask_RR])/RR_imag_err_model[mask_RR])**2.0)))
        ssr_LL = pm.Deterministic('ssr_LL',pm.math.sum((((LL_real_model[mask_LL]-LL_real[mask_LL])/LL_real_err_model[mask_LL])**2.0) + (((LL_imag_model[mask_LL]-LL_imag[mask_LL])/LL_imag_err_model[mask_LL])**2.0)))
        ssr_RL = pm.Deterministic('ssr_RL',pm.math.sum((((RL_real_model[mask_RL]-RL_real[mask_RL])/RL_real_err_model[mask_RL])**2.0) + (((RL_imag_model[mask_RL]-RL_imag[mask_RL])/RL_imag_err_model[mask_RL])**2.0)))
        ssr_LR = pm.Deterministic('ssr_LR',pm.math.sum((((LR_real_model[mask_LR]-LR_real[mask_LR])/LR_real_err_model[mask_LR])**2.0) + (((LR_imag_model[mask_LR]-LR_imag[mask_LR])/LR_imag_err_model[mask_LR])**2.0)))
        
    ###################################################
    # fit the model

    # NOTE: the current tuning scheme is rather arbitrary
    # and could likely benefit from systematization

    # set up tuning windows
    windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        start = None
        burnin_trace = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH,regularize=regularize)
            burnin_trace = pm.sample(start=start, tune=steps, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            start = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=start, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'polpoint',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'fit_total_flux': fit_total_flux,
                 'total_flux_estimate': total_flux_estimate,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'RLequal': RLequal,
                 'fit_StokesV': fit_StokesV,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains
                 }

    return modelinfo

