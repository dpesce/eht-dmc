#######################################################
# imports
#######################################################

from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh
import pymc3 as pm
import pickle
import os
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt

from . import data_utils as du
from . import model_utils as mu
from . import plotting as pl

#######################################################
# constants
#######################################################

MAX_TREEDEPTH = 10
EARLY_MAX_TREEDEPTH = 10

#######################################################
# functions
#######################################################

def image(obs,nx,ny,FOVx,FOVy,x0=0.0,y0=0.0,start=None,total_flux_estimate=None,loose_change=False,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,n_start=25,n_burn=500,
          n_tune=5000,ntuning=2000,ntrials=10000,fit_smooth=False,smooth=None,fit_gains=True,
          fit_syserr=True,syserr=None,tuning_windows=None,output_tuning=False,
          gain_amp_prior='normal',dirichlet_weight=1.0,fit_dirichlet_weight=False,
          total_flux_prior=['uniform',0.0,1.0],**kwargs):
    """ Fit a Stokes I image to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           nx (int): number of image pixels in the x-direction
           ny (int): number of image pixels in the y-direction
           FOVx (float): field of view in the x-direction (uas)
           FOVy (float): field of view in the y-direction (uas)
           x0 (float): image center location along x-axis (uas)
           y0 (float): image center location along y-axis (uas)
           
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           offset_window (float): width of square offset window (uas)
    
           loose_change (bool): flag to use the "loose change" noise prescription
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           fit_smooth (bool): flag to fit for the smoothing kernel
           fit_syserr (bool): flag to fit for a multiplicative systematic error component
           allow_offset (bool): flag to permit image centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths

           dirichlet_weight (float): Dirichlet concentration parameter; 1 = flat; <1 = sparse; >1 = smooth
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """
    
    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

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

    # pixel size
    psize_x = FOVx / float(nx)
    psize_y = FOVy / float(ny)

    # one-dimensional pixel vectors
    halFOVx_cent1 = -(FOVx - psize_x)/2.0
    halFOVx_cent2 = (FOVx - psize_x)/2.0
    halFOVy_cent1 = -(FOVy - psize_y)/2.0
    halFOVy_cent2 = (FOVy - psize_y)/2.0
    x_1d = np.linspace(halFOVx_cent1,halFOVx_cent2,nx)
    y_1d = np.linspace(halFOVy_cent1,halFOVy_cent2,ny)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

    # prior info for image pixels
    if (dirichlet_weight != None):
        # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
        dirichlet_weights = dirichlet_weight*np.ones_like(x)

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
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        if (dirichlet_weight != None):
            # Impose a Dirichlet prior on the pixel intensities,
            # with summation constraint equal to the total flux
            if (fit_dirichlet_weight == False):
                pix = pm.Dirichlet('pix',dirichlet_weights)
                I = pm.Deterministic('I',pix*F)
            elif (fit_dirichlet_weight == 'full'):
                a = pm.Uniform('a',lower=0.0,upper=5.0,shape=npix)
                pix = pm.Dirichlet('pix',a=a,shape=npix)
                I = pm.Deterministic('I',pix*F)
            else:
                a = pm.Uniform('a',lower=0.0,upper=5.0,shape=1)
                pix = pm.Dirichlet('pix',a=a*np.ones_like(x),shape=npix)
                I = pm.Deterministic('I',pix*F)
        else:
            pix = pm.Uniform('pix',lower=0.0,upper=1.0,shape=npix)
            I = pm.Deterministic('I',(pix/pm.math.sum(pix))*F)

        # systematic noise prescription
        if loose_change:
            # set the prior on the multiplicative systematic error term to be uniform on [0,1]
            multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=1.0,testval=0.01)

            # set the prior on the additive systematic error term to be uniform on [0,1] Jy
            additive = pm.Uniform('additive',lower=0.0,upper=1.0,testval=0.001)

        else:
            if fit_syserr:
                # set the prior on the systematic error term to be uniform on [0,1]
                f = pm.Uniform('f',lower=0.0,upper=1.0,testval=0.01)
            else:
                if syserr is not None:
                    f = syserr
                else:
                    f = 0.0

        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        # Gaussian smoothing kernel parameters
        if (smooth is not None) & (fit_smooth == False):
            # smoothing width
            sigma = eh.RADPERUAS*(smooth/(2.0*np.sqrt(2.0*np.log(2.0))))
        elif (fit_smooth == True):
            # smoothing width
            sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=100.0)

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal_pregain_preshift_presmooth = pm.math.dot(A_real,I)
        Iimag_pregain_preshift_presmooth = pm.math.dot(A_imag,I)

        ###############################################
        # smooth with the Gaussian kernel
        
        if (smooth is not None) | (fit_smooth == True):
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        else:
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))
        Ireal_pregain = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # compute the corruption terms

        if fit_gains:

            gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
            gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

            gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
            gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        else:
            gainamp_1 = 1.0
            gainamp_2 = 1.0

            gainphase_1 = 0.0
            gainphase_2 = 0.0

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
            Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
            Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + (f*Itot_model)**2.0)
            Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + (f*Itot_model)**2.0)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'image',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'image',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

    return modelinfo

def image_neg(obs,nx,ny,FOVx,FOVy,x0=0.0,y0=0.0,start=None,total_flux_estimate=None,loose_change=False,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,n_start=25,n_burn=500,
          n_tune=5000,ntuning=2000,ntrials=10000,fit_smooth=False,smooth=None,fit_gains=True,
          fit_syserr=True,syserr=None,tuning_windows=None,output_tuning=False,
          gain_amp_prior='normal',dirichlet_weight=1.0,fit_dirichlet_weight=False,
          total_flux_prior=['uniform',0.0,1.0],**kwargs):
    """ Fit a Stokes I image to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           nx (int): number of image pixels in the x-direction
           ny (int): number of image pixels in the y-direction
           FOVx (float): field of view in the x-direction (uas)
           FOVy (float): field of view in the y-direction (uas)
           x0 (float): image center location along x-axis (uas)
           y0 (float): image center location along y-axis (uas)
           
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           offset_window (float): width of square offset window (uas)
    
           loose_change (bool): flag to use the "loose change" noise prescription
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           fit_smooth (bool): flag to fit for the smoothing kernel
           fit_syserr (bool): flag to fit for a multiplicative systematic error component
           allow_offset (bool): flag to permit image centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths

           dirichlet_weight (float): Dirichlet concentration parameter; 1 = flat; <1 = sparse; >1 = smooth
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

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

    # pixel size
    psize_x = FOVx / float(nx)
    psize_y = FOVy / float(ny)

    # one-dimensional pixel vectors
    halFOVx_cent1 = -(FOVx - psize_x)/2.0
    halFOVx_cent2 = (FOVx - psize_x)/2.0
    halFOVy_cent1 = -(FOVy - psize_y)/2.0
    halFOVy_cent2 = (FOVy - psize_y)/2.0
    x_1d = np.linspace(halFOVx_cent1,halFOVx_cent2,nx)
    y_1d = np.linspace(halFOVy_cent1,halFOVy_cent2,ny)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

    # prior info for image pixels
    if (dirichlet_weight != None):
        # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
        dirichlet_weights = dirichlet_weight*np.ones_like(x)

    ###################################################
    # setting up the model

    # number of gains
    N_gains = len(loggainamp_mean)

    model = pm.Model()

    with model:

        ###############################################
        # set the priors for the image parameters
        
        # # total flux prior
        # if fit_total_flux:
        #     if 'uniform' in total_flux_prior:
        #         F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
        #     else:
        #         # set to be normal around the correct value, but bounded positive
        #         BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        #         F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        # else:
        #     # fix at input value
        #     F = total_flux_estimate

        # if (dirichlet_weight != None):
        #     # Impose a Dirichlet prior on the pixel intensities,
        #     # with summation constraint equal to the total flux
        #     if (fit_dirichlet_weight == False):
        #         pix = pm.Dirichlet('pix',dirichlet_weights)
        #         I = pm.Deterministic('I',pix*F)
        #     elif (fit_dirichlet_weight == 'full'):
        #         a = pm.Uniform('a',lower=0.0,upper=5.0,shape=npix)
        #         pix = pm.Dirichlet('pix',a=a,shape=npix)
        #         I = pm.Deterministic('I',pix*F)
        #     else:
        #         a = pm.Uniform('a',lower=0.0,upper=5.0,shape=1)
        #         pix = pm.Dirichlet('pix',a=a*np.ones_like(x),shape=npix)
        #         I = pm.Deterministic('I',pix*F)
        # else:
        #     pix = pm.Uniform('pix',lower=0.0,upper=1.0,shape=npix)
        #     I = pm.Deterministic('I',(pix/pm.math.sum(pix))*F)

        I = pm.Normal('I',mu=0.0,sd=0.1,shape=npix)

        # systematic noise prescription
        if loose_change:
            # set the prior on the multiplicative systematic error term to be uniform on [0,1]
            multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=1.0,testval=0.01)

            # set the prior on the additive systematic error term to be uniform on [0,1] Jy
            additive = pm.Uniform('additive',lower=0.0,upper=1.0,testval=0.001)

        else:
            if fit_syserr:
                # set the prior on the systematic error term to be uniform on [0,1]
                f = pm.Uniform('f',lower=0.0,upper=1.0,testval=0.01)
            else:
                if syserr is not None:
                    f = syserr
                else:
                    f = 0.0

        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        # Gaussian smoothing kernel parameters
        if (smooth is not None) & (fit_smooth == False):
            # smoothing width
            sigma = eh.RADPERUAS*(smooth/(2.0*np.sqrt(2.0*np.log(2.0))))
        elif (fit_smooth == True):
            # smoothing width
            sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=100.0)

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal_pregain_preshift_presmooth = pm.math.dot(A_real,I)
        Iimag_pregain_preshift_presmooth = pm.math.dot(A_imag,I)

        ###############################################
        # smooth with the Gaussian kernel
        
        if (smooth is not None) | (fit_smooth == True):
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        else:
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))
        Ireal_pregain = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # compute the corruption terms

        if fit_gains:

            gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
            gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

            gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
            gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        else:
            gainamp_1 = 1.0
            gainamp_2 = 1.0

            gainphase_1 = 0.0
            gainphase_2 = 0.0

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
            Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
            Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + (f*Itot_model)**2.0)
            Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + (f*Itot_model)**2.0)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'image',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'image',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

    return modelinfo

def limage(obs,nx,ny,FOVx,FOVy,x0=0.0,y0=0.0,N=50,start=None,total_flux_estimate=None,loose_change=False,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,n_start=25,n_burn=500,
          n_tune=5000,ntuning=2000,ntrials=10000,fit_smooth=False,smooth=None,fit_gains=True,
          fit_syserr=True,syserr=None,tuning_windows=None,output_tuning=False,
          gain_amp_prior='normal',dirichlet_weight=1.0,fit_dirichlet_weight=False,
          total_flux_prior=['uniform',0.0,1.0],limacon_flux_prior=[0.0,1.0],
          lambda1_prior=[0.0,50.0],lambda2_prior=[0.0,0.5],**kwargs):
    """ Fit a Stokes I image + limacon to a VLBI observation
        
       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           nx (int): number of image pixels in the x-direction
           ny (int): number of image pixels in the y-direction
           FOVx (float): field of view in the x-direction (uas)
           FOVy (float): field of view in the y-direction (uas)
           x0 (float): image center location along x-axis (uas)
           y0 (float): image center location along y-axis (uas)
           N (int): number of points to use to approximate the limacon
           
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           offset_window (float): width of square offset window (uas)
    
           loose_change (bool): flag to use the "loose change" noise prescription
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           fit_smooth (bool): flag to fit for the smoothing kernel
           fit_syserr (bool): flag to fit for a multiplicative systematic error component
           allow_offset (bool): flag to permit image centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths

           dirichlet_weight (float): Dirichlet concentration parameter; 1 = flat; <1 = sparse; >1 = smooth
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

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

    # pixel size
    psize_x = FOVx / float(nx)
    psize_y = FOVy / float(ny)

    # one-dimensional pixel vectors
    halFOVx_cent1 = -(FOVx - psize_x)/2.0
    halFOVx_cent2 = (FOVx - psize_x)/2.0
    halFOVy_cent1 = -(FOVy - psize_y)/2.0
    halFOVy_cent2 = (FOVy - psize_y)/2.0
    x_1d = np.linspace(halFOVx_cent1,halFOVx_cent2,nx)
    y_1d = np.linspace(halFOVy_cent1,halFOVy_cent2,ny)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

    # prior info for image pixels
    if (dirichlet_weight != None):
        # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
        dirichlet_weights = dirichlet_weight*np.ones_like(x)

    ###################################################
    # organizing limacon information

    # azimuthal coordinate locations
    phi_lim = np.arange(0.0,2.0*np.pi,2.0*np.pi/float(N))
    cosphi_lim = np.cos(phi_lim)
    sinphi_lim = np.sin(phi_lim)

    # share the (u,v) coordinates
    u_lim = theano.shared(u)
    v_lim = theano.shared(v)

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
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        if (dirichlet_weight != None):
            # Impose a Dirichlet prior on the pixel intensities,
            # with summation constraint equal to the total flux
            if (fit_dirichlet_weight == False):
                pix = pm.Dirichlet('pix',dirichlet_weights)
                I = pm.Deterministic('I',pix*F)
            elif (fit_dirichlet_weight == 'full'):
                a = pm.Uniform('a',lower=0.0,upper=5.0,shape=npix)
                pix = pm.Dirichlet('pix',a=a,shape=npix)
                I = pm.Deterministic('I',pix*F)
            else:
                a = pm.Uniform('a',lower=0.0,upper=5.0,shape=1)
                pix = pm.Dirichlet('pix',a=a*np.ones_like(x),shape=npix)
                I = pm.Deterministic('I',pix*F)
        else:
            pix = pm.Uniform('pix',lower=0.0,upper=1.0,shape=npix)
            I = pm.Deterministic('I',(pix/pm.math.sum(pix))*F)

        # systematic noise prescription
        if loose_change:
            # set the prior on the multiplicative systematic error term to be uniform on [0,1]
            multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=1.0,testval=0.01)

            # set the prior on the additive systematic error term to be uniform on [0,1] Jy
            additive = pm.Uniform('additive',lower=0.0,upper=1.0,testval=0.001)

        else:
            if fit_syserr:
                # set the prior on the systematic error term to be uniform on [0,1]
                f = pm.Uniform('f',lower=0.0,upper=1.0,testval=0.01)
            else:
                if syserr is not None:
                    f = syserr
                else:
                    f = 0.0

        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        # Gaussian smoothing kernel parameters
        if (smooth is not None) & (fit_smooth == False):
            # smoothing width
            sigma = eh.RADPERUAS*(smooth/(2.0*np.sqrt(2.0*np.log(2.0))))
        elif (fit_smooth == True):
            # smoothing width
            sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=100.0)

        ###############################################
        # set the priors for the limacon parameters

        # limacon flux
        F_lim = pm.Uniform('F_lim',limacon_flux_prior[0],limacon_flux_prior[1])

        # limacon size parameter
        lambda1 = eh.RADPERUAS*pm.Uniform('lam1',lambda1_prior[0],lambda1_prior[1])

        # limacon shape parameter
        lambda2 = pm.Uniform('lam2',lambda2_prior[0],lambda2_prior[1])

        # limacon orientation parameter
        phi = pm.VonMises('phi',mu=0.0,kappa=0.0001)

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal_pregain_preshift_presmooth = pm.math.dot(A_real,I)
        Iimag_pregain_preshift_presmooth = pm.math.dot(A_imag,I)

        ###############################################
        # smooth with the Gaussian kernel
        
        if (smooth is not None) | (fit_smooth == True):
            Ireal_pregain_preshift_prelim = Ireal_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Iimag_pregain_preshift_prelim = Iimag_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        else:
            Ireal_pregain_preshift_prelim = Ireal_pregain_preshift_presmooth
            Iimag_pregain_preshift_prelim = Iimag_pregain_preshift_presmooth

        ###############################################
        # add in limacon

        # determine point coordinates
        R = lambda1*(1.0 + (lambda2*cosphi_lim))
        x_lim_prerot = R*cosphi_lim
        y_lim_prerot = R*sinphi_lim

        # rotate
        x_lim = (x_lim_prerot*pm.math.cos(phi)) - (y_lim_prerot*pm.math.sin(phi))
        y_lim = (x_lim_prerot*pm.math.sin(phi)) + (y_lim_prerot*pm.math.cos(phi))

        # get point intensities
        I_lim = (F_lim*R)/pm.math.sum(R)

        # construct FT matrix
        matrix_lim = tt.outer(u_lim,x_lim) + tt.outer(v_lim,y_lim)
        A_real_lim = pm.math.cos(2.0*np.pi*matrix_lim)
        A_imag_lim = pm.math.sin(2.0*np.pi*matrix_lim)

        # compute visibilities
        Ireal_lim = pm.math.dot(A_real_lim,I_lim)
        Iimag_lim = pm.math.dot(A_imag_lim,I_lim)

        # add to image visibilities
        Ireal_pregain_preshift = Ireal_pregain_preshift_prelim + Ireal_lim
        Iimag_pregain_preshift = Iimag_pregain_preshift_prelim + Iimag_lim
        
        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))
        Ireal_pregain = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # compute the corruption terms

        if fit_gains:

            gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
            gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

            gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
            gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        else:
            gainamp_1 = 1.0
            gainamp_2 = 1.0

            gainphase_1 = 0.0
            gainphase_2 = 0.0

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
            Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
            Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + (f*Itot_model)**2.0)
            Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + (f*Itot_model)**2.0)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'limage',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'N': N,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'limage',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'N': N,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

    return modelinfo

def limacon(obs,x0=0.0,y0=0.0,N=50,start=None,total_flux_estimate=None,loose_change=False,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,n_start=25,n_burn=500,
          n_tune=5000,ntuning=2000,ntrials=10000,fit_gains=True,
          fit_syserr=True,syserr=None,tuning_windows=None,output_tuning=False,
          gain_amp_prior='normal',total_flux_prior=['uniform',0.0,1.0],**kwargs):
    """ Fit a Stokes I image + limacon to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           x0 (float): image center location along x-axis (uas)
           y0 (float): image center location along y-axis (uas)
           N (int): number of points to use to approximate the limacon
           
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           offset_window (float): width of square offset window (uas)
    
           loose_change (bool): flag to use the "loose change" noise prescription
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           fit_syserr (bool): flag to fit for a multiplicative systematic error component
           allow_offset (bool): flag to permit image centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths
                       
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

    ###################################################
    # organizing limacon information

    # azimuthal coordinate locations
    phi_lim = np.arange(0.0,2.0*np.pi,2.0*np.pi/float(N))
    cosphi_lim = np.cos(phi_lim)
    sinphi_lim = np.sin(phi_lim)

    # share the (u,v) coordinates
    u_lim = theano.shared(u)
    v_lim = theano.shared(v)

    ###################################################
    # setting up the model

    # number of gains
    N_gains = len(loggainamp_mean)

    model = pm.Model()

    with model:

        ###############################################
        # set the priors
        
        # total flux prior
        if fit_total_flux:
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        # limacon size parameter
        lambda1 = eh.RADPERUAS*pm.Uniform('lam1',0.0,50.0)

        # limacon shape parameter
        lambda2 = pm.Uniform('lam2',0.0,0.5)

        # limacon orientation parameter
        phi = pm.VonMises('phi',mu=0.0,kappa=0.0001)

        # systematic noise prescription
        if loose_change:
            # set the prior on the multiplicative systematic error term to be uniform on [0,1]
            multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=1.0,testval=0.01)

            # set the prior on the additive systematic error term to be uniform on [0,1] Jy
            additive = pm.Uniform('additive',lower=0.0,upper=1.0,testval=0.001)

        else:
            if fit_syserr:
                # set the prior on the systematic error term to be uniform on [0,1]
                f = pm.Uniform('f',lower=0.0,upper=1.0,testval=0.01)
            else:
                if syserr is not None:
                    f = syserr
                else:
                    f = 0.0

        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # add in limacon
        
        # determine point coordinates
        R = lambda1*(1.0 + (lambda2*cosphi_lim))
        x_lim_prerot = R*cosphi_lim
        y_lim_prerot = R*sinphi_lim

        # rotate
        x_lim = (x_lim_prerot*pm.math.cos(phi)) - (y_lim_prerot*pm.math.sin(phi))
        y_lim = (x_lim_prerot*pm.math.sin(phi)) + (y_lim_prerot*pm.math.cos(phi))
        
        # get point intensities
        I_lim = (F*R)/pm.math.sum(R)
        
        # construct FT matrix
        matrix_lim = tt.outer(u_lim,x_lim) + tt.outer(v_lim,y_lim)
        A_real_lim = pm.math.cos(2.0*np.pi*matrix_lim)
        A_imag_lim = pm.math.sin(2.0*np.pi*matrix_lim)
        
        # compute visibilities
        Ireal_pregain_preshift = pm.math.dot(A_real_lim,I_lim)
        Iimag_pregain_preshift = pm.math.dot(A_imag_lim,I_lim)
        
        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))
        Ireal_pregain = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # compute the corruption terms

        if fit_gains:

            gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
            gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

            gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
            gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        else:
            gainamp_1 = 1.0
            gainamp_2 = 1.0

            gainphase_1 = 0.0
            gainphase_2 = 0.0
        
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
            Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
            Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + (f*Itot_model)**2.0)
            Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + (f*Itot_model)**2.0)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'limacon',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'x0': x0,
                 'y0': y0,
                 'N': N,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'limage',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'x0': x0,
                 'y0': y0,
                 'N': N,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag
                 }

    return modelinfo

def polimage(obs,nx,ny,FOVx,FOVy,x0=0.0,y0=0.0,start=None,total_flux_estimate=None,RLequal=False,
          fit_StokesV=True,fit_total_flux=False,allow_offset=False,offset_window=200.0,
          smooth=None,n_start=25,n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,
          gain_amp_prior='normal',const_ref_RL=True,fit_gains=True,fit_leakages=True,
          fit_smooth=False,fit_syserr=True,syserr=None,tuning_windows=None,output_tuning=False,
          dirichlet_weight=1.0,fit_dirichlet_weight=False,total_flux_prior=['uniform',0.0,1.0],**kwargs):
    """ Fit a polarimetric image to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           nx (int): number of image pixels in the x-direction
           ny (int): number of image pixels in the y-direction
           FOVx (float): field of view in the x-direction (uas)
           FOVy (float): field of view in the y-direction (uas)
           x0 (float): image center location along x-axis (uas)
           y0 (float): image center location along y-axis (uas)
           
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           offset_window (float): width of square offset window (uas)
           smooth (float): smoothing kernel FWHM (uas)
           syserr (float): fractional systematic error
           
           RLequal (bool): flag to fix right and left gain terms to be equal
           fit_StokesV (bool): flag to fit for Stokes V; set to False to fix V = 0
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           fit_leakages (bool): flag to fit for the complex leakages
           fit_smooth (bool): flag to fit for the smoothing kernel
           fit_syserr (bool): flag to fit for a multiplicative systematic error component
           allow_offset (bool): flag to permit image centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
           
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths

           dirichlet_weight (float): Dirichlet concentration parameter; 1 = flat; <1 = sparse; >1 = smooth
           
       Returns:
           modelinfo: a dictionary containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors_R = kwargs.get('gain_phase_priors_R',(0.0,0.0001))
    gain_phase_priors_L = kwargs.get('gain_phase_priors_L',(0.0,0.0001))
    dterm_amp_priors = kwargs.get('dterm_amp_priors',(0.0,1.0))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

    ###################################################
    # data bookkeeping

    # first, make sure we're using a circular representation
    if obs.polrep != 'circ':
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

    # pixel size
    psize_x = FOVx / float(nx)
    psize_y = FOVy / float(ny)

    # one-dimensional pixel vectors
    halFOVx_cent1 = -(FOVx - psize_x)/2.0
    halFOVx_cent2 = (FOVx - psize_x)/2.0
    halFOVy_cent1 = -(FOVy - psize_y)/2.0
    halFOVy_cent2 = (FOVy - psize_y)/2.0
    x_1d = np.linspace(halFOVx_cent1,halFOVx_cent2,nx)
    y_1d = np.linspace(halFOVy_cent1,halFOVy_cent2,ny)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu_R, gainphase_kappa_R = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors_R)
    gainphase_mu_L, gainphase_kappa_L = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors_L)
    
    if const_ref_RL:
        if ref_station is not None:
            ind_ref = (A_gains == ref_station)
            gainphase_kappa_L[ind_ref] = 10000.0
    else:
        if ref_station is not None:
            ind_ref = (A_gains == ref_station)
            gainphase_kappa_temp = np.copy(gainphase_kappa_L[ind_ref])
            gainphase_kappa_temp[0] = 10000.0
            gainphase_kappa_L[ind_ref] = gainphase_kappa_temp

    # prior info for leakage amplitudes
    dtermamp_lo, dtermamp_hi = mu.dterm_amp_prior(obs,dterm_amp_priors=dterm_amp_priors)

    # prior info for image pixels
    if (dirichlet_weight != None):
        # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
        dirichlet_weights = dirichlet_weight*np.ones_like(x)

    ###################################################
    # setting up the model

    # number of gains and dterms
    N_Dterms = dterm_design_mat_1.shape[1]
    N_gains = len(gainamp_mean)

    model = pm.Model()

    with model:

        ###############################################
        # set the priors for the image parameters
        
        # total flux prior
        if fit_total_flux:
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        if (dirichlet_weight != None):
            # Impose a Dirichlet prior on the pixel intensities,
            # with summation constraint equal to the total flux
            if (fit_dirichlet_weight == False):
                pix = pm.Dirichlet('pix',dirichlet_weights)
                I = pm.Deterministic('I',pix*F)
            elif (fit_dirichlet_weight == 'full'):
                a = pm.Uniform('a',lower=0.0,upper=5.0,shape=npix)
                pix = pm.Dirichlet('pix',a=a,shape=npix)
                I = pm.Deterministic('I',pix*F)
            else:
                a = pm.Uniform('a',lower=0.0,upper=5.0,shape=1)
                pix = pm.Dirichlet('pix',a=a*np.ones_like(x),shape=npix)
                I = pm.Deterministic('I',pix*F)
        else:
            pix = pm.Uniform('pix',lower=0.0,upper=1.0,shape=npix)
            I = pm.Deterministic('I',(pix/pm.math.sum(pix))*F)
        
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

        if fit_syserr:
            # set the prior on the systematic error term to be uniform on [0,1]
            f = pm.Uniform('f',lower=0.0,upper=1.0,testval=0.01)
        else:
            if syserr is not None:
                f = syserr
            else:
                f = 0.0

        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        # Gaussian smoothing kernel parameters
        if (smooth is not None) & (fit_smooth == False):
            # smoothing width
            sigma = eh.RADPERUAS*(smooth/(2.0*np.sqrt(2.0*np.log(2.0))))
        elif (fit_smooth == True):
            # smoothing width
            sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=100.0)

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg_R = pm.Normal('right_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g_R = pm.Deterministic('right_gain_amps',pm.math.exp(logg_R))

                if RLequal:
                    logg_L = pm.Deterministic('left_logg',logg_R)
                    g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
                else:
                    logg_L = pm.Normal('left_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                    g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g_R = BoundedNormal('right_gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg_R = pm.Deterministic('right_logg',pm.math.log(g_R))

                if RLequal:
                    logg_L = pm.Deterministic('left_logg',logg_R)
                    g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
                else:
                    g_L = BoundedNormal('left_gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                    logg_L = pm.Deterministic('left_logg',pm.math.log(g_L))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta_R = pm.VonMises('right_gain_phases',mu=gainphase_mu_R,kappa=gainphase_kappa_R,shape=N_gains)
            
            if RLequal:
                theta_L = pm.Deterministic('left_gain_phases',theta_R)
            else:
                theta_L = pm.VonMises('left_gain_phases',mu=gainphase_mu_L,kappa=gainphase_kappa_L,shape=N_gains)
            
        ###############################################
        # set the priors for the leakage parameters
        
        if fit_leakages:

            # set the D term amplitude priors to be uniform on [0,1]
            Damp_R = pm.Uniform('right_Dterm_amps',lower=dtermamp_lo,upper=dtermamp_hi,shape=N_Dterms)
            logDamp_R = pm.math.log(Damp_R)

            Damp_L = pm.Uniform('left_Dterm_amps',lower=dtermamp_lo,upper=dtermamp_hi,shape=N_Dterms)
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
        
        Ireal_presmooth = pm.math.dot(A_real,I)
        Iimag_presmooth = pm.math.dot(A_imag,I)
        
        Qreal_presmooth = pm.math.dot(A_real,Q)
        Qimag_presmooth = pm.math.dot(A_imag,Q)
        
        Ureal_presmooth = pm.math.dot(A_real,U)
        Uimag_presmooth = pm.math.dot(A_imag,U)
        
        Vreal_presmooth = pm.math.dot(A_real,V)
        Vimag_presmooth = pm.math.dot(A_imag,V)

        ###############################################
        # smooth with the Gaussian kernel
        
        if (smooth is not None) | (fit_smooth == True):
            Ireal = Ireal_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Iimag = Iimag_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))

            Qreal = Qreal_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Qimag = Qimag_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))

            Ureal = Ureal_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Uimag = Uimag_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))

            Vreal = Vreal_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Vimag = Vimag_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        else:
            Ireal = Ireal_presmooth
            Iimag = Iimag_presmooth

            Qreal = Qreal_presmooth
            Qimag = Qimag_presmooth

            Ureal = Ureal_presmooth
            Uimag = Uimag_presmooth

            Vreal = Vreal_presmooth
            Vimag = Vimag_presmooth

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))

        Ireal_pregain = (Ireal*pm.math.cos(shift_term)) + (Iimag*pm.math.sin(shift_term))
        Iimag_pregain = (Iimag*pm.math.cos(shift_term)) - (Ireal*pm.math.sin(shift_term))

        Qreal_pregain = (Qreal*pm.math.cos(shift_term)) + (Qimag*pm.math.sin(shift_term))
        Qimag_pregain = (Qimag*pm.math.cos(shift_term)) - (Qreal*pm.math.sin(shift_term))

        Ureal_pregain = (Ureal*pm.math.cos(shift_term)) + (Uimag*pm.math.sin(shift_term))
        Uimag_pregain = (Uimag*pm.math.cos(shift_term)) - (Ureal*pm.math.sin(shift_term))

        Vreal_pregain = (Vreal*pm.math.cos(shift_term)) + (Vimag*pm.math.sin(shift_term))
        Vimag_pregain = (Vimag*pm.math.cos(shift_term)) - (Vreal*pm.math.sin(shift_term))

        ###############################################
        # construct the pre-corrupted circular basis model visibilities

        RR_real_pregain = Ireal_pregain + Vreal_pregain
        RR_imag_pregain = Iimag_pregain + Vimag_pregain
        
        LL_real_pregain = Ireal_pregain - Vreal_pregain
        LL_imag_pregain = Iimag_pregain - Vimag_pregain

        RL_real_pregain = Qreal_pregain - Uimag_pregain
        RL_imag_pregain = Qimag_pregain + Ureal_pregain

        LR_real_pregain = Qreal_pregain + Uimag_pregain
        LR_imag_pregain = Qimag_pregain - Ureal_pregain

        ###############################################
        # compute the corruption terms
        
        if fit_gains:
            gainamp_R1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg_R))
            gainamp_R2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg_R))
            gainamp_L1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg_L))
            gainamp_L2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg_L))

            gainphase_R1 = pm.math.dot(gain_design_mat_1,theta_R)
            gainphase_R2 = pm.math.dot(gain_design_mat_2,theta_R)
            gainphase_L1 = pm.math.dot(gain_design_mat_1,theta_L)
            gainphase_L2 = pm.math.dot(gain_design_mat_2,theta_L)

        else:
            gainamp_R1 = 1.0
            gainamp_R2 = 1.0
            gainamp_L1 = 1.0
            gainamp_L2 = 1.0

            gainphase_R1 = 0.0
            gainphase_R2 = 0.0
            gainphase_L1 = 0.0
            gainphase_L2 = 0.0
        
        if fit_leakages:
            Damp_R1 = pm.math.exp(pm.math.dot(dterm_design_mat_1,logDamp_R))
            Damp_R2 = pm.math.exp(pm.math.dot(dterm_design_mat_2,logDamp_R))
            Damp_L1 = pm.math.exp(pm.math.dot(dterm_design_mat_1,logDamp_L))
            Damp_L2 = pm.math.exp(pm.math.dot(dterm_design_mat_2,logDamp_L))
            
            Dphase_R1_preFR = pm.math.dot(dterm_design_mat_1,delta_R)
            Dphase_R2_preFR = pm.math.dot(dterm_design_mat_2,delta_R)
            Dphase_L1_preFR = pm.math.dot(dterm_design_mat_1,delta_L)
            Dphase_L2_preFR = pm.math.dot(dterm_design_mat_2,delta_L)

        else:
            Damp_R1 = 0.0
            Damp_R2 = 0.0
            Damp_L1 = 0.0
            Damp_L2 = 0.0
            
            Dphase_R1_preFR = 0.0
            Dphase_R2_preFR = 0.0
            Dphase_L1_preFR = 0.0
            Dphase_L2_preFR = 0.0
            
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

        Itot_model = pm.math.sqrt(((0.5*(RR_real_model+LL_real_model))**2.0) + ((0.5*(RR_imag_model+LL_imag_model))**2.0))
        
        RR_real_err_model = pm.math.sqrt((RR_real_err**2.0) + (f*Itot_model)**2.0)
        RR_imag_err_model = pm.math.sqrt((RR_imag_err**2.0) + (f*Itot_model)**2.0)

        LL_real_err_model = pm.math.sqrt((LL_real_err**2.0) + (f*Itot_model)**2.0)
        LL_imag_err_model = pm.math.sqrt((LL_imag_err**2.0) + (f*Itot_model)**2.0)

        RL_real_err_model = pm.math.sqrt((RL_real_err**2.0) + (f*Itot_model)**2.0)
        RL_imag_err_model = pm.math.sqrt((RL_imag_err**2.0) + (f*Itot_model)**2.0)

        LR_real_err_model = pm.math.sqrt((LR_real_err**2.0) + (f*Itot_model)**2.0)
        LR_imag_err_model = pm.math.sqrt((LR_imag_err**2.0) + (f*Itot_model)**2.0)

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

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'polimage',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'total_flux_estimate': total_flux_estimate,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'RLequal': RLequal,
                 'fit_StokesV': fit_StokesV,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'smooth': smooth,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_leakages': fit_leakages,
                 'fit_smooth': fit_smooth,
                 'fit_syserr': fit_syserr,
                 'syserr': syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'polimage',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'total_flux_estimate': total_flux_estimate,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'RLequal': RLequal,
                 'fit_StokesV': fit_StokesV,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'smooth': smooth,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_leakages': fit_leakages,
                 'fit_smooth': fit_smooth,
                 'fit_syserr': fit_syserr,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

    return modelinfo

def image_noisy(obs,nx,ny,FOVx,FOVy,LC=None,x0=0.0,y0=0.0,start=None,total_flux_estimate=None,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,n_start=25,
          n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,fit_smooth=False,
          smooth=None,fit_gains=True,tuning_windows=None,output_tuning=False,
          gain_amp_prior='normal',dirichlet_weight=1.0,fit_dirichlet_weight=False,
          total_flux_prior=['uniform',0.0,1.0],**kwargs):
    """ Fit a Stokes I image to a VLBI observation
        
       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           nx (int): number of image pixels in the x-direction
           ny (int): number of image pixels in the y-direction
           FOVx (float): field of view in the x-direction (uas)
           FOVy (float): field of view in the y-direction (uas)
           x0 (float): image center location along x-axis (uas)
           y0 (float): image center location along y-axis (uas)
           LC (array): light curve
           
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           offset_window (float): width of square offset window (uas)
               
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           fit_smooth (bool): flag to fit for the smoothing kernel
           allow_offset (bool): flag to permit image centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths

           dirichlet_weight (float): Dirichlet concentration parameter; 1 = flat; <1 = sparse; >1 = smooth
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

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

    # pixel size
    psize_x = FOVx / float(nx)
    psize_y = FOVy / float(ny)

    # one-dimensional pixel vectors
    halFOVx_cent1 = -(FOVx - psize_x)/2.0
    halFOVx_cent2 = (FOVx - psize_x)/2.0
    halFOVy_cent1 = -(FOVy - psize_y)/2.0
    halFOVy_cent2 = (FOVy - psize_y)/2.0
    x_1d = np.linspace(halFOVx_cent1,halFOVx_cent2,nx)
    y_1d = np.linspace(halFOVy_cent1,halFOVy_cent2,ny)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

    # prior info for image pixels
    if (dirichlet_weight != None):
        # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
        dirichlet_weights = dirichlet_weight*np.ones_like(x)

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
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        if (dirichlet_weight != None):
            # Impose a Dirichlet prior on the pixel intensities,
            # with summation constraint equal to the total flux
            if (fit_dirichlet_weight == False):
                pix = pm.Dirichlet('pix',dirichlet_weights)
                I = pm.Deterministic('I',pix*F)
            elif (fit_dirichlet_weight == 'full'):
                a = pm.Uniform('a',lower=0.0,upper=5.0,shape=npix)
                pix = pm.Dirichlet('pix',a=a,shape=npix)
                I = pm.Deterministic('I',pix*F)
            else:
                a = pm.Uniform('a',lower=0.0,upper=5.0,shape=1)
                pix = pm.Dirichlet('pix',a=a*np.ones_like(x),shape=npix)
                I = pm.Deterministic('I',pix*F)
        else:
            pix = pm.Uniform('pix',lower=0.0,upper=1.0,shape=npix)
            I = pm.Deterministic('I',(pix/pm.math.sum(pix))*F)

        # systematic noise prescription
        # set the prior on the multiplicative systematic error term to be uniform on [0,1]
        multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=0.05,testval=0.01)

        # set the prior on the additive systematic error term to be uniform on [0,1] Jy
        additive = pm.Uniform('additive',lower=0.0,upper=0.05,testval=0.001)
        
        # set the power law error term priors
        logn0 = pm.Uniform('logn0',lower=-10.0,upper=-2.0)
        n0 = pm.Deterministic('n0',pm.math.exp(logn0))
        umax = pm.Uniform('umax',lower=1.0e9,upper=5.0e9)
        b = pm.Uniform('b',lower=0.0,upper=5.0)
        c = pm.Uniform('c',lower=0.0,upper=3.0)

        # multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=0.001)
        # additive = pm.Uniform('additive',lower=0.0,upper=0.001)
        # logn0 = pm.Uniform('logn0',lower=-3.5,upper=-3.4)
        # n0 = pm.Deterministic('n0',pm.math.exp(logn0))
        # umax = pm.Uniform('umax',lower=3.65e9,upper=3.7e9)
        # b = pm.Uniform('b',lower=2.65,upper=2.7)
        # c = pm.Uniform('c',lower=1.2,upper=1.25)
        
        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        # Gaussian smoothing kernel parameters
        if (smooth is not None) & (fit_smooth == False):
            # smoothing width
            sigma = eh.RADPERUAS*(smooth/(2.0*np.sqrt(2.0*np.log(2.0))))
        elif (fit_smooth == True):
            # smoothing width
            sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=100.0)

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal_pregain_preshift_presmooth = pm.math.dot(A_real,I)
        Iimag_pregain_preshift_presmooth = pm.math.dot(A_imag,I)

        ###############################################
        # smooth with the Gaussian kernel
        
        if (smooth is not None) | (fit_smooth == True):
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        else:
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))
        Ireal_pregain_preLC = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain_preLC = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # multiply by light curve

        if (LC is not None):
            Ireal_pregain = Ireal_pregain_preLC*LC
            Iimag_pregain = Iimag_pregain_preLC*LC
        else:
            Ireal_pregain = Ireal_pregain_preLC*1.0
            Iimag_pregain = Iimag_pregain_preLC*1.0

        ###############################################
        # compute the corruption terms

        if fit_gains:

            gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
            gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

            gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
            gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        else:
            gainamp_1 = 1.0
            gainamp_2 = 1.0

            gainphase_1 = 0.0
            gainphase_2 = 0.0

        ###############################################
        # apply corruptions to the model visibilities

        Ireal_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Ireal_pregain) - (pm.math.sin(gainphase_1 - gainphase_2)*Iimag_pregain))
        Iimag_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Iimag_pregain) + (pm.math.sin(gainphase_1 - gainphase_2)*Ireal_pregain))

        ###############################################
        # add in the systematic noise component

        Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
        err2 = ((multiplicative*Itot_model)**2.0) + (additive**2.0) + ((n0**2.0)*((rho/umax)**c) / (1.0 + ((rho/umax)**(b+c))))

        Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + err2)
        Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + err2)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            # if istep == 0:
            #     step = mu.get_step_for_trace(None,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            # else:
            #     step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'image',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'image',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'nx': nx,
                 'ny': ny,
                 'FOVx': FOVx,
                 'FOVy': FOVy,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 'dirichlet_weight': dirichlet_weight,
                 'fit_dirichlet_weight': fit_dirichlet_weight
                 }

    return modelinfo

def crescent(obs,x0=0.0,y0=0.0,start=None,total_flux_estimate=None,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,n_start=25,
          n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,fit_smooth=False,
          smooth=None,fit_gains=True,tuning_windows=None,output_tuning=False,
          gain_amp_prior='normal',total_flux_prior=['uniform',0.0,10.0],
          fit_syserr=True,syserr=None,**kwargs):
    """ Fit a Stokes I crescent model to a VLBI observation
        
       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           x0 (float): crescent center location along x-axis (uas)
           y0 (float): crescent center location along y-axis (uas)
                      
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I crescent flux (Jy)
           offset_window (float): width of square offset window (uas)
               
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           fit_smooth (bool): flag to fit for the smoothing kernel
           allow_offset (bool): flag to permit crescent centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths
                      
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

    ###################################################
    # setting up the model

    # number of gains
    N_gains = len(loggainamp_mean)

    # get Bessel function
    J1 = mu.J1()

    model = pm.Model()

    with model:

        ###############################################
        # set the priors for the crescent parameters
        
        # total flux prior
        if fit_total_flux:
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        # outer crescent radius
        Rout_pre = pm.Uniform('Rout',lower=0.0,upper=100.0)

        # dimensionless width
        psi = pm.Uniform('psi',lower=0.0,upper=1.0)
        Rin_pre = pm.Deterministic('Rin',Rout_pre*(1.0-psi))

        # dimension scaling
        Rout = eh.RADPERUAS*Rout_pre
        Rin = eh.RADPERUAS*Rin_pre

        # dimensionless shift
        tau = pm.Uniform('tau',lower=0.0,upper=1.0)

        # orientation
        phi = pm.VonMises('phi',mu=0.0,kappa=0.0001)

        if fit_syserr:
            # set the prior on the systematic error term to be uniform on [0,1]
            f = pm.Uniform('f',lower=0.0,upper=1.0,testval=0.01)
        else:
            if syserr is not None:
                f = syserr
            else:
                f = 0.0

        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        # Gaussian smoothing kernel parameters
        if (smooth is not None) & (fit_smooth == False):
            # smoothing width
            sigma = eh.RADPERUAS*(smooth/(2.0*np.sqrt(2.0*np.log(2.0))))
        elif (fit_smooth == True):
            # smoothing width
            sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=100.0)

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # determine the base visibilities

        inner_shift_real = pm.math.cos(-2.0*np.pi*psi*Rout*tau*((u*pm.math.cos(phi)) + (v*pm.math.sin(phi))))
        inner_shift_imag = pm.math.sin(-2.0*np.pi*psi*Rout*tau*((u*pm.math.cos(phi)) + (v*pm.math.sin(phi))))
        
        flux_factor = F / (np.pi*((Rout**2.0) - (Rin**2.0)))

        arg_out = 2.0*np.pi*Rout*rho
        # J1_out = (arg_out / 2.0) - ((arg_out**3.0)/16.0) + ((arg_out**5.0)/384.0) - ((arg_out**7.0)/18432.0) + ((arg_out**9.0)/1474560.0) - ((arg_out**11.0)/176947200.0) + ((arg_out**13.0)/29727129600.0) - ((arg_out**15.0)/6658877030400.0) + ((arg_out**17.0)/1917756584755200.0) - ((arg_out**19.0)/690392370511872000.0) + ((arg_out**21.0)/303772643025223680000.0) - ((arg_out**23.0)/160391955517318103040000.0) + ((arg_out**25.0)/100084580242806496296960000.0)
        J1_out = J1(arg_out)

        arg_in = 2.0*np.pi*Rin*rho
        # J1_in = (arg_in / 2.0) - ((arg_in**3.0)/16.0) + ((arg_in**5.0)/384.0) - ((arg_in**7.0)/18432.0) + ((arg_in**9.0)/1474560.0) - ((arg_in**11.0)/176947200.0) + ((arg_in**13.0)/29727129600.0) - ((arg_in**15.0)/6658877030400.0) + ((arg_in**17.0)/1917756584755200.0) - ((arg_in**19.0)/690392370511872000.0) + ((arg_in**21.0)/303772643025223680000.0) - ((arg_in**23.0)/160391955517318103040000.0) + ((arg_in**25.0)/100084580242806496296960000.0)
        J1_in = J1(arg_in)

        Ireal_pregain_preshift_presmooth = flux_factor*(((Rout/rho)*J1_out) - (inner_shift_real*((Rin/rho)*J1_in)))
        Iimag_pregain_preshift_presmooth = - (inner_shift_imag*((Rin/rho)*J1_in))

        ###############################################
        # smooth with the Gaussian kernel
        
        if (smooth is not None) | (fit_smooth == True):
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        else:
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))
        Ireal_pregain = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # compute the corruption terms

        if fit_gains:

            gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
            gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

            gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
            gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        else:
            gainamp_1 = 1.0
            gainamp_2 = 1.0

            gainphase_1 = 0.0
            gainphase_2 = 0.0

        ###############################################
        # apply corruptions to the model visibilities

        Ireal_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Ireal_pregain) - (pm.math.sin(gainphase_1 - gainphase_2)*Iimag_pregain))
        Iimag_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Iimag_pregain) + (pm.math.sin(gainphase_1 - gainphase_2)*Ireal_pregain))

        ###############################################
        # add in the systematic noise component

        Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
        err2 = ((f*Itot_model)**2.0)

        Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + err2)
        Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + err2)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        # save it as an output
        lnL = pm.Deterministic('lnL',pm.math.sum(-0.5*((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0) + pm.math.log(2.0*np.pi*(Ireal_err_model[mask]**2.0)) + pm.math.log(2.0*np.pi*(Iimag_err_model[mask]**2.0)))))

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            # if istep == 0:
            #     step = mu.get_step_for_trace(None,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            # else:
            #     step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'crescent',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'crescent',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag
                 }

    return modelinfo

def crescent_noisy(obs,LC=None,x0=0.0,y0=0.0,start=None,total_flux_estimate=None,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,n_start=25,
          n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,fit_smooth=False,
          smooth=None,fit_gains=True,tuning_windows=None,output_tuning=False,
          gain_amp_prior='normal',total_flux_prior=['uniform',0.0,1.0],**kwargs):
    """ Fit a Stokes I crescent model to a VLBI observation
        
       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           x0 (float): crescent center location along x-axis (uas)
           y0 (float): crescent center location along y-axis (uas)
           LC (array): light curve
           
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I crescent flux (Jy)
           offset_window (float): width of square offset window (uas)
               
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           fit_smooth (bool): flag to fit for the smoothing kernel
           allow_offset (bool): flag to permit crescent centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths
                      
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

    ###################################################
    # setting up the model

    # number of gains
    N_gains = len(loggainamp_mean)

    # get Bessel function
    J1 = mu.J1()

    model = pm.Model()

    with model:

        ###############################################
        # set the priors for the crescent parameters
        
        # total flux prior
        if fit_total_flux:
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        # outer crescent radius
        Rout_pre = pm.Uniform('Rout',lower=0.0,upper=100.0)

        # dimensionless width
        psi = pm.Uniform('psi',lower=0.0,upper=1.0)
        Rin_pre = pm.Deterministic('Rin',Rout_pre*(1.0-psi))

        # dimension scaling
        Rout = eh.RADPERUAS*Rout_pre
        Rin = eh.RADPERUAS*Rin_pre

        # dimensionless shift
        tau = pm.Uniform('tau',lower=0.0,upper=1.0)

        # orientation
        phi = pm.VonMises('phi',mu=0.0,kappa=0.0001)

        # systematic noise prescription
        # set the prior on the multiplicative systematic error term to be uniform on [0,1]
        multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=0.1,testval=0.02)

        # set the prior on the additive systematic error term to be uniform on [0,1] Jy
        additive = pm.Uniform('additive',lower=0.0,upper=0.1,testval=0.01)
        
        # set the power law error term priors
        logn0 = pm.Uniform('logn0',lower=-10.0,upper=0.0)
        n0 = pm.Deterministic('n0',pm.math.exp(logn0))
        umax = pm.Uniform('umax',lower=1.0e9,upper=5.0e9)
        b = pm.Uniform('b',lower=0.0,upper=5.0)
        c = pm.Uniform('c',lower=0.0,upper=3.0)

        # # systematic noise prescription
        # # set the prior on the multiplicative systematic error term to be uniform on [0,1]
        # multiplicative = pm.Normal('multiplicative',mu=0.02,sd=0.0001)

        # # set the prior on the additive systematic error term to be uniform on [0,1] Jy
        # additive = pm.Normal('additive',mu=0.01,sd=0.0001)
        
        # # set the power law error term priors
        # logn0 = pm.Normal('logn0',mu=np.log(0.1*np.sqrt(2.0)),sd=0.0001)
        # n0 = pm.Deterministic('n0',pm.math.exp(logn0))
        # umax = pm.Normal('umax',mu=2.0e9,sd=0.0001e9)
        # b = pm.Normal('b',mu=3.0,sd=0.001)
        # c = pm.Normal('c',mu=2.0,sd=0.001)

        # multiplicative = 0.0
        # additive = 0.0
        # n0 = 0.0
        # umax = 1.0e9
        # b = 0.0
        # c = 0.0
        
        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        # Gaussian smoothing kernel parameters
        if (smooth is not None) & (fit_smooth == False):
            # smoothing width
            sigma = eh.RADPERUAS*(smooth/(2.0*np.sqrt(2.0*np.log(2.0))))
        elif (fit_smooth == True):
            # smoothing width
            sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=100.0)

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # determine the base visibilities

        inner_shift_real = pm.math.cos(-2.0*np.pi*psi*Rout*tau*((u*pm.math.cos(phi)) + (v*pm.math.sin(phi))))
        inner_shift_imag = pm.math.sin(-2.0*np.pi*psi*Rout*tau*((u*pm.math.cos(phi)) + (v*pm.math.sin(phi))))
        
        flux_factor = F / (np.pi*((Rout**2.0) - (Rin**2.0)))

        arg_out = 2.0*np.pi*Rout*rho
        # J1_out = (arg_out / 2.0) - ((arg_out**3.0)/16.0) + ((arg_out**5.0)/384.0) - ((arg_out**7.0)/18432.0) + ((arg_out**9.0)/1474560.0) - ((arg_out**11.0)/176947200.0) + ((arg_out**13.0)/29727129600.0) - ((arg_out**15.0)/6658877030400.0) + ((arg_out**17.0)/1917756584755200.0) - ((arg_out**19.0)/690392370511872000.0) + ((arg_out**21.0)/303772643025223680000.0) - ((arg_out**23.0)/160391955517318103040000.0) + ((arg_out**25.0)/100084580242806496296960000.0)
        J1_out = J1(arg_out)

        arg_in = 2.0*np.pi*Rin*rho
        # J1_in = (arg_in / 2.0) - ((arg_in**3.0)/16.0) + ((arg_in**5.0)/384.0) - ((arg_in**7.0)/18432.0) + ((arg_in**9.0)/1474560.0) - ((arg_in**11.0)/176947200.0) + ((arg_in**13.0)/29727129600.0) - ((arg_in**15.0)/6658877030400.0) + ((arg_in**17.0)/1917756584755200.0) - ((arg_in**19.0)/690392370511872000.0) + ((arg_in**21.0)/303772643025223680000.0) - ((arg_in**23.0)/160391955517318103040000.0) + ((arg_in**25.0)/100084580242806496296960000.0)
        J1_in = J1(arg_in)

        Ireal_pregain_preshift_presmooth = flux_factor*(((Rout/rho)*J1_out) - (inner_shift_real*((Rin/rho)*J1_in)))
        Iimag_pregain_preshift_presmooth = - (inner_shift_imag*((Rin/rho)*J1_in))

        ###############################################
        # smooth with the Gaussian kernel
        
        if (smooth is not None) | (fit_smooth == True):
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        else:
            Ireal_pregain_preshift = Ireal_pregain_preshift_presmooth
            Iimag_pregain_preshift = Iimag_pregain_preshift_presmooth

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))
        Ireal_pregain_preLC = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain_preLC = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # multiply by light curve

        if (LC is not None):
            Ireal_pregain = Ireal_pregain_preLC*LC
            Iimag_pregain = Iimag_pregain_preLC*LC
        else:
            Ireal_pregain = Ireal_pregain_preLC*1.0
            Iimag_pregain = Iimag_pregain_preLC*1.0

        ###############################################
        # compute the corruption terms

        if fit_gains:

            gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
            gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

            gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
            gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        else:
            gainamp_1 = 1.0
            gainamp_2 = 1.0

            gainphase_1 = 0.0
            gainphase_2 = 0.0

        ###############################################
        # apply corruptions to the model visibilities

        Ireal_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Ireal_pregain) - (pm.math.sin(gainphase_1 - gainphase_2)*Iimag_pregain))
        Iimag_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Iimag_pregain) + (pm.math.sin(gainphase_1 - gainphase_2)*Ireal_pregain))

        ###############################################
        # add in the systematic noise component

        Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
        err2 = ((multiplicative*Itot_model)**2.0) + (additive**2.0) + ((n0**2.0)*((rho/umax)**c) / (1.0 + ((rho/umax)**(b+c))))

        Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + err2)
        Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + err2)

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        # save it as an output
        lnL = pm.Deterministic('lnL',pm.math.sum(-0.5*((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0) + pm.math.log(2.0*np.pi*(Ireal_err_model[mask]**2.0)) + pm.math.log(2.0*np.pi*(Iimag_err_model[mask]**2.0)))))

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            # if istep == 0:
            #     step = mu.get_step_for_trace(None,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            # else:
            #     step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'crescent',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'crescent',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'smooth': smooth,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'fit_smooth': fit_smooth,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag
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
    loggainamp_mean, loggainamp_std = mu.gain_amp_prior(obs)

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
        step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH,regularize=regularize)
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
             n_start=25,n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,
             gain_amp_prior='normal',const_ref_RL=True,**kwargs):
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

           gain_amp_prior (str): form of the gain amplitude prior; options are 'log', 'normal'
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)

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
    loggainamp_mean, loggainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    
    # prior info for gain phases
    gainphase_mu_R, gainphase_kappa_R = mu.gain_phase_prior(obs,ref_station=ref_station)
    gainphase_mu_L, gainphase_kappa_L = mu.gain_phase_prior(obs,ref_station=None)

    if const_ref_RL:
        if ref_station is not None:
            ind_ref = (A_gains == ref_station)
            gainphase_kappa_L[ind_ref] = 10000.0
    else:
        if ref_station is not None:
            ind_ref = (A_gains == ref_station)
            gainphase_kappa_temp = np.copy(gainphase_kappa_L[ind_ref])
            gainphase_kappa_temp[0] = 10000.0
            gainphase_kappa_L[ind_ref] = gainphase_kappa_temp

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

        if gain_amp_prior == 'log':
            # set the gain amplitude priors to be log-normal around the specified inputs
            logg_R = pm.Normal('right_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
            g_R = pm.Deterministic('right_gain_amps',pm.math.exp(logg_R))

            if RLequal:
                logg_L = pm.Deterministic('left_logg',logg_R)
                g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
            else:
                logg_L = pm.Normal('left_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
        if gain_amp_prior == 'normal':
            # set the gain amplitude priors to be normal around the specified inputs
            BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
            g_R = BoundedNormal('right_gain_amps',mu=np.exp(loggainamp_mean),sd=loggainamp_std,shape=N_gains)
            logg_R = pm.Deterministic('right_logg',pm.math.log(g_R))

            if RLequal:
                logg_L = pm.Deterministic('left_logg',logg_R)
                g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
            else:
                g_L = BoundedNormal('left_gain_amps',mu=np.exp(loggainamp_mean),sd=loggainamp_std,shape=N_gains)
                logg_L = pm.Deterministic('left_logg',pm.math.log(g_L))
        
        # set the gain phase priors to be periodic uniform on (-pi,pi)
        theta_R = pm.VonMises('right_gain_phases',mu=gainphase_mu_R,kappa=gainphase_kappa_R,shape=N_gains)
        
        if RLequal:
            theta_L = pm.Deterministic('left_gain_phases',theta_R)
        else:
            theta_L = pm.VonMises('left_gain_phases',mu=gainphase_mu_L,kappa=gainphase_kappa_L,shape=N_gains)
        
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
            step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth,regularize=regularize)
            burnin_trace = pm.sample(draws=steps, start=start, tune=500, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            start = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth,regularize=regularize)
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
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'const_ref_RL': const_ref_RL
                 }

    return modelinfo

def gausscirc_noisy(obs,LC=None,x0=0.0,y0=0.0,start=None,total_flux_estimate=None,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,n_start=25,
          n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,fit_gains=True,
          tuning_windows=None,output_tuning=False,gain_amp_prior='normal',
          total_flux_prior=['uniform',0.0,1.0],**kwargs):
    """ Fit a Stokes I image to a VLBI observation
        
       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           x0 (float): image center location along x-axis (uas)
           y0 (float): image center location along y-axis (uas)
           LC (array): light curve
           
           start (modelinfo): the DMC model info for a previous run from which to continue sampling
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           offset_window (float): width of square offset window (uas)
               
           fit_total_flux (bool): flag to fit for the total flux
           fit_gains (bool): flag to fit for the complex gains
           allow_offset (bool): flag to permit image centroid to be a free parameter
           output_tuning (bool): flag to output intermediate tuning chains
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           tuning_windows (list): sequence of tuning window lengths
                       
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

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
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        sigma = eh.RADPERUAS*pm.Uniform('sigma',lower=0.0,upper=100.0)

        # systematic noise prescription
        # set the prior on the multiplicative systematic error term to be uniform on [0,1]
        multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=1.0,testval=0.01)

        # set the prior on the additive systematic error term to be uniform on [0,1] Jy
        additive = pm.Uniform('additive',lower=0.0,upper=1.0,testval=0.001)

        # set the power law error term priors
        n0 = pm.Uniform('n0',lower=0.0,upper=2.0)
        umax = pm.Uniform('umax',lower=0.0,upper=1.0e10)
        b = pm.Uniform('b',lower=0.0,upper=10.0)
        c = pm.Uniform('c',lower=0.0,upper=10.0)

        # permit a centroid shift in the image
        if allow_offset:
            x0_model = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=x0)
            y0_model = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0),testval=y0)
        else:
            x0_model = x0*eh.RADPERUAS
            y0_model = y0*eh.RADPERUAS

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # perform the required Fourier transforms
        
        Ireal_pregain_preshift = F*pm.math.exp(-2.0*(np.pi**2.0)*(sigma**2.0)*(rho**2.0))
        Iimag_pregain_preshift = 0.0

        ###############################################
        # shift centroid

        shift_term = 2.0*np.pi*((u*x0_model) + (v*y0_model))
        Ireal_pregain_preLC = (Ireal_pregain_preshift*pm.math.cos(shift_term)) + (Iimag_pregain_preshift*pm.math.sin(shift_term))
        Iimag_pregain_preLC = (Iimag_pregain_preshift*pm.math.cos(shift_term)) - (Ireal_pregain_preshift*pm.math.sin(shift_term))

        ###############################################
        # multiply by light curve

        if (LC is None):
            Ireal_pregain = Ireal_pregain_preLC*1.0
            Iimag_pregain = Iimag_pregain_preLC*1.0
        else:
            Ireal_pregain = Ireal_pregain_preLC*LC
            Iimag_pregain = Iimag_pregain_preLC*LC

        ###############################################
        # compute the corruption terms

        if fit_gains:

            gainamp_1 = pm.math.exp(pm.math.dot(gain_design_mat_1,logg))
            gainamp_2 = pm.math.exp(pm.math.dot(gain_design_mat_2,logg))

            gainphase_1 = pm.math.dot(gain_design_mat_1,theta)
            gainphase_2 = pm.math.dot(gain_design_mat_2,theta)

        else:
            gainamp_1 = 1.0
            gainamp_2 = 1.0

            gainphase_1 = 0.0
            gainphase_2 = 0.0

        ###############################################
        # apply corruptions to the model visibilities

        Ireal_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Ireal_pregain) - (pm.math.sin(gainphase_1 - gainphase_2)*Iimag_pregain))
        Iimag_model = gainamp_1*gainamp_2*((pm.math.cos(gainphase_1 - gainphase_2)*Iimag_pregain) + (pm.math.sin(gainphase_1 - gainphase_2)*Ireal_pregain))

        ###############################################
        # add in the systematic noise component

        Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
        err2 = ((multiplicative*Itot_model)**2.0) + (additive**2.0) + ((n0**2.0)*((rho/umax)**c) / (1.0 + ((rho/umax)**(b+c))))

        Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + err2)
        Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + err2)

        ###############################################
        # define the likelihood

        # Ireal_model_int = pm.Normal('Ireal_model_int',mu=0.0,sd=1.0,shape=len(I_real))
        # Ireal_model_int2 = pm.Deterministic('Ireal_model_int2',Ireal_model + (Ireal_model_int*pm.math.sqrt(err2)))
        # L_real = pm.Normal('L_real',mu=Ireal_model_int2[mask],sd=I_real_err[mask],observed=I_real[mask])

        # Iimag_model_int = pm.Normal('Iimag_model_int',mu=0.0,sd=1.0,shape=len(I_imag))
        # Iimag_model_int2 = pm.Deterministic('Iimag_model_int2',Iimag_model + (Iimag_model_int*pm.math.sqrt(err2)))
        # L_imag = pm.Normal('L_imag',mu=Iimag_model_int2[mask],sd=I_imag_err[mask],observed=I_imag[mask])

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'gauss',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'gauss',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'x0': x0,
                 'y0': y0,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'total_flux_estimate': total_flux_estimate,
                 'fit_gains': fit_gains,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains,
                 'gain_amp_prior': gain_amp_prior,
                 'fit_gains': fit_gains,
                 'tuning_windows': tuning_windows,
                 'output_tuning': output_tuning,
                 'diag': diag,
                 }
                 
    return modelinfo

def gauss(obs,start=None,total_flux_estimate=None,loose_change=False,
          fit_total_flux=False,allow_offset=False,offset_window=200.0,
          n_start=25,n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,
          total_flux_prior=['uniform',0.0,1.0],gain_amp_prior='normal',
          fit_gains=True,tuning_windows=None,output_tuning=False,**kwargs):
    """ Fit an elliptical Gaussian source structure to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
    
           loose_change (bool): flag to use the "loose change" noise prescription
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

    if gain_amp_prior not in ['log','normal']:
        raise Exception('gain_amp_prior keyword argument must be log or normal.')

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')
    regularize = kwargs.get('regularize',True)
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))
    gain_phase_priors = kwargs.get('gain_phase_priors',(0.0,0.0001))
    max_treedepth = kwargs.get('max_treedepth',MAX_TREEDEPTH)
    early_max_treedepth = kwargs.get('early_max_treedepth',EARLY_MAX_TREEDEPTH)
    output_tuning_dir = kwargs.get('output_tuning_dir','./tuning')
    diag = kwargs.get('diag',False)

    ###################################################
    # data bookkeeping

    # first, make sure we're using a Stokes representation
    if obs.polrep is not 'stokes':
        obs = obs.switch_polrep('stokes')

    # (u,v) coordinate info
    u = obs.data['u']
    v = obs.data['v']
    rho = np.sqrt((u**2.0) + (v**2.0))
    phi = (np.pi/2.0) - np.arctan2(v,u)

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
    gainamp_mean, gainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,gain_phase_priors=gain_phase_priors)
    
    if ref_station is not None:
        ind_ref = (A_gains == ref_station)
        gainphase_kappa[ind_ref] = 10000.0

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
            if 'uniform' in total_flux_prior:
                F = pm.Uniform('F',lower=total_flux_prior[1],upper=total_flux_prior[2],testval=total_flux_estimate)
            else:
                # set to be normal around the correct value, but bounded positive
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                F = BoundedNormal('F',mu=total_flux_estimate,sd=0.1*total_flux_estimate,testval=total_flux_estimate)
        else:
            # fix at input value
            F = total_flux_estimate

        # Gaussian widths
        sigma_x = eh.RADPERUAS*pm.Uniform('sigma_x',lower=0.0,upper=100.0)
        sigma_y = eh.RADPERUAS*pm.Uniform('sigma_y',lower=0.0,upper=100.0)

        # Gaussian orientation
        psi = pm.VonMises('psi',mu=0.0,kappa=0.0001)

        # systematic noise prescription
        if loose_change:
            # set the prior on the multiplicative systematic error term to be uniform on [0,1]
            multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=1.0)

            # set the prior on the additive systematic error term to be uniform on [0,1] Jy
            additive = pm.Uniform('additive',lower=0.0,upper=1.0)
        else:
            multiplicative = 0.0
            additive = 0.0

        # permit a centroid shift in the image
        if allow_offset:
            x0 = eh.RADPERUAS*pm.Uniform('x0',lower=-(offset_window/2.0),upper=(offset_window/2.0))
            y0 = eh.RADPERUAS*pm.Uniform('y0',lower=-(offset_window/2.0),upper=(offset_window/2.0))
        else:
            x0 = 0.0
            y0 = 0.0

        ###############################################
        # set the priors for the gain parameters

        if fit_gains:

            if gain_amp_prior == 'log':
                # set the gain amplitude priors to be log-normal around the specified inputs
                logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
                g = pm.Deterministic('gain_amps',pm.math.exp(logg))
            if gain_amp_prior == 'normal':
                # set the gain amplitude priors to be normal around the specified inputs
                BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
                g = BoundedNormal('gain_amps',mu=gainamp_mean,sd=gainamp_std,shape=N_gains)
                logg = pm.Deterministic('logg',pm.math.log(g))
            
            # set the gain phase priors to be periodic uniform on (-pi,pi)
            theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # compute Fourier transform

        # rotation
        a = ((pm.math.cos(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.sin(psi)**2.0)/(2.0*(sigma_y**2.0)))
        b = (pm.math.sin(2.0*psi)/(2.0*(sigma_x**2.0))) - (pm.math.sin(2.0*psi)/(2.0*(sigma_y**2.0)))
        c = ((pm.math.sin(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.cos(psi)**2.0)/(2.0*(sigma_y**2.0)))
        rot_term = (c*(u**2.0)) - (b*u*v) + (a*(v**2.0))

        # shift
        shift_term = 2.0*np.pi*((u*x0) + (v*y0))

        # FT
        Ireal_pregain = F*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)
        Iimag_pregain = -F*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)

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

        Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
        Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + ((multiplicative*Itot_model)**2.0) + (additive**2.0))
        Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + ((multiplicative*Itot_model)**2.0) + (additive**2.0))

        ###############################################
        # define the likelihood

        L_real = pm.Normal('L_real',mu=Ireal_model[mask],sd=Ireal_err_model[mask],observed=I_real[mask])
        L_imag = pm.Normal('L_imag',mu=Iimag_model[mask],sd=Iimag_err_model[mask],observed=I_imag[mask])

        ###############################################
        # keep track of summed-squared residuals (ssr)

        ssr = pm.Deterministic('ssr',pm.math.sum((((Ireal_model[mask]-I_real[mask])/Ireal_err_model[mask])**2.0) + (((Iimag_model[mask]-I_imag[mask])/Iimag_err_model[mask])**2.0)))

    ###################################################
    # fit the model

    # set up tuning windows
    if tuning_windows is not None:
        windows = np.array(tuning_windows)
    else:
        windows = n_start * (2**np.arange(np.floor(np.log2((n_tune - n_burn) / n_start))))

    # make directory for saving intermediate output
    if output_tuning:
        if not os.path.exists(output_tuning_dir):
            os.mkdir(output_tuning_dir)

    # keep track of the tuning runs
    tuning_trace_list = list()
    with model:
        
        # initialize using previous run if supplied
        if start is not None:
            burnin_trace = start['trace']
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
        else:
            burnin_trace = None
            starting_values = None

        # burn-in and initial mass matrix tuning
        for istep, steps in enumerate(windows):
            step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
            burnin_trace = pm.sample(draws=steps, start=starting_values, tune=n_burn, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            starting_values = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

            # save intermediate output
            if output_tuning:
                
                modelinfo = {'modeltype': 'gauss',
                 'model': model,
                 'trace': burnin_trace,
                 'tuning_traces': tuning_trace_list,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'total_flux_estimate': total_flux_estimate,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains
                 }

                # make directory for this step
                dirname = output_tuning_dir+'/step'+str(istep).zfill(5)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # save modelinfo
                mu.save_model(modelinfo,dirname+'/modelinfo.p')

                # save trace plots
                pl.plot_trace(modelinfo)
                plt.savefig(dirname+'/traceplots.png',dpi=300)
                plt.close()

                # HMC energy plot
                energyplot = pl.plot_energy(modelinfo)
                plt.savefig(dirname+'/energyplot.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for main sampling run
                stepplot = pl.plot_stepsize(modelinfo)
                plt.savefig(dirname+'/stepsize.png',dpi=300,bbox_inches='tight')
                plt.close()

                # plot HMC step size for full run
                stepplot = plt.figure(figsize=(6,6))
                ax = stepplot.add_axes([0.15,0.1,0.8,0.8])
                steparr = list()
                for ttrace in tuning_trace_list:
                    stepsize = ttrace.get_sampler_stats('step_size')
                    steparr.append(stepsize)
                steparr = np.concatenate(steparr)
                ax.plot(steparr,'b-')
                ax.semilogy()
                ax.set_ylabel('Step size')
                ax.set_xlabel('Trial number')
                plt.savefig(dirname+'/stepsize_full.png',dpi=300,bbox_inches='tight')
                plt.close()

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,regularize=regularize,diag=diag,adapt_step_size=True,max_treedepth=max_treedepth,early_max_treedepth=early_max_treedepth)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=starting_values, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'gauss',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'allow_offset': allow_offset,
                 'offset_window': offset_window,
                 'total_flux_estimate': total_flux_estimate,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains
                 }

    return modelinfo

def polgauss(obs,total_flux_estimate=None,RLequal=False,fit_StokesV=True,
             fit_total_flux=False,allow_offset=False,offset_window=200.0,
             n_start=25,n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,**kwargs):
    """ Fit a polarized elliptical Gaussian source structure to a VLBI observation

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
    gain_amp_priors = kwargs.get('gain_amp_priors',(1.0,0.1))

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
    loggainamp_mean, loggainamp_std = mu.gain_amp_prior(obs,gain_amp_priors=gain_amp_priors)
    
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

        # Gaussian widths
        sigma_x = eh.RADPERUAS*pm.Uniform('sigma_x',lower=0.0,upper=10000000.0)
        sigma_y = eh.RADPERUAS*pm.Uniform('sigma_y',lower=0.0,upper=10000000.0)

        # Gaussian orientation
        psi = pm.VonMises('psi',mu=0.0,kappa=0.0001)

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
        # compute Fourier transform

        # rotation
        a = ((pm.math.cos(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.sin(psi)**2.0)/(2.0*(sigma_y**2.0)))
        b = (pm.math.sin(2.0*psi)/(2.0*(sigma_x**2.0))) - (pm.math.sin(2.0*psi)/(2.0*(sigma_y**2.0)))
        c = ((pm.math.sin(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.cos(psi)**2.0)/(2.0*(sigma_y**2.0)))
        rot_term = (c*(u**2.0)) - (b*u*v) + (a*(v**2.0))
        
        # shift
        shift_term = 2.0*np.pi*((u*x0) + (v*y0))
        
        # FT
        Ireal = I*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)
        Iimag = -I*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)

        Qreal = Q*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)
        Qimag = -Q*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)

        Ureal = U*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)
        Uimag = -U*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)

        Vreal = V*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)
        Vimag = -V*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)*rot_term)
        
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
            burnin_trace = pm.sample(draws=steps, start=start, tune=500, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
            start = [t[-1] for t in burnin_trace._straces.values()]
            tuning_trace_list.append(burnin_trace)

        # posterior sampling
        step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH,regularize=regularize)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=start, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'polgauss',
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

def multigauss(obs,N,window=200.0,total_flux_estimate=None,
          loose_change=False,fit_total_flux=False,n_start=25,
          n_burn=500,n_tune=5000,ntuning=2000,ntrials=10000,**kwargs):
    """ Fit a source structure consisting of N elliptical Gaussians to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           N (int): number of Gaussian components
           total_flux_estimate (float): estimate of total Stokes I image flux (Jy)
           window (float): width of region where Gaussian centroids can be placed (uas)
    
           loose_change (bool): flag to use the "loose change" noise prescription
           fit_total_flux (bool): flag to fit for the total flux
            
           n_start (int): initial number of default tuning steps
           n_burn (int): number of burn-in steps
           n_tune (int): number of mass matrix tuning steps

           ntuning (int): number of tuning steps to take during last leg
           ntrials (int): number of posterior samples to take
           
       Returns:
           modelinfo: a dictionary object containing the model fit information

    """
    
    if True:
        raise Exception('multigauss model is not yet functional.')
    
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
    phi = (np.pi/2.0) - np.arctan2(v,u)

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
    loggainamp_mean, loggainamp_std = mu.gain_amp_prior(obs)
    
    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,ref_station=ref_station)

    # number of gains
    N_gains = len(loggainamp_mean)

    # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
    dirichlet_weights = 1.0*np.ones(N)

    ###################################################
    # setting up the model

    # cloning the data vectors for broadcasting
    u = np.array([u]*N)
    v = np.array([v]*N)
    

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

        # Impose a Dirichlet prior on the Gaussian intensities,
        # with summation constraint equal to the total flux
        I0 = pm.Dirichlet('I0',dirichlet_weights)
        I = pm.Deterministic('I',I0*F)

        # Gaussian widths
        sigma_x = eh.RADPERUAS*pm.Uniform('sigma_x',lower=0.0,upper=100.0,shape=N)
        sigma_y = eh.RADPERUAS*pm.Uniform('sigma_y',lower=0.0,upper=100.0,shape=N)

        # Gaussian orientation
        psi = pm.VonMises('psi',mu=0.0,kappa=0.0001,shape=N)

        # Gaussian centroid
        x0 = eh.RADPERUAS*pm.Uniform('x0',lower=-(window/2.0),upper=(window/2.0),shape=N)
        y0 = eh.RADPERUAS*pm.Uniform('y0',lower=-(window/2.0),upper=(window/2.0),shape=N)

        """
        # make priors for the different components
        sigma_x_vec = list()
        sigma_y_vec = list()
        psi_vec = list()
        x0_vec = list()
        y0_vec = list()
        for i in range(N):

            # Gaussian widths
            sigma_x_vec.append(eh.RADPERUAS*pm.Uniform('sigma_x_'+str(i).zfill(2),lower=0.0,upper=100.0))
            sigma_y_vec.append(eh.RADPERUAS*pm.Uniform('sigma_y_'+str(i).zfill(2),lower=0.0,upper=100.0))

            # Gaussian orientation
            psi_vec.append(pm.VonMises('psi_'+str(i).zfill(2),mu=0.0,kappa=0.0001))

            # Gaussian centroid
            x0_vec.append(eh.RADPERUAS*pm.Uniform('x0_'+str(i).zfill(2),lower=-(window/2.0),upper=(window/2.0)))
            y0_vec.append(eh.RADPERUAS*pm.Uniform('y0_'+str(i).zfill(2),lower=-(window/2.0),upper=(window/2.0)))
        """

        # systematic noise prescription
        if loose_change:
            # set the prior on the multiplicative systematic error term to be uniform on [0,1]
            multiplicative = pm.Uniform('multiplicative',lower=0.0,upper=1.0)

            # set the prior on the additive systematic error term to be uniform on [0,1] Jy
            additive = pm.Uniform('additive',lower=0.0,upper=1.0)
        else:
            multiplicative = 0.0
            additive = 0.0

        ###############################################
        # set the priors for the gain parameters

        # set the gain amplitude priors to be log-normal around the specified inputs
        logg = pm.Normal('logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
        g = pm.Deterministic('gain_amps',pm.math.exp(logg))
        
        # set the gain phase priors to be periodic uniform on (-pi,pi)
        theta = pm.VonMises('gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

        ###############################################
        # compute Fourier transform

        """
        # rotation
        a = ((pm.math.cos(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.sin(psi)**2.0)/(2.0*(sigma_y**2.0)))
        b = (pm.math.sin(2.0*psi)/(2.0*(sigma_x**2.0))) - (pm.math.sin(2.0*psi)/(2.0*(sigma_y**2.0)))
        c = ((pm.math.sin(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.cos(psi)**2.0)/(2.0*(sigma_y**2.0)))
        rot_prefac = -4.0*(np.pi**2.0)*(sigma_x**2.0)*(sigma_y**2.0)
        rot_term = pm.math.dot(rot_prefac*c,u**2.0) - pm.math.dot(rot_prefac*b,u*v) + pm.math.dot(rot_prefac*a,v**2.0)

        # shift
        shift_term = 2.0*np.pi*(pm.math.dot(x0,u) + pm.math.dot(y0,v))
        
        # FT
        Ireal_pregain = pm.math.sum(I[None,:]*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x[None,:]**2.0)*(sigma_y[None,:]**2.0)*rot_term),axis=1)
        Iimag_pregain = pm.math.sum(-I[None,:]*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x[None,:]**2.0)*(sigma_y[None,:]**2.0)*rot_term),axis=1)
        """

        # rotation
        a = ((pm.math.cos(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.sin(psi)**2.0)/(2.0*(sigma_y**2.0)))
        b = (pm.math.sin(2.0*psi)/(2.0*(sigma_x**2.0))) - (pm.math.sin(2.0*psi)/(2.0*(sigma_y**2.0)))
        c = ((pm.math.sin(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.cos(psi)**2.0)/(2.0*(sigma_y**2.0)))
        rot_term = (c[None,:]*(u**2.0)) - (b[None,:]*(u*v)) + (a[None,:]*(v**2.0))

        # shift
        shift_term = 2.0*np.pi*((x0[None,:]*u) + (y0[None,:]*v))

        # FT
        Ireal_pregain = pm.math.sum(I[None,:]*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x[None,:]**2.0)*(sigma_y[None,:]**2.0)*rot_term),axis=1)
        Iimag_pregain = pm.math.sum(-I[None,:]*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x[None,:]**2.0)*(sigma_y[None,:]**2.0)*rot_term),axis=1)
        

        """
        # rotation
        a = ((pm.math.cos(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.sin(psi)**2.0)/(2.0*(sigma_y**2.0)))
        b = (pm.math.sin(2.0*psi)/(2.0*(sigma_x**2.0))) - (pm.math.sin(2.0*psi)/(2.0*(sigma_y**2.0)))
        c = ((pm.math.sin(psi)**2.0)/(2.0*(sigma_x**2.0))) + ((pm.math.cos(psi)**2.0)/(2.0*(sigma_y**2.0)))
        rot_term = (c[None,:]*(u[:,None]**2.0)) - (b[None,:]*(u[:,None]*v[:,None])) + (a[None,:]*(v[:,None]**2.0))
        # rot_term = (c*(u**2.0)) - (b*u*v) + (a*(v**2.0))
        # rot_term = pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x[None,:]**2.0)*(sigma_y[None,:]**2.0)*((c[None,:]*(u[:,None]**2.0)) - (b[None,:]*(u[:,None]*v[:,None])) + (a[None,:]*(v[:,None]**2.0))))

        # shift
        shift_term = 2.0*np.pi*((x0[None,:]*u[:,None]) + (y0[None,:]*v[:,None]))
        # shift_term = 2.0*np.pi*((u*x0) + (v*y0))

        # FT
        Ireal_pregain = pm.math.sum(I[None,:]*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x[None,:]**2.0)*(sigma_y[None,:]**2.0)*rot_term),axis=1)
        Iimag_pregain = pm.math.sum(-I[None,:]*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x[None,:]**2.0)*(sigma_y[None,:]**2.0)*rot_term),axis=1)
        """

        """
        Ireal_pregain_presum = list()
        Iimag_pregain_presum = list()
        for i in range(N):

            # rotation
            a = ((pm.math.cos(psi_vec[i])**2.0)/(2.0*(sigma_x_vec[i]**2.0))) + ((pm.math.sin(psi_vec[i])**2.0)/(2.0*(sigma_y_vec[i]**2.0)))
            b = (pm.math.sin(2.0*psi_vec[i])/(2.0*(sigma_x_vec[i]**2.0))) - (pm.math.sin(2.0*psi_vec[i])/(2.0*(sigma_y_vec[i]**2.0)))
            c = ((pm.math.sin(psi_vec[i])**2.0)/(2.0*(sigma_x_vec[i]**2.0))) + ((pm.math.cos(psi_vec[i])**2.0)/(2.0*(sigma_y_vec[i]**2.0)))
            rot_term = (c*(u**2.0)) - (b*u*v) + (a*(v**2.0))

            # shift
            shift_term = 2.0*np.pi*((u*x0_vec[i]) + (v*y0_vec[i]))

            # FT
            Ireal_pregain_presum = I[i]*pm.math.cos(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x_vec[i]**2.0)*(sigma_y_vec[i]**2.0)*rot_term)
            Iimag_pregain_presum = -I[i]*pm.math.sin(shift_term)*pm.math.exp(-4.0*(np.pi**2.0)*(sigma_x_vec[i]**2.0)*(sigma_y_vec[i]**2.0)*rot_term)

        Ireal_pregain = pm.math.sum(Ireal_pregain_presum,axis=0)
        Iimag_pregain = pm.math.sum(Iimag_pregain_presum,axis=0)
        """

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

        Itot_model = pm.math.sqrt((Ireal_model**2.0) + (Iimag_model**2.0))
        Ireal_err_model = pm.math.sqrt((I_real_err**2.0) + ((multiplicative*Itot_model)**2.0) + (additive**2.0))
        Iimag_err_model = pm.math.sqrt((I_imag_err**2.0) + ((multiplicative*Itot_model)**2.0) + (additive**2.0))

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
        step = mu.get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=MAX_TREEDEPTH,early_max_treedepth=EARLY_MAX_TREEDEPTH,regularize=regularize)
        trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=start, chains=1, discard_tuned_samples=False)

    ###################################################
    # package the model info

    modelinfo = {'modeltype': 'multigauss',
                 'model': model,
                 'trace': trace,
                 'tuning_traces': tuning_trace_list,
                 'loose_change': loose_change,
                 'fit_total_flux': fit_total_flux,
                 'window': window,
                 'total_flux_estimate': total_flux_estimate,
                 'ntuning': ntuning,
                 'ntrials': ntrials,
                 'obs': obs,
                 'stations': stations,
                 'T_gains': T_gains,
                 'A_gains': A_gains
                 }

    return modelinfo