#######################################################
# imports
#######################################################

from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh
import pymc3 as pm
import pickle

#######################################################
# constants
#######################################################

SEFD_error_budget = {'AA':0.10,
                     'AP':0.11,
                     'AZ':0.07,
                     'LM':0.22,
                     'PV':0.10,
                     'SM':0.15,
                     'JC':0.14,
                     'SP':0.07}

#######################################################
# functions
#######################################################

def gain_logamp_prior(obs,SEFD_error_budget=SEFD_error_budget):
    """ Construct vector of prior means and standard deviations
        for log gain amplitudes

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           SEFD_error_budget (dict): dictionary of a priori error budget
                                     for station SEFDs (fractional)
           
       Returns:
           loggainamp_mean: prior means for log gain amplitudes
           loggainamp_std: prior standard deviations for log gain amplitudes

    """

    # get arrays of station names
    ant1 = obs.data['t1']
    ant2 = obs.data['t2']

    # get array of timestamps
    time = obs.data['time']
    timestamps = np.unique(time)

    # Determine the total number of gains that need to be solved for
    N_gains = 0
    T_gains = list()
    A_gains = list()
    for it, t in enumerate(timestamps):
        ind_here = (time == t)
        N_gains += len(np.unique(np.concatenate((ant1[ind_here],ant2[ind_here]))))
        stations_here = np.unique(np.concatenate((ant1[ind_here],ant2[ind_here])))
        for ii in range(len(stations_here)):
            A_gains.append(stations_here[ii])
            T_gains.append(t)
    T_gains = np.array(T_gains)
    A_gains = np.array(A_gains)

    # initialize vectors of gain prior means and standard deviations
    gainamp_mean = np.ones(N_gains)
    gainamp_std = np.ones(N_gains)

    # loop over stations
    for key in SEFD_error_budget.keys():
        index = (A_gains == key)
        gainamp_mean[index] = 1.0
        gainamp_std[index] = SEFD_error_budget[key]

    # take log
    loggainamp_mean = np.log(gainamp_mean)
    loggainamp_std = gainamp_std/gainamp_mean

    return loggainamp_mean, loggainamp_std

def gain_phase_prior(obs,ref_station='AA'):
    """ Construct vector of prior means and inverse standard deviations
        for gain phases

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           ref_station (str): name of reference station
           
       Returns:
           gainphase_mu: prior means for gain phases
           gainphase_kappa: prior inverse standard deviations for gain phases

    """

    # get arrays of station names
    ant1 = obs.data['t1']
    ant2 = obs.data['t2']

    # get array of timestamps
    time = obs.data['time']
    timestamps = np.unique(time)

    # Determine the total number of gains that need to be solved for
    N_gains = 0
    T_gains = list()
    A_gains = list()
    for it, t in enumerate(timestamps):
        ind_here = (time == t)
        N_gains += len(np.unique(np.concatenate((ant1[ind_here],ant2[ind_here]))))
        stations_here = np.unique(np.concatenate((ant1[ind_here],ant2[ind_here])))
        for ii in range(len(stations_here)):
            A_gains.append(stations_here[ii])
            T_gains.append(t)
    T_gains = np.array(T_gains)
    A_gains = np.array(A_gains)

    # initialize vectors of gain phase means and inverse standard deviations
    gainphase_mu = np.zeros(N_gains)
    gainphase_kappa = 0.0001*np.ones(N_gains)

    # set reference station standard devation to be tiny
    if ref_station is not None:
        for it, t in enumerate(timestamps):
            index = (T_gains == t)
            ants_here = A_gains[index]
            for ant in ants_here:
                if ant == ref_station:
                    ind = ((T_gains == t) & (A_gains == ant))
                    gainphase_kappa[ind] = 10000.0
                    break

    return gainphase_mu, gainphase_kappa

def get_step_for_trace(trace=None, model=None, regularize=True, regular_window=5, regular_variance=1e-3, **kwargs):
    """ Define a tuning procedure that adapts off-diagonal mass matrix terms
        adapted from a blog post by Dan Foreman-Mackey here:
        https://dfm.io/posts/pymc3-mass-matrix/

       Args:
           trace (trace): pymc3 trace object
           model (model): pymc3 model object
           
           regularize (bool): flag to turn on covariance matrix regularization
           regular_window (int): size of parameter space at which regularization becomes important
           regular_variance (float): magnitude of covariance floor
           
       Returns:
           pymc3 step_methods object

    """

    model = pm.modelcontext(model)
    
    # If not given, use the trivial metric
    if trace is None:
        potential = pm.step_methods.hmc.quadpotential.QuadPotentialFull(np.eye(model.ndim))
        return pm.NUTS(potential=potential, **kwargs)
    
    # Loop over samples and convert to the relevant parameter space
    # while removing divergent samples
    div_mask = np.invert(np.copy(trace.diverging))
    samples = np.empty((div_mask.sum() * trace.nchains, model.ndim))
    i = 0
    imask = 0
    for chain in trace._straces.values():
        for p in chain:
            if div_mask[imask]:
                samples[i] = model.bijection.map(p)
                i += 1
            imask += 1
    
    # Compute the sample covariance
    cov = np.cov(samples, rowvar=0)
    
    # Stan uses a regularized estimator for the covariance matrix to
    # be less sensitive to numerical issues for large parameter spaces.
    if regularize:
        N = len(samples)
        cov = cov * N / (N + regular_window)
        cov[np.diag_indices_from(cov)] += regular_variance * regular_window / (N + regular_window)
    
    # Use the sample covariance as the inverse metric
    potential = pm.step_methods.hmc.quadpotential.QuadPotentialFull(cov)

    return pm.NUTS(potential=potential, **kwargs)

#######################################################
# io utilities
#######################################################

def save_model(modelinfo,outfile):
    """ Save a model as a binary pickle file

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           outfile (str): name of output file
           
       Returns:
           None
    
    """

    # saving the model file
    pickle.dump(modelinfo,open(outfile,'wb'),protocol=pickle.HIGHEST_PROTOCOL)

def load_model(infile):
    """ Load a model saved with save_model

       Args:
           infile (str): name of model file
           
       Returns:
           modelinfo (dict): dmc modelinfo dictionary
    
    """

    # loading the model file
    modelinfo = pickle.load(open(infile,'rb'))

    return modelinfo

def make_image(modelinfo,moment,burnin=0):
    """ Make eht-imaging image object

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           moment (str): the type of posterior moment to save; choices are mean, median, std, snr
           burnin (int): length of burn-in
           
       Returns:
           im: ehtim image object
    
    """
    if modelinfo['modeltype'] not in ['image','polimage']:
        raise Exception('modeltype is not image or polimage!')
    if moment not in ['mean','median','std','snr']:
        raise Exception('moment ' + moment + ' not recognized!')

    ###################################################
    # organizing image information

    nx = modelinfo['nx']
    ny = modelinfo['ny']
    xmin = modelinfo['xmin']
    xmax = modelinfo['xmax']
    ymin = modelinfo['ymin']
    ymax = modelinfo['ymax']

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

    # make edge arrays
    xspacing = np.mean(x_1d[1:]-x_1d[0:-1])
    x_edges_1d = np.append(x_1d,x_1d[-1]+xspacing) - (xspacing/2.0)
    yspacing = np.mean(y_1d[1:]-y_1d[0:-1])
    y_edges_1d = np.append(y_1d,y_1d[-1]+yspacing) - (yspacing/2.0)
    x_edges, y_edges = np.meshgrid(x_edges_1d,y_edges_1d,indexing='ij')

    ###################################################
    # organize chain info and compute moment

    trace = modelinfo['trace']

    # remove burnin
    I = trace['I'][burnin:]
    if modelinfo['modeltype'] is 'polimage':
        Q = trace['Q'][burnin:]
        U = trace['U'][burnin:]
        V = trace['V'][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)
    I = I[div_mask]
    if modelinfo['modeltype'] is 'polimage':
        Q = Q[div_mask]
        U = U[div_mask]
        V = V[div_mask]

    # reshape array
    if moment == 'mean':
        Ivec = np.mean(I,axis=0).reshape((nx,ny))
        if modelinfo['modeltype'] is 'polimage':
            Qvec = np.mean(Q,axis=0).reshape((nx,ny))
            Uvec = np.mean(U,axis=0).reshape((nx,ny))
            Vvec = np.mean(V,axis=0).reshape((nx,ny))
    elif moment == 'median':
        Ivec = np.median(I,axis=0).reshape((nx,ny))
        if modelinfo['modeltype'] is 'polimage':
            Qvec = np.median(Q,axis=0).reshape((nx,ny))
            Uvec = np.median(U,axis=0).reshape((nx,ny))
            Vvec = np.median(V,axis=0).reshape((nx,ny))
    elif moment == 'std':
        Ivec = np.std(I,axis=0).reshape((nx,ny))
        if modelinfo['modeltype'] is 'polimage':
            Qvec = np.std(Q,axis=0).reshape((nx,ny))
            Uvec = np.std(U,axis=0).reshape((nx,ny))
            Vvec = np.std(V,axis=0).reshape((nx,ny))
    elif moment == 'snr':
        Ivec = np.mean(I,axis=0).reshape((nx,ny)) / np.std(I,axis=0).reshape((nx,ny))
        if modelinfo['modeltype'] is 'polimage':
            Qvec = np.mean(Q,axis=0).reshape((nx,ny)) / np.std(Q,axis=0).reshape((nx,ny))
            Uvec = np.mean(U,axis=0).reshape((nx,ny)) / np.std(U,axis=0).reshape((nx,ny))
            Vvec = np.mean(V,axis=0).reshape((nx,ny)) / np.std(V,axis=0).reshape((nx,ny))

    ###################################################
    # create eht-imaging image object

    # transfer observation data
    obs = modelinfo['obs']
    im = eh.image.Image(Ivec, psize=xspacing*eh.RADPERUAS, ra=obs.ra, dec=obs.dec, polrep='stokes', rf=obs.rf, source=obs.source, mjd=obs.mjd, time=np.mean(obs.data['time']))

    # populate image
    im.ivec = Ivec.T[::-1,::-1].ravel()
    if modelinfo['modeltype'] is 'polimage':
        im.qvec = Qvec.T[::-1,::-1].ravel()
        im.uvec = Uvec.T[::-1,::-1].ravel()
        im.vvec = Vvec.T[::-1,::-1].ravel()

    # check if image used a smoothing kernel, and if so apply it
    if modelinfo['smooth']:
        sigma = trace['sigma'][burnin:]
        sigma = sigma[div_mask]
        if moment == 'mean':
            fwhm_blur = 2.0*np.sqrt(2.0*np.log(2.0))*np.mean(sigma)
        if moment == 'median':
            fwhm_blur = 2.0*np.sqrt(2.0*np.log(2.0))*np.median(sigma)
        if moment == 'std':
            fwhm_blur = 2.0*np.sqrt(2.0*np.log(2.0))*np.std(sigma)
        if moment == 'snr':
            fwhm_blur = 2.0*np.sqrt(2.0*np.log(2.0))*np.mean(sigma)
        im = im.blur_circ(fwhm_blur*eh.RADPERUAS,fwhm_blur*eh.RADPERUAS)

    return im

def save_fits(modelinfo,moment,outfile,burnin=0):
    """ Save eht-imaging compatible fits file

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           moment (str): the type of posterior moment to save; choices are mean, median, std, snr
           outfile (str): name of output file
           burnin (int): length of burn-in
           
       Returns:
           None

    """

    # create eht-imaging image object
    im = make_image(modelinfo,moment,burnin=burnin)

    # save it
    im.save_fits(outfile)
