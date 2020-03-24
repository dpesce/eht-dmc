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