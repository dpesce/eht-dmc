#######################################################
# imports
#######################################################

from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh
import pymc3 as pm
import pickle
from tqdm import tqdm

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

def get_step_for_trace(trace=None, model=None, diag=False, regularize=True, regular_window=5, regular_variance=1e-3, **kwargs):
    """ Define a tuning procedure that adapts off-diagonal mass matrix terms
        adapted from a blog post by Dan Foreman-Mackey here:
        https://dfm.io/posts/pymc3-mass-matrix/

       Args:
           trace (trace): pymc3 trace object
           model (model): pymc3 model object
           
           diag (bool): flag to tune only the diagonal elements
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
    if diag:
        cov = np.diag(np.diag(cov))
    
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

def make_image(modelinfo,moment='mean',nx=100,ny=100,fov_scale=1.5,burnin=0,nsamps=None,mean_image=None):
    """ Make eht-imaging image object

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           moment (str): the type of posterior moment to save; choices are mean, std
           nx (int): number of image pixels in the x-direction
           ny (int): number of image pixels in the y-direction
           fov_scale (float): factor by which to increase the model FOV when making the image
           burnin (int): length of burn-in
           nsamps (int): number of chain samples to use when making the image
           mean_image (image): ehtim image object containing the mean image, if precomputed
           
       Returns:
           im: ehtim image object
    
    """
    if modelinfo['modeltype'] not in ['image','polimage']:
        raise Exception('modeltype is not image or polimage!')
    if moment not in ['mean','std']:
        raise Exception('moment ' + moment + ' not recognized!')
    if nx != ny:
        raise Exception('nx is not equal to ny!  eht-imaging does not support rectangular images.')

    ###################################################
    # organizing image information

    nx_input = modelinfo['nx']
    ny_input = modelinfo['ny']
    xmin = modelinfo['xmin']
    xmax = modelinfo['xmax']
    ymin = modelinfo['ymin']
    ymax = modelinfo['ymax']
    smooth = modelinfo['smooth']
    fit_smooth = modelinfo['fit_smooth']
    
    fullwidth_x = xmax - xmin
    pix_size_x = fullwidth_x / (nx-1)
    fov_x = nx*pix_size_x

    fullwidth_y = ymax - ymin
    pix_size_y = fullwidth_y / (ny-1)
    fov_y = ny*pix_size_y

    if ((smooth != None) & (fit_smooth == False)):
        sigma = smooth / (2.0*np.sqrt(2.0*np.log(2.0)))
    
    # one-dimensional pixel vectors for the model
    x_1d = np.linspace(xmin,xmax,nx_input)
    y_1d = np.linspace(ymin,ymax,ny_input)
    
    # one-dimensional pixel vectors for the image
    xim_1d = np.linspace(fov_scale*xmin,fov_scale*xmax,nx)
    yim_1d = np.linspace(fov_scale*ymin,fov_scale*ymax,ny)

    # two-dimensional pixel vectors for the image
    xim_2d, yim_2d = np.meshgrid(xim_1d,yim_1d,indexing='ij')
    x = xim_2d.ravel()
    y = yim_2d.ravel()

    # make edge arrays
    xspacing = np.mean(xim_1d[1:]-xim_1d[0:-1])
    x_edges_1d = np.append(xim_1d,xim_1d[-1]+xspacing) - (xspacing/2.0)
    yspacing = np.mean(yim_1d[1:]-yim_1d[0:-1])
    y_edges_1d = np.append(yim_1d,yim_1d[-1]+yspacing) - (yspacing/2.0)
    x_edges, y_edges = np.meshgrid(x_edges_1d,y_edges_1d,indexing='ij')

    # pixel size for the image
    psize_x_im = fov_scale*(xmax-xmin)/(nx-1)
    psize_y_im = fov_scale*(ymax-ymin)/(ny-1)

    ###################################################
    # organize chain info and compute moment

    trace = modelinfo['trace']

    # remove burnin
    I = trace['I'][burnin:]
    if (modelinfo['modeltype'] == 'polimage'):
        Q = trace['Q'][burnin:]
        U = trace['U'][burnin:]
        V = trace['V'][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)
    I = I[div_mask]
    if (modelinfo['modeltype'] == 'polimage'):
        Q = Q[div_mask]
        U = U[div_mask]
        V = V[div_mask]

    # initialize image vector
    Ivec = np.zeros(nx*ny)
    if (modelinfo['modeltype'] == 'polimage'):
        Qvec = np.zeros(nx*ny)
        Uvec = np.zeros(nx*ny)
        Vvec = np.zeros(nx*ny)

    # loop over samples
    if (modelinfo['modeltype'] == 'image'):

        # check if the mean image has already been provided
        if (mean_image is None):

            if (nsamps is not None):
                print('Computing mean using '+str(nsamps)+' samples from the chain...')
                for i in tqdm(range(nsamps)):

                    index = np.random.randint(len(I))
                    Ivec_here = I[index].reshape((nx_input,ny_input))

                    # loop over modeled pixels
                    for ix in range(len(x_1d)):
                        for iy in range(len(y_1d)):
                            Ivec += (Ivec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))

                # get average
                Ivec /= float(nsamps)

            # if number of samples isn't specified, loop over the entire chain
            else:
                print('Warning: nsamps is not specified; computing mean over the entire chain!')
                for i in tqdm(range(len(I))):

                    Ivec_here = I[i].reshape((nx_input,ny_input))

                    # loop over modeled pixels
                    for ix in range(len(x_1d)):
                        for iy in range(len(y_1d)):
                            Ivec += (Ivec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))

                # get average
                Ivec /= float(len(I))

        # if mean image has been provided, use it
        else:
            Ivec = mean_image.ivec.reshape((nx,ny)).T[::-1,::-1].ravel()

        # compute standard deviation
        if (moment == 'std'):
            Ivec_avg = np.copy(Ivec)

            # initialize image vector
            Ivec = np.zeros(nx*ny)

            # loop over samples
            if (nsamps is not None):
                print('Computing standard deviation using '+str(nsamps)+' samples from the chain...')
                for i in tqdm(range(nsamps)):

                    index = np.random.randint(len(I))
                    Ivec_here = I[index].reshape((nx_input,ny_input))

                    # loop over modeled pixels
                    imhere = np.zeros(nx*ny)
                    for ix in range(len(x_1d)):
                        for iy in range(len(y_1d)):
                            imhere += (Ivec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                    Ivec += (imhere - Ivec_avg)**2.0

                # get standard deviation
                Ivec = np.sqrt(Ivec/float(nsamps))

            # if number of samples isn't specified, loop over the entire chain
            else:
                print('Warning: nsamps is not specified; computing standard deviation over the entire chain!')
                for i in tqdm(range(len(I))):

                    Ivec_here = I[i].reshape((nx_input,ny_input))

                    # loop over modeled pixels
                    imhere = np.zeros(nx*ny)
                    for ix in range(len(x_1d)):
                        for iy in range(len(y_1d)):
                            imhere += (Ivec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                    Ivec += (imhere - Ivec_avg)**2.0

                # get average
                Ivec = np.sqrt(Ivec/float(len(I)))

        # reshape array
        Ivec = Ivec.reshape((nx,ny))

    # loop over samples
    elif (modelinfo['modeltype'] == 'polimage'):

        # check if the mean image has already been provided
        if (mean_image is None):

            if (nsamps is not None):
                print('Computing mean using '+str(nsamps)+' samples from the chain...')
                for i in tqdm(range(nsamps)):

                    index = np.random.randint(len(I))
                    Ivec_here = I[index].reshape((nx_input,ny_input))
                    Qvec_here = Q[index].reshape((nx_input,ny_input))
                    Uvec_here = U[index].reshape((nx_input,ny_input))
                    Vvec_here = V[index].reshape((nx_input,ny_input))

                    # loop over modeled pixels
                    for ix in range(len(x_1d)):
                        for iy in range(len(y_1d)):
                            Ivec += (Ivec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Qvec += (Qvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Uvec += (Uvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Vvec += (Vvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))

                # get average
                Ivec /= float(nsamps)
                Qvec /= float(nsamps)
                Uvec /= float(nsamps)
                Vvec /= float(nsamps)

            # if number of samples isn't specified, loop over the entire chain
            else:
                print('Warning: nsamps is not specified; computing mean over the entire chain!')
                for i in tqdm(range(len(I))):

                    Ivec_here = I[i].reshape((nx_input,ny_input))
                    Qvec_here = Q[i].reshape((nx_input,ny_input))
                    Uvec_here = U[i].reshape((nx_input,ny_input))
                    Vvec_here = V[i].reshape((nx_input,ny_input))

                    # loop over modeled pixels
                    for ix in range(len(x_1d)):
                        for iy in range(len(y_1d)):
                            Ivec += (Ivec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Qvec += (Qvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Uvec += (Uvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Vvec += (Vvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))

                # get average
                Ivec /= float(len(I))
                Qvec /= float(len(Q))
                Uvec /= float(len(U))
                Vvec /= float(len(V))

        # if mean image has been provided, use it
        else:
            Ivec = mean_image.ivec.reshape((nx,ny)).T[::-1,::-1].ravel()
            Qvec = mean_image.qvec.reshape((nx,ny)).T[::-1,::-1].ravel()
            Uvec = mean_image.uvec.reshape((nx,ny)).T[::-1,::-1].ravel()
            Vvec = mean_image.vvec.reshape((nx,ny)).T[::-1,::-1].ravel()

        # compute standard deviation
        if (moment == 'std'):
            Ivec_avg = np.copy(Ivec)
            Qvec_avg = np.copy(Qvec)
            Uvec_avg = np.copy(Uvec)
            Vvec_avg = np.copy(Vvec)

            # initialize image vector
            Ivec = np.zeros(nx*ny)
            Qvec = np.zeros(nx*ny)
            Uvec = np.zeros(nx*ny)
            Vvec = np.zeros(nx*ny)

            # loop over samples
            if (nsamps is not None):
                print('Computing standard deviation using '+str(nsamps)+' samples from the chain...')
                for i in tqdm(range(nsamps)):

                    index = np.random.randint(len(I))
                    Ivec_here = I[index].reshape((nx_input,ny_input))
                    Qvec_here = Q[index].reshape((nx_input,ny_input))
                    Uvec_here = U[index].reshape((nx_input,ny_input))
                    Vvec_here = V[index].reshape((nx_input,ny_input))

                    # loop over modeled pixels
                    Ihere = np.zeros(nx*ny)
                    Qhere = np.zeros(nx*ny)
                    Uhere = np.zeros(nx*ny)
                    Vhere = np.zeros(nx*ny)
                    for ix in range(len(x_1d)):
                        for iy in range(len(y_1d)):
                            Ihere += (Ivec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Qhere += (Qvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Uhere += (Uvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Vhere += (Vvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))

                    Ivec += (Ihere - Ivec_avg)**2.0
                    Qvec += (Qhere - Qvec_avg)**2.0
                    Uvec += (Uhere - Uvec_avg)**2.0
                    Vvec += (Vhere - Vvec_avg)**2.0

                # get standard deviation
                Ivec = np.sqrt(Ivec/float(nsamps))
                Qvec = np.sqrt(Qvec/float(nsamps))
                Uvec = np.sqrt(Uvec/float(nsamps))
                Vvec = np.sqrt(Vvec/float(nsamps))

            # if number of samples isn't specified, loop over the entire chain
            else:
                print('Warning: nsamps is not specified; computing standard deviation over the entire chain!')
                for i in tqdm(range(len(I))):

                    Ivec_here = I[i].reshape((nx_input,ny_input))
                    Qvec_here = Q[i].reshape((nx_input,ny_input))
                    Uvec_here = U[i].reshape((nx_input,ny_input))
                    Vvec_here = V[i].reshape((nx_input,ny_input))

                    # loop over modeled pixels
                    Ihere = np.zeros(nx*ny)
                    Qhere = np.zeros(nx*ny)
                    Uhere = np.zeros(nx*ny)
                    Vhere = np.zeros(nx*ny)
                    for ix in range(len(x_1d)):
                        for iy in range(len(y_1d)):
                            Ihere += (Ivec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Qhere += (Qvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Uhere += (Uvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))
                            Vhere += (Vvec_here[ix,iy]/(2.0*np.pi*(sigma/psize_x_im)*(sigma/psize_y_im)))*np.exp(-((((x-x_1d[ix])**2.0) + ((y-y_1d[iy])**2.0)) / (2.0*(sigma**2.0))))

                    Ivec += (Ihere - Ivec_avg)**2.0
                    Qvec += (Qhere - Qvec_avg)**2.0
                    Uvec += (Uhere - Uvec_avg)**2.0
                    Vvec += (Vhere - Vvec_avg)**2.0

                # get average
                Ivec = np.sqrt(Ivec/float(len(I)))
                Qvec = np.sqrt(Qvec/float(len(Q)))
                Uvec = np.sqrt(Uvec/float(len(U)))
                Vvec = np.sqrt(Vvec/float(len(V)))

        # reshape array
        Ivec = Ivec.reshape((nx,ny))
        Qvec = Qvec.reshape((nx,ny))
        Uvec = Uvec.reshape((nx,ny))
        Vvec = Vvec.reshape((nx,ny))

    ###################################################
    # create eht-imaging image object

    # transfer observation data
    obs = modelinfo['obs']
    im = eh.image.Image(Ivec, psize=xspacing*eh.RADPERUAS, ra=obs.ra, dec=obs.dec, polrep='stokes', rf=obs.rf, source=obs.source, mjd=obs.mjd, time=np.mean(obs.data['time']))

    # populate image
    im.ivec = Ivec.T[::-1,::-1].ravel()
    if (modelinfo['modeltype'] == 'polimage'):
        im.qvec = Qvec.T[::-1,::-1].ravel()
        im.uvec = Uvec.T[::-1,::-1].ravel()
        im.vvec = Vvec.T[::-1,::-1].ravel()

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
