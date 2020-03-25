#######################################################
# imports
#######################################################

from __future__ import division
from __future__ import print_function
from builtins import list
from builtins import len
from builtins import range
from builtins import enumerate
from builtins import Exception

import numpy as np
import ehtim as eh
import pickle

#######################################################
# functions
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
    Q = trace['Q'][burnin:]
    U = trace['U'][burnin:]
    V = trace['V'][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)
    I = I[div_mask]
    Q = Q[div_mask]
    U = U[div_mask]
    V = V[div_mask]

    # reshape array
    if moment == 'mean':
        Ivec = np.mean(I,axis=0).reshape((nx,ny))
        Qvec = np.mean(Q,axis=0).reshape((nx,ny))
        Uvec = np.mean(U,axis=0).reshape((nx,ny))
        Vvec = np.mean(V,axis=0).reshape((nx,ny))
    elif moment == 'median':
        Ivec = np.median(I,axis=0).reshape((nx,ny))
        Qvec = np.median(Q,axis=0).reshape((nx,ny))
        Uvec = np.median(U,axis=0).reshape((nx,ny))
        Vvec = np.median(V,axis=0).reshape((nx,ny))
    elif moment == 'std':
        Ivec = np.std(I,axis=0).reshape((nx,ny))
        Qvec = np.std(Q,axis=0).reshape((nx,ny))
        Uvec = np.std(U,axis=0).reshape((nx,ny))
        Vvec = np.std(V,axis=0).reshape((nx,ny))
    elif moment == 'snr':
        Ivec = np.mean(I,axis=0).reshape((nx,ny)) / np.std(I,axis=0).reshape((nx,ny))
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


