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
import pymc3 as pm
import matplotlib.pyplot as plt
import ehtplot

#######################################################
# functions
#######################################################

def plot_trace(modelinfo,**kwargs):
    """ Plot parameter traces

       Args:
           modelinfo (modelinfo): dmc modelinfo object
           
       Returns:
           traceplot: figure containing the trace plot

    """

    traceplot = pm.plots.traceplot(modelinfo['trace'],**kwargs)

    return traceplot

def plot_image(trace,imtype,moment,burnin=0):
    """ Plot image pixel parameters

       Args:
           modelinfo (modelinfo): dmc modelinfo object
           imtype (str): the type of image to plot; choices are I, Q, U, V, p, EVPA
           moment (str): the type of posterior moment to plot; choices are mean, median, std, snr
           burnin (int): length of burn-in
           
       Returns:
           imageplot: figure containing the image plot

    """

    if imtype not in ['I','Q','U','V','p','EVPA']:
        raise Exception('imtype ' + imtype + ' not recognized!')
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
    # organize chain info and compute image moment

    trace = modelinfo['trace']

    # remove burnin
    imvec = trace[imtype][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)
    imvec = imvec[div_mask]

    # reshape array
    if moment == 'mean':
        im = np.mean(imvec,axis=0).reshape((nx,ny))
    elif moment == 'median':
        im = np.median(imvec,axis=0).reshape((nx,ny))
    elif moment == 'std':
        im = np.std(imvec,axis=0).reshape((nx,ny))
    elif moment == 'snr':
        im = np.std(imvec,axis=0).reshape((nx,ny)) / np.mean(imvec,axis=0).reshape((nx,ny))

    ###################################################
    # create figure

    imageplot = plt.figure(figsize=(6,6))
    ax = imageplot.add_axes([0.1,0.1,0.8,0.8])

    if imtype in ['I','p']:
        ax.pcolormesh(x_edges,y_edges,im,vmin=0,cmap='afmhot_us')
    else:
        ax.pcolormesh(x_edges,y_edges,im,cmap='seismic')

    ax.set_xlim((xmax,xmin))
    ax.set_ylim((ymin,ymax))
    ax.set_xlabel(r'RA ($\mu$as)')
    ax.set_ylabel(r'Dec ($\mu$as)')
    ax.colorbar()
    
    return imageplot