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

def plot_trace(modelinfo,**kwargs):
    """ Plot parameter traces

       Args:
           modelinfo (modelinfo): dmc modelinfo object
           
       Returns:
           traceplot: figure containing the trace plot

    """

    traceplot = pm.plots.traceplot(modelinfo['trace'],**kwargs)

    return traceplot

def plot_image(modelinfo,imtype,moment,burnin=0,title=None):
    """ Plot image pixel parameters

       Args:
           modelinfo (modelinfo): dmc modelinfo object
           imtype (str): the type of image to plot; choices are I, Q, U, V, p, EVPA
           moment (str): the type of posterior moment to plot; choices are mean, median, std, snr
           burnin (int): length of burn-in
           title (str): plot title
           
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
        im = np.mean(imvec,axis=0).reshape((nx,ny)) / np.std(imvec,axis=0).reshape((nx,ny))

    ###################################################
    # create figure

    imageplot = plt.figure(figsize=(6,5))
    ax = imageplot.add_axes([0.1,0.1,0.75,0.75])

    if imtype in ['I','p']:
        vmin = 0
        vmax = np.max(im)
        implot = ax.pcolormesh(x_edges,y_edges,im,vmin=vmin,vmax=vmax,cmap='afmhot_us')
    else:
        vmin = np.min([-np.abs(np.min(im)),-np.abs(np.max(im))])
        vmax = np.max([np.abs(np.min(im)),np.abs(np.max(im))])
        implot = ax.pcolormesh(x_edges,y_edges,im,vmin=vmin,vmax=vmax,cmap='seismic')

    ax.set_xlim((xmax+(xspacing/2.0),xmin-(xspacing/2.0)))
    ax.set_ylim((ymin-(xspacing/2.0),ymax+(xspacing/2.0)))
    ax.set_xlabel(r'RA ($\mu$as)')
    ax.set_ylabel(r'Dec ($\mu$as)')
    ax.set_aspect(1)

    if title is not None:
        ax.set_title(title)

    axpos = ax.get_position().bounds
    cax = imageplot.add_axes([axpos[0]+axpos[2]+0.03,axpos[1],0.03,axpos[3]])
    cbar = plt.colorbar(mappable=implot,cax=cax)
    
    return imageplot

def plot_gains(modelinfo,gaintype,burnin=0):
    """ Plot image pixel parameters

       Args:
           modelinfo (modelinfo): dmc modelinfo object
           gain (str): the type of gain to plot; choices are amp, phase
           burnin (int): length of burn-in
           
       Returns:
           gainplot: figure containing the gain plot

    """

    if gaintype not in ['amp','phase']:
        raise Exception('gaintype ' + gaintype + ' not recognized!')

    ###################################################
    # organize chain info

    trace = modelinfo['trace']

    # read gains
    if gaintype == 'amp':
        gain_R = trace['right_gain_amps'][burnin:]        
        gain_L = trace['left_gain_amps'][burnin:]
    if gaintype == 'phase':
        gain_R = trace['right_gain_phases'][burnin:]
        gain_L = trace['left_gain_phases'][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)

    # compute moments
    std_R = np.std(gain_R[div_mask],axis=0)
    med_R = np.median(gain_R[div_mask],axis=0)
    std_L = np.std(gain_L[div_mask],axis=0)
    med_L = np.median(gain_L[div_mask],axis=0)

    # additional gain info
    T_gains = modelinfo['T_gains']
    A_gains = modelinfo['A_gains']

    ###################################################
    # create figure

    gainplot = plt.figure(figsize=(6,6))
    ax = gainplot.add_axes([0.12,0.1,0.8,0.8])

    # initial offset and vertical axis range
    ymax = 0.0
    offset = 0.0
    if gaintype == 'amp':
        offset_increment = 1.0
    if gaintype == 'phase':
        offset_increment = 3.0

    # loop through all stations
    station_labels = list()
    for ant in SEFD_error_budget.keys():

        # get current station
        index = (A_gains == ant)

        # plot gains
        ax.plot(T_gains[index],med_R[index]+offset,linewidth=0,marker='o',markersize=3)
        ax.plot(T_gains[index],med_L[index]+offset,linewidth=0,marker='s',markersize=3)
        
        if index.sum() > 0:

            # plot horizontal line
            if gaintype == 'amp':
                ax.plot([-100.0,100.0],[1.0+offset,1.0+offset],'k--',linewidth=0.5,alpha=0.3,zorder=-11)
            if gaintype == 'phase':
                ax.plot([-100.0,100.0],[offset,offset],'k--',linewidth=0.5,alpha=0.3,zorder=-11)
                
            # increment vertical axis range
            if np.max(med_R[index]+offset) > ymax:
                ymax = np.max(med_R[index]+offset)
            if np.max(med_L[index]+offset) > ymax:
                ymax = np.max(med_L[index]+offset)

            # track stations
            station_labels.append(ant)

        # increment offset
        offset += offset_increment

    # set axis ranges and labels
    ax.set_xlim(np.min(T_gains)-0.5,np.max(T_gains)+0.5)
    ax.set_xlabel('Time (hours)')

    if gaintype == 'amp':
        ax.set_ylim(0,ymax+1)
        ax.set_ylabel('Gain amplitudes')
    if gaintype == 'phase':
        ax.set_ylim(-np.pi,ymax+np.pi)
        ax.set_ylabel('Gain phases (radians)')

    ylim = ax.get_ylim()

    # label stations on second axis
    ax2 = ax.twinx()
    if gaintype == 'amp':
        ax2.set_yticks(np.arange(1.0,offset_increment*(len(station_labels)+1),offset_increment))
    if gaintype == 'phase':
        ax2.set_yticks(np.arange(0.0,offset_increment*len(station_labels),offset_increment))
    ax2.set_yticklabels(station_labels)
    ax2.set_ylim(ylim)
    
    return gainplot