#######################################################
# imports
#######################################################

from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh
import pymc3 as pm
import matplotlib.pyplot as plt
import ehtplot
import corner
import os

from . import model_utils as mu

#######################################################
# functions
#######################################################

def plot_trace(modelinfo,**kwargs):
    """ Plot parameter traces

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           
       Returns:
           traceplot: figure containing the trace plot

    """

    traceplot = pm.plots.traceplot(modelinfo['trace'],**kwargs)

    return traceplot

def plot_image(modelinfo,imtype,moment,burnin=0,title=None):
    """ Plot image

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           imtype (str): the type of image to plot; choices are I, Q, U, V, p, EVPA
           moment (str): the type of posterior moment to plot; choices are mean, median, std, snr
           burnin (int): length of burn-in
           title (str): plot title
           
       Returns:
           imageplot: figure containing the image plot

    """

    if modelinfo['modeltype'] not in ['image','polimage']:
        raise Exception('modeltype is not image or polimage!')
    if imtype not in ['I','Q','U','V','p','EVPA']:
        raise Exception('imtype ' + imtype + ' not recognized!')
    if moment not in ['mean','median','std','snr']:
        raise Exception('moment ' + moment + ' not recognized!')
    if (modelinfo['modeltype'] == 'image') & (imtype != 'I'):
        raise Exception('modeltype image does not contain '+imtype+'!')

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

def plot_polimage(modelinfo,moment,burnin=0,regrid=64,smooth=0.0,pcut=0.1,skip=4,Pmin=0.0,Pmax=0.3,cmap='gray_r',cmap2='rainbow'):
    """ Plot polarized image with Stokes I and EVPA ticks
        Adapted from script originally created by S. Issaoun

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           moment (str): the type of posterior moment to plot; choices are mean, median, std, snr
           burnin (int): length of burn-in
           regrid (int): number of pixels to regrid the image by; set to None for no regridding
           smooth (float): FWHM of Gaussian smoothing kernel (uas)
           pcut (float): polarization fraction plotting threshold
           skip (int): tick plotting interval, in pixels
           Pmin (float): minimum polarization fraction for colorbar
           Pmax (float): maximum polarization fraction for colorbar
           cmap (str): colormap for background Stokes I image
           cmap2 (str): colormap for polarization ticks

       Returns:
           imageplot: figure containing the image plot

    """

    if modelinfo['modeltype'] is not 'polimage':
        raise Exception('modeltype is not polimage!')
    if moment not in ['mean','median','std','snr']:
        raise Exception('moment ' + moment + ' not recognized!')

    ###################################################
    # organize polarization info

    # create eht-imaging image object
    im = mu.make_image(modelinfo,moment,burnin=burnin)

    # regrid and smooth image
    if regrid is not None:
        im = im.regrid_image(im.psize*im.xdim, regrid, interp='linear')
    if smooth > 0.0:
        im = im.blur_circ(smooth*eh.RADPERUAS,smooth*eh.RADPERUAS)

    # pixel size and FOV, in uas
    pixel = im.psize/eh.RADPERUAS
    FOV = pixel*im.xdim

    # generate 2D grids for the x & y bounds
    y, x = np.mgrid[slice(-FOV/2, FOV/2, pixel),
                    slice(-FOV/2, FOV/2, pixel)]

    Iarr = im.imvec.reshape(im.ydim, im.xdim)
    Qarr = im.qvec.reshape(im.ydim, im.xdim)
    Uarr = im.uvec.reshape(im.ydim, im.xdim)

    # tick length proportional to P = sqrt(Q^2+U^2)
    P = np.sqrt((im.qvec**2.0) + (im.uvec**2.0))
    scale = np.max(P)    

    # tick 'velocities'
    vx = (-np.sin(np.angle(im.qvec+1j*im.uvec)/2)*P/scale).reshape(im.ydim, im.xdim)
    vy = ( np.cos(np.angle(im.qvec+1j*im.uvec)/2)*P/scale).reshape(im.ydim, im.xdim)

    # tick color will be proportional to polarization fraction
    m = (P/im.imvec).reshape(im.xdim, im.ydim)

    # mask arrays based on polarization fraction threshold criterion
    m = np.ma.masked_where(Iarr < pcut * np.max(Iarr), m)
    x_masked = np.ma.masked_where(Iarr < pcut * np.max(Iarr), x)
    y_masked = np.ma.masked_where(Iarr < pcut * np.max(Iarr), y)
    vx = np.ma.masked_where(Iarr < pcut * np.max(Iarr), vx)
    vy = np.ma.masked_where(Iarr < pcut * np.max(Iarr), vy)

    ###################################################
    # create figure

    imageplot = plt.figure(figsize=(6,5))
    ax = imageplot.add_axes([0.1,0.1,0.75,0.75])

    # plot background Stokes I image
    ax.pcolormesh(-x,-y,Iarr,cmap=cmap,vmin=0,vmax=np.max(Iarr))

    # set axis ranges and labels
    ax.set_xlim(FOV/2,-FOV/2)
    ax.set_ylim(-FOV/2,FOV/2)
    ax.set_xlabel(r'RA ($\mu$as)')
    ax.set_ylabel(r'Dec ($\mu$as)')
    ax.set_aspect(1)

    # plot ticks
    qa = ax.quiver(-x_masked[::skip, ::skip],-y_masked[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
               m[::skip,::skip],
               headlength=0,
               headwidth = 1,
               pivot='mid',
               width=0.01,
               cmap=cmap2,
               scale=16,
               clim=(Pmin,Pmax))

    # add colorbar
    axpos = ax.get_position().bounds
    cax = imageplot.add_axes([axpos[0]+axpos[2]+0.03,axpos[1],0.03,axpos[3]])
    cbar = plt.colorbar(mappable=qa,cax=cax,label='Fractional Polarization')

    return imageplot

def plot_gains(modelinfo,gaintype,burnin=0):
    """ Plot gain amplitudes or phases over time for each station

       Args:
           modelinfo (dict): dmc modelinfo dictionary
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
    if modelinfo['modeltype'] in ['polpoint','polimage']:
        if gaintype == 'amp':
            gain_R = trace['right_gain_amps'][burnin:]        
            gain_L = trace['left_gain_amps'][burnin:]
        if gaintype == 'phase':
            gain_R = trace['right_gain_phases'][burnin:]
            gain_L = trace['left_gain_phases'][burnin:]
    else:
        if gaintype == 'amp':
            gain = trace['gain_amps'][burnin:]
        if gaintype == 'phase':
            gain = trace['gain_phases'][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)

    # compute moments
    if modelinfo['modeltype'] in ['polpoint','polimage']:
        std_R = np.std(gain_R[div_mask],axis=0)
        med_R = np.median(gain_R[div_mask],axis=0)
        std_L = np.std(gain_L[div_mask],axis=0)
        med_L = np.median(gain_L[div_mask],axis=0)
    else:
        med = np.median(gain[div_mask],axis=0)
        std = np.std(gain[div_mask],axis=0)

    # additional gain info
    T_gains = modelinfo['T_gains']
    A_gains = modelinfo['A_gains']
    stations = modelinfo['stations']

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
    for ant in stations:

        # get current station
        index = (A_gains == ant)

        # plot gains
        if modelinfo['modeltype'] in ['polpoint','polimage']:
            ax.plot(T_gains[index],med_R[index]+offset,linewidth=0,marker='o',markersize=3)
            ax.plot(T_gains[index],med_L[index]+offset,linewidth=0,marker='s',markersize=3)
        else:
            ax.plot(T_gains[index],med[index]+offset,linewidth=0,marker='o',markersize=3)
        
        if index.sum() > 0:

            # plot horizontal line
            if gaintype == 'amp':
                ax.plot([-100.0,100.0],[1.0+offset,1.0+offset],'k--',linewidth=0.5,alpha=0.3,zorder=-11)
            if gaintype == 'phase':
                ax.plot([-100.0,100.0],[offset,offset],'k--',linewidth=0.5,alpha=0.3,zorder=-11)
                
            # increment vertical axis range
            if modelinfo['modeltype'] in ['polpoint','polimage']:
                if np.max(med_R[index]+offset) > ymax:
                    ymax = np.max(med_R[index]+offset)
                if np.max(med_L[index]+offset) > ymax:
                    ymax = np.max(med_L[index]+offset)
            else:
                if np.max(med[index]+offset) > ymax:
                    ymax = np.max(med[index]+offset)

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

def gain_cornerplots(modelinfo,gaintype,burnin=0,dirname=None,levels=None,smooth=1.0):
    """ Make a cornerplot of the gain amplitudes or phases for all timestamps

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           gain (str): the type of gain to plot; choices are amp, phase
           burnin (int): length of burn-in
           dirname (str): name of the directory in which to save the cornerplots
           levels (list): a list of contour levels to plot
           smooth (float): amount by which to KDE smooth the histograms

       Returns:
           None

    """

    if gaintype not in ['amp','phase']:
        raise Exception('gaintype ' + gaintype + ' not recognized!')

    ###################################################
    # make directory in which to save plots

    if dirname is None:
        if gaintype == 'amp':
            dirname = './gain_amplitudes'
        if gaintype == 'phase':
            dirname = './gain_phases'

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    ###################################################
    # organize chain info

    trace = modelinfo['trace']
    T_gains = modelinfo['T_gains']
    A_gains = modelinfo['A_gains']
    timestamps = np.sort(np.unique(T_gains))

    # read gains
    if modelinfo['modeltype'] in ['polpoint','polimage']:
        if gaintype == 'amp':
            gain_R = trace['right_gain_amps'][burnin:]        
            gain_L = trace['left_gain_amps'][burnin:]
        if gaintype == 'phase':
            gain_R = trace['right_gain_phases'][burnin:]
            gain_L = trace['left_gain_phases'][burnin:]
    else:
        if gaintype == 'amp':
            gain = trace['gain_amps'][burnin:]
        if gaintype == 'phase':
            gain = trace['gain_phases'][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)
    if modelinfo['modeltype'] in ['polpoint','polimage']:
        gain_R = gain_R[div_mask]
        gain_L = gain_L[div_mask]
    else:
        gain = gain[div_mask]

    ###################################################
    # plot and save cornerplots

    # contour levels
    if levels is None:
        levels = [1.0-np.exp(-0.5*(1.1775**2.0)),1.0-np.exp(-0.5*(2.146**2.0)),1.0-np.exp(-0.5*(3.035**2.0))]

    # loop over all unique timestamps
    count = 0
    for it, t in enumerate(timestamps):
        
        # find the stations observing at this time
        ind_here = (T_gains == t)
        ants_here = A_gains[ind_here]

        # initialize arrays of samples
        if modelinfo['modeltype'] in ['polpoint','polimage']:
            samples_R = np.ndarray(shape=(len(gain_R[:,count]),len(ants_here)))
            samples_L = np.ndarray(shape=(len(gain_L[:,count]),len(ants_here)))
        else:
            samples = np.ndarray(shape=(len(gain[:,count]),len(ants_here)))

        # loop over all stations
        labels = list()
        ranges = list()
        for ia, ant in enumerate(ants_here):

            # extract the relevant chain samples
            if modelinfo['modeltype'] in ['polpoint','polimage']:
                samples_R[:,ia] = gain_R[:,count]
                samples_L[:,ia] = gain_L[:,count]
            else:
                samples[:,ia] = gain[:,count]

            # make plot labels + axis ranges
            if gaintype == 'amp':
                labels.append(r'$|G|_{\rm{'+ant+'}}$')
                ranges.append((0.0,2.0))
            if gaintype == 'phase':
                labels.append(r'$\theta_{\rm{'+ant+'}}$')
                ranges.append((-np.pi,np.pi))

            # increment counter
            count += 1

        if modelinfo['modeltype'] in ['polpoint','polimage']:
            fig = corner.corner(samples_R,labels=labels,show_titles=False,title_fmt='.4f',levels=levels,
                                title_kwargs={"fontsize": 12},smooth=smooth,smooth1d=smooth,plot_datapoints=False,
                                plot_density=False,fill_contours=True,range=ranges,bins=100,color='cornflowerblue')
            corner.corner(samples_L,fig=fig,smooth=smooth,smooth1d=smooth,plot_datapoints=False,levels=levels,
                                plot_density=False,fill_contours=True,range=ranges,bins=100,color='salmon')
        else:
            fig = corner.corner(samples,labels=labels,show_titles=False,title_fmt='.4f',levels=levels,
                                title_kwargs={"fontsize": 12},smooth=smooth,smooth1d=smooth,plot_datapoints=False,
                                plot_density=False,fill_contours=True,range=ranges,bins=100,color='cornflowerblue')
        fig.savefig(dirname+'/gains_scan'+str(it).zfill(5)+'.png',dpi=300)
        plt.close()

def plot_dterms(modelinfo,station,burnin=0,print_dterms=True,levels=None,smooth=1.0):
    """ Plot right and left D-terms for a single station

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           station (str): the station whose D-terms to plot
           burnin (int): length of burn-in
           print_dterms (bool): flag to print the D-terms and uncertainties on the plot
           levels (list): a list of contour levels to plot
           smooth (float): amount by which to KDE smooth the histograms
           
       Returns:
           dtermplot: figure containing the D-term plot

    """

    # contour levels
    if levels is None:
        levels = [1.0-np.exp(-0.5*(1.1775**2.0)),1.0-np.exp(-0.5*(2.146**2.0)),1.0-np.exp(-0.5*(3.035**2.0))]

    ###################################################
    # organize chain info

    trace = modelinfo['trace']
    stations = modelinfo['stations']

    # get index for requested station
    istat = (stations == station)

    # read D-terms
    Dterm_reals_R = trace['right_Dterm_reals'][burnin:]
    Dterm_reals_L = trace['left_Dterm_reals'][burnin:]
    Dterm_imags_R = trace['right_Dterm_imags'][burnin:]
    Dterm_imags_L = trace['left_Dterm_imags'][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)
    Dterm_reals_R = Dterm_reals_R[div_mask]
    Dterm_reals_L = Dterm_reals_L[div_mask]
    Dterm_imags_R = Dterm_imags_R[div_mask]
    Dterm_imags_L = Dterm_imags_L[div_mask]

    ###################################################
    # make figure

    dtermplot = plt.figure(figsize=(6,6))
    ax = dtermplot.add_axes([0.18,0.18,0.8,0.8])

    # add axis lines
    ax.plot([-10,10],[0,0],'k--',alpha=0.5,zorder=-10)
    ax.plot([0,0],[-10,10],'k--',alpha=0.5,zorder=-10)

    # plot contours
    corner.hist2d(Dterm_reals_R[:,istat],Dterm_imags_R[:,istat],fig=dtermplot, levels=levels, color='cornflowerblue', fill_contours=True,plot_datapoints=False,plot_density=False,plot_contours=True,smooth=smooth)
    corner.hist2d(Dterm_reals_L[:,istat],Dterm_imags_L[:,istat],fig=dtermplot, levels=levels, color='salmon', fill_contours=True,plot_datapoints=False,plot_density=False,plot_contours=True,smooth=smooth)
    
    # axis labels
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')

    # axis ranges
    limit = np.max(np.array([np.max(np.abs(Dterm_reals_R[:,istat])),np.max(np.abs(Dterm_imags_R[:,istat])),np.max(np.abs(Dterm_reals_L[:,istat])),np.max(np.abs(Dterm_imags_L[:,istat]))]))
    ax.set_xlim(-1.1*limit,1.1*limit)
    ax.set_ylim(-1.1*limit,1.1*limit)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')

    # print out D-term values and uncertainties
    if print_dterms:
        realpart_R = np.percentile(Dterm_reals_R[:,istat],50.0)
        realpart_R_lo = np.percentile(Dterm_reals_R[:,istat],50.0) - np.percentile(Dterm_reals_R[:,istat],15.87)
        realpart_R_hi = np.percentile(Dterm_reals_R[:,istat],84.13) - np.percentile(Dterm_reals_R[:,istat],50.0)

        realpart_L = np.percentile(Dterm_reals_L[:,istat],50.0)
        realpart_L_lo = np.percentile(Dterm_reals_L[:,istat],50.0) - np.percentile(Dterm_reals_L[:,istat],15.87)
        realpart_L_hi = np.percentile(Dterm_reals_L[:,istat],84.13) - np.percentile(Dterm_reals_L[:,istat],50.0)

        imagpart_R = np.percentile(Dterm_imags_R[:,istat],50.0)
        imagpart_R_lo = np.percentile(Dterm_imags_R[:,istat],50.0) - np.percentile(Dterm_imags_R[:,istat],15.87)
        imagpart_R_hi = np.percentile(Dterm_imags_R[:,istat],84.13) - np.percentile(Dterm_imags_R[:,istat],50.0)

        imagpart_L = np.percentile(Dterm_imags_L[:,istat],50.0)
        imagpart_L_lo = np.percentile(Dterm_imags_L[:,istat],50.0) - np.percentile(Dterm_imags_L[:,istat],15.87)
        imagpart_L_hi = np.percentile(Dterm_imags_L[:,istat],84.13) - np.percentile(Dterm_imags_L[:,istat],50.0)

        str1 = r'$\rm{Re}(D_R) = ' + str(np.round(realpart_R,5)) + r'_{-' + str(np.round(realpart_R_lo,5)) + r'}^{+' + str(np.round(realpart_R_hi,5)) + r'}$'
        str2 = r'$\rm{Im}(D_R) = ' + str(np.round(imagpart_R,5)) + r'_{-' + str(np.round(imagpart_R_lo,5)) + r'}^{+' + str(np.round(imagpart_R_hi,5)) + r'}$'
        str3 = r'$\rm{Re}(D_L) = ' + str(np.round(realpart_L,5)) + r'_{-' + str(np.round(realpart_L_lo,5)) + r'}^{+' + str(np.round(realpart_L_hi,5)) + r'}$'
        str4 = r'$\rm{Im}(D_L) = ' + str(np.round(imagpart_L,5)) + r'_{-' + str(np.round(imagpart_L_lo,5)) + r'}^{+' + str(np.round(imagpart_L_hi,5)) + r'}$'
        strhere = str1 + '\n' + str2 + '\n' + str3 + '\n' + str4
        ax.text(-1.05*limit,1.05*limit,strhere,ha='left',va='top',fontsize=8)

    return dtermplot

def plot_energy(modelinfo,burnin=0):
    """ Plot HMC energy distribution

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           burnin (int): length of burn-in
           
       Returns:
           energyplot: figure containing the energy plot

    """

    ###################################################
    # organize chain info

    trace = modelinfo['trace']

    # read energy
    energy = trace[burnin:].energy
    energy_difference = np.diff(energy)

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)
    energy = energy[div_mask] - np.mean(energy[div_mask])
    energy_difference = energy_difference[div_mask[:-1]]

    ###################################################
    # make figure

    energyplot = plt.figure(figsize=(6,6))
    ax = energyplot.add_axes([0.1,0.1,0.8,0.8])

    # histogram bounds
    xupper = 1.1*max([np.max(energy),np.max(energy_difference)])
    xlower = 1.1*min([np.min(energy),np.min(energy_difference)])

    # make histograms
    ax.hist(energy,label='energy',alpha=0.5,bins=50,range=(xlower,xupper),density=True)
    ax.hist(energy_difference,label='energy difference',alpha=0.5,bins=50,range=(xlower,xupper),density=True)

    # label axis and make legend
    ax.set_xlabel('Energy')
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.legend()

    return energyplot

def plot_stepsize(modelinfo,burnin=0):
    """ Plot step size of chain

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           burnin (int): length of burn-in
           
       Returns:
           stepplot: figure containing the energy plot

    """

    ###################################################
    # organize chain info

    trace = modelinfo['trace']

    # get step sizes
    stepsize = trace.get_sampler_stats('step_size')

    ###################################################
    # make figure

    stepplot = plt.figure(figsize=(6,6))
    ax = stepplot.add_axes([0.15,0.1,0.8,0.8])

    ax.plot(stepsize,'b-')
    ax.semilogy()

    # label axes
    ax.set_ylabel('Step size')
    ax.set_xlabel('Trial number')
    
    return stepplot

def plot_polimtot(modelinfo,outname,burnin=0,levels=None):
    """ Make a cornerplot of image-integrated quantities

       Args:
           modelinfo (dict): dmc modelinfo dictionary
           outname (str): name of the output file
           burnin (int): length of burn-in
           levels (list): a list of contour levels to plot
           
       Returns:
           None

    """

    if modelinfo['modeltype'] not in ['polimage']:
        raise Exception('modeltype is not polimage!')

    # contour levels
    if levels is None:
        levels = [1.0-np.exp(-0.5*(1.1775**2.0)),1.0-np.exp(-0.5*(2.146**2.0)),1.0-np.exp(-0.5*(3.035**2.0))]

    ###################################################
    # organize chain info

    trace = modelinfo['trace']

    # remove burnin
    Ivec = trace['I'][burnin:]
    Qvec = trace['Q'][burnin:]
    Uvec = trace['U'][burnin:]
    Vvec = trace['V'][burnin:]

    # remove divergences
    div_mask = np.invert(trace[burnin:].diverging)
    Ivec = Ivec[div_mask]
    Qvec = Qvec[div_mask]
    Uvec = Uvec[div_mask]
    Vvec = Vvec[div_mask]

    ###################################################
    # construct image-integrated quantities

    # initialize arrays
    pvec = np.zeros(len(Ivec))
    EVPAvec = np.zeros(len(Ivec))
    vvec = np.zeros(len(Ivec))

    # loop over elements in chain
    for i in range(len(Ivec)):
        I_here = Ivec[i]
        Q_here = Qvec[i]
        U_here = Uvec[i]
        V_here = Vvec[i]

        # linear polarization fraction
        p_here = (np.sum(Q_here) + ((1j)*np.sum(U_here))) / np.sum(I_here)
        pvec[i] = np.abs(p_here)

        # EVPA
        EVPAvec[i] = (180.0/np.pi)*0.5*np.angle(p_here)

        # circular polarization fraction
        v_here = np.sum(V_here) / np.sum(I_here)
        vvec[i] = v_here

    ###################################################
    # make cornerplot

    samples = np.ndarray(shape=(len(pvec),3))
    samples[:,0] = np.copy(pvec)
    samples[:,1] = np.copy(EVPAvec)
    samples[:,2] = np.copy(vvec)

    labels = [r'$p$',r'$\chi$',r'$v$']

    fig = corner.corner(samples,labels=labels,levels=levels,show_titles=True,title_fmt='.4f',
                title_kwargs={"fontsize": 12},smooth=1.0,smooth1d=1.0,plot_datapoints=False,
                plot_density=False,fill_contours=True,bins=30,color='black')
    fig.axes[7].set_xlabel(r'$\chi$ (deg.)')
    fig.savefig(outname,dpi=300)
    plt.close()
