#######################################################
# imports

import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import dmc as dm
import pickle
import os

#######################################################
# loading in data

# convert uvfits file to ehtim obsdata object
obsfile = 'synthetic_data.uvfits'
obs = eh.obsdata.load_uvfits(obsfile,polrep='circ')

# convert antenna file to ehtim array object
antfile = 'template_array.txt'
eht = eh.io.load.load_array_txt('template_array.txt')

# associate the array information with the obs
obs.tarr = eht.tarr

# scan average
obs.add_scans()
obs = obs.avg_coherent(0.0,scan_avg=True)

# flag zero-baselines
obs = obs.flag_bl(['AA','AP'])
obs = obs.flag_bl(['AP','AA'])
obs = obs.flag_bl(['SM','JC'])
obs = obs.flag_bl(['JC','SM'])

#######################################################
# fit a polarized image model

# number of pixels in each dimension
nx = 6
ny = 6

# axis ranges in each dimension
xmin = -20.0
xmax = 20.0
ymin = -20.0
ymax = 20.0

# set number of burn-in, tuning, and sampling steps
ntuning = 2000
ntrials = 10000

# estimate the total flux, in Jy
total_flux_estimate = 0.4

# perform the model-fitting (note: takes a long time!)
modelinfo = dm.models.polimage(obs,nx,ny,xmin,xmax,ymin,ymax,ntuning=ntuning,ntrials=ntrials,total_flux_estimate=total_flux_estimate,ref_station='AA')

# save the model file
dm.io.save_model(modelinfo,'modelinfo.p')

#######################################################
# make a bunch of summary plots

# trace plots
dm.plotting.plot_trace(modelinfo,var_names=['f','I','Q','U','V','right_gain_amps','left_gain_amps','right_gain_phases','left_gain_phases','right_Dterm_reals','left_Dterm_reals','right_Dterm_imags','left_Dterm_imags'])
plt.savefig('traceplots.png',dpi=300)
plt.close()

# Stokes plots
for stokes in ['I','Q','U','V']:

    # plot mean image
    imageplot_mean = dm.plotting.plot_image(modelinfo,stokes,'mean',title='Stokes '+stokes+' mean image')
    imageplot_mean.savefig('Stokes'+stokes+'_mean.png',dpi=300)
    plt.close(imageplot_mean)

    # plot snr image
    imageplot_snr = dm.plotting.plot_image(modelinfo,stokes,'snr',title='Stokes '+stokes+' SNR image')
    imageplot_snr.savefig('Stokes'+stokes+'_snr.png',dpi=300)
    plt.close(imageplot_snr)

# gain plots
gainplot_amps = dm.plotting.plot_gains(modelinfo,'amp')
gainplot_amps.savefig('gainplot_amps.png',dpi=300)
plt.close(gainplot_amps)

gainplot_phases = dm.plotting.plot_gains(modelinfo,'phase')
gainplot_phases.savefig('gainplot_phases.png',dpi=300)
plt.close(gainplot_phases)

dm.plotting.gain_cornerplots(modelinfo,'amp')
dm.plotting.gain_cornerplots(modelinfo,'phase')

# dterm plots
if not os.path.exists('./dterms'):
    os.mkdir('./dterms')

for station in modelinfo['stations']:
    dtermplot = dm.plotting.plot_dterms(modelinfo,station)
    dtermplot.savefig('./dterms/dterms_'+station+'.png',dpi=300)
    plt.close(dtermplot)
