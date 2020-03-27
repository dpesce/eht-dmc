#######################################################
# imports

import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import dmc as dm
import pickle

#######################################################
# loading in data

# convert uvfits file to ehtim obsdata object
obsfile = 'synthetic_data.uvfits'
obs = eh.obsdata.load_uvfits(obsfile,polrep='circ')

# convert antenna file to ehtim array object
antfile = 'template_array.txt'
eht = eh.io.load.load_array_txt(antfile)

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
nx = 12
ny = 12

# axis ranges in each dimension
xmin = -30.0
xmax = 30.0
ymin = -30.0
ymax = 30.0

# set number of burn-in, tuning, and sampling steps
ntuning = 2000
ntrials = 10000

# perform the model-fitting (note: takes a long time!)
modelinfo = dm.models.image(obs,nx,ny,xmin,xmax,ymin,ymax,ntuning=ntuning,ntrials=ntrials,total_flux_estimate=1.0,smooth=True)

# save the model file
dm.io.save_model(modelinfo,'modelinfo.p')

#######################################################
# make some summary plots

# trace plots
dm.plotting.plot_trace(modelinfo,var_names=['f','I','sigma','gain_amps','gain_phases'])
plt.savefig('traceplots.png',dpi=300)
plt.close()

# plot mean image
imageplot_mean = dm.plotting.plot_image(modelinfo,'I','mean',title='Stokes I mean image')
imageplot_mean.savefig('StokesI_mean.png',dpi=300)
plt.close(imageplot_mean)

# plot snr image
imageplot_snr = dm.plotting.plot_image(modelinfo,'I','snr',title='Stokes I snr image')
imageplot_snr.savefig('StokesI_snr.png',dpi=300)
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

#######################################################
# output some eht-imaging compatible files

# save fits file of mean image
dm.io.save_fits(modelinfo,'mean','image_mean.fits')
