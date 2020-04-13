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
obs_orig = eh.obsdata.load_uvfits(obsfile)

# convert antenna file to ehtim array object
antfile = 'template_array.txt'
eht = eh.io.load.load_array_txt(antfile)

# associate the array information with the obs
obs_orig.tarr = eht.tarr

# scan average
obs_orig.add_scans()
obs_orig = obs_orig.avg_coherent(0.0,scan_avg=True)

# flag zero-baselines
obs_orig = obs_orig.flag_bl(['AA','AP'])
obs_orig = obs_orig.flag_bl(['AP','AA'])
obs_orig = obs_orig.flag_bl(['SM','JC'])
obs_orig = obs_orig.flag_bl(['JC','SM'])

# make an empty image
npix = 1024
fov = 200.0*eh.RADPERUAS
im = eh.image.make_empty(npix=npix,fov=fov,ra=obs_orig.ra,dec=obs_orig.dec,rf=obs_orig.rf,source=obs_orig.source,mjd=obs_orig.mjd)

# add a Gaussian to the image
sigma_x = 10.0*eh.RADPERUAS
sigma_y = 15.0*eh.RADPERUAS
FWHM_fac = 2.0*np.sqrt(2.0*np.log(2.0))
theta = 30.0*eh.DEGREE
im = im.add_gauss(1.0,[sigma_x*FWHM_fac,sigma_y*FWHM_fac,theta,0.0,0.0])

# observe the image with the original observation
obs = im.observe_same(obs_orig,ttype='direct',fft_pad_factor=2,add_th_noise=True, opacitycal=True, ampcal=False, phasecal=False, dcal=True, frcal=True, rlgaincal=True, stabilize_scan_phase=False, stabilize_scan_amp=True)

#######################################################
# fit a point source model

# set number of tuning and sampling steps
ntuning = 2000
ntrials = 10000

# perform the model-fitting (note: takes a long time!)
modelinfo = dm.models.gauss(obs,ntuning=ntuning,ntrials=ntrials,total_flux_estimate=1.0)

# save the model file
dm.io.save_model(modelinfo,'modelinfo.p')

#######################################################
# make some summary plots

# trace plots
dm.plotting.plot_trace(modelinfo,var_names=['sigma_x','sigma_y','psi','gain_amps','gain_phases'])
plt.savefig('traceplots.png',dpi=300)
plt.close()

# gain plots
gainplot_amps = dm.plotting.plot_gains(modelinfo,'amp')
gainplot_amps.savefig('gainplot_amps.png',dpi=300)
plt.close(gainplot_amps)

gainplot_phases = dm.plotting.plot_gains(modelinfo,'phase')
gainplot_phases.savefig('gainplot_phases.png',dpi=300)
plt.close(gainplot_phases)

dm.plotting.gain_cornerplots(modelinfo,'amp')
dm.plotting.gain_cornerplots(modelinfo,'phase')
