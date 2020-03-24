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
obsfile = 'hops_lo_3600_M87+zbl-dtcal_selfcal.uvfits'
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
xmin = -30.0
xmax = 30.0
ymin = -30.0
ymax = 30.0

# set number of burn-in, tuning, and sampling steps
ntuning = 2000
ntrials = 10000

# perform the model-fitting (note: takes a long time!)
modelinfo = dm.models.polim(obs,nx,ny,xmin,xmax,ymin,ymax,ntuning=ntuning,ntrials=ntrials)

# saving the trace file
pickle.dump(modelinfo,open('modelinfo.p','wb'),protocol=pickle.HIGHEST_PROTOCOL)

#######################################################
# create some summary plots

# modelinfo = pickle.load(open('modelinfo.p','rb'))

# save a set of trace plots
dm.plotting.plot_trace(modelinfo,var_names=['f','I','Q','U','V','right_gain_amps','left_gain_amps','right_gain_phases','left_gain_phases','right_Dterm_reals','left_Dterm_reals','right_Dterm_imags','left_Dterm_imags'])
plt.savefig('traceplots.png',dpi=300)
plt.close()

# save Stokes I plots
dm.plotting.plot_image(modelinfo,'I','mean')
plt.savefig('StokesI_mean.png',dpi=300)
plt.close()










