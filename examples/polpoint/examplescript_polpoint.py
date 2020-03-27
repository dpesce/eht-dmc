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
eht = eh.io.load.load_array_txt(antfile)

# associate the array information with the obs
obs.tarr = eht.tarr

# scan average
obs.add_scans()
obs = obs.avg_coherent(0.0,scan_avg=True)

# flag nonzero baselines
# retain only AA-AP
obs = obs.flag_sites(['LM','AZ','PV','SP','JC','SM'])

#######################################################
# fit a point source model

# set number of tuning and sampling steps
ntuning = 2000
ntrials = 10000

# perform the model-fitting (note: takes a long time!)
modelinfo = dm.models.polpoint(obs,ntuning=ntuning,ntrials=ntrials,total_flux_estimate=1.0)

# save the model file
dm.io.save_model(modelinfo,'modelinfo.p')

#######################################################
# make some summary plots

# trace plots
dm.plotting.plot_trace(modelinfo,var_names=['f','Q','U','V','right_gain_amps','left_gain_amps','right_gain_phases','left_gain_phases','right_Dterm_reals','left_Dterm_reals','right_Dterm_imags','left_Dterm_imags'])
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

# dterm plots
if not os.path.exists('./dterms'):
    os.mkdir('./dterms')

for station in modelinfo['stations']:
    dtermplot = dm.plotting.plot_dterms(modelinfo,station)
    dtermplot.savefig('./dterms/dterms_'+station+'.png',dpi=300)
    plt.close(dtermplot)