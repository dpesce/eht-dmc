#######################################################
# imports

import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import dmc as dm

#######################################################
# loading in data

# convert uvfits file to ehtim obsdata object
obsfile = 'hops_lo_3601_M87+zbl-dtcal_selfcal.uvfits'
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
nx = 20
ny = 20

# axis ranges in each dimension
xmin = -30.0
xmax = 30.0
ymin = -30.0
ymax = 30.0

output = dm.models.polim(obs,nx,ny,xmin,xmax,ymin,ymax)





