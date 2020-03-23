#######################################################
# imports
#######################################################

from __future__ import division
from __future__ import print_function
from builtins import list
from builtins import len
from builtins import range
from builtins import enumerate

import numpy as np
import ehtim as eh
import dmc.data_utils as du
import dmc.model_utils as mu

#######################################################
# functions
#######################################################

def polim(obs,nx,ny,xmin,xmax,ymin,ymax,**kwargs):
    """ Fit a polarimetric image (i.e., Stokes I, Q, U, and V) to a VLBI observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           nx (int): number of pixels in the x-direction
           ny (int): number of pixels in the y-direction
           xmin(float): minimum x pixel value (uas)
           xmax(float): maximum x pixel value (uas)
           ymin(float): minimum y pixel value (uas)
           ymax(float): maximum x pixel value (uas)
           
       Returns:
           tracefile (trace): a pymc3 trace object containin the model fit

    """

    # some kwarg default values
    ehtim_convention = kwargs.get('ehtim_convention', True)
    ref_station = kwargs.get('ref_station','AA')

    ###################################################
    # initializing the image

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

    ###################################################
    # data bookkeeping

    # construct design matrices for gain terms
    gain_design_mat_1, gain_design_mat_2 = du.gain_design_mats(obs)

    # construct design matrices for leakage terms
    dterm_design_mat_1, dterm_design_mat_2 = du.dterm_design_mats(obs)

    # construct vectors of field rotation corrections
    FR_vec_R1, FR_vec_R2, FR_vec_L1, FR_vec_L2 = du.FRvec(obs,ehtim_convention=ehtim_convention)

    ###################################################
    # organizing prior information

    # prior info for log gain amplitudes
    loggainamp_mean, loggainamp_std = mu.gain_logamp_prior(obs)

    # prior info for gain phases
    gainphase_mu, gainphase_kappa = mu.gain_phase_prior(obs,ref_station=ref_station)

    # specify the Dirichlet weights; 1 = flat; <1 = sparse; >1 = smooth
    dirichlet_weights = 1.0*np.ones_like(x)















    return tracefile



