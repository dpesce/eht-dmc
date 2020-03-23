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

#######################################################
# functions
#######################################################

def gain_design_mats(obs):
    """ Construct design matrices for gain amplitudes and phases, assuming
        that there is a single complex gain associated with each timestamp
        and that all gains are independent of each other

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           
       Returns:
           gain_design_mat_1: gain design matrix for the first station
           gain_design_mat_2: gain design matrix for the second station

    """

    # get arrays of station names
    ant1 = obs.data['t1']
    ant2 = obs.data['t2']

    # get array of timestamps
    time = obs.data['time']
    timestamps = np.unique(time)

    # Determine the total number of gains that need to be solved for
    N_gains = 0
    T_gains = list()
    A_gains = list()
    for it, t in enumerate(timestamps):
        ind_here = (time == t)
        N_gains += len(np.unique(np.concatenate((ant1[ind_here],ant2[ind_here]))))
        stations_here = np.unique(np.concatenate((ant1[ind_here],ant2[ind_here])))
        for ii in range(len(stations_here)):
            A_gains.append(stations_here[ii])
            T_gains.append(t)
    T_gains = np.array(T_gains)
    A_gains = np.array(A_gains)

    # initialize design matrices
    gain_design_mat_1 = np.zeros((len(time),N_gains))
    gain_design_mat_2 = np.zeros((len(time),N_gains))

    # fill in design matrices
    for i in range(len(time)):
        ind1 = ((T_gains == time[i]) & (A_gains == ant1[i]))
        ind2 = ((T_gains == time[i]) & (A_gains == ant2[i]))

        gain_design_mat_1[i,ind1] = 1.0
        gain_design_mat_2[i,ind2] = 1.0

    return gain_design_mat_1, gain_design_mat_2

def dterm_design_mats(obs):
    """ Construct design matrices for leakage (D-term) amplitudes and phases,
        assuming that there is only a single (i.e., time-independent) complex
        D-term per hand per station over the entire observation

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           
       Returns:
           dterm_design_mat_1: D-term design matrix for the first station
           dterm_design_mat_2: D-term design matrix for the second station

    """

    # get array of stations
    ant1 = obs.data['t1']
    ant2 = obs.data['t2']
    stations = np.unique(np.concatenate((ant1,ant2)))

    # get array of timestamps
    time = obs.data['time']
    timestamps = np.unique(time)

    # determine the total number of D-terms that need to be solved for
    N_Dterms = len(stations)

    # initialize design matrices
    Dterm_design_mat_1 = np.zeros((len(time),N_Dterms))
    Dterm_design_mat_2 = np.zeros((len(time),N_Dterms))

    # fill in design matrices
    for istat,station in enumerate(stations):
        ind1 = (ant1 == station)
        ind2 = (ant2 == station)

        Dterm_design_mat_1[ind1,istat] = 1.0
        Dterm_design_mat_2[ind2,istat] = 1.0

    return dterm_design_mat_1, dterm_design_mat_2

def FRvec(obs,ehtim_convention=True):
    """ Construct vectors of field rotation corrections for each station and hand

       Args:
           obs (obsdata): eht-imaging obsdata object containing VLBI data
           ehtim_convention (bool): if true, assume the field rotation uses the
                                    eht-imaging pre-rotation convention
           
       Returns:
           FR1: field rotation corrections for the first station
           FR2: field rotation corrections for the second station

    """

    # read the elevation angles for each station
    el1 = obs.unpack(['el1'],ang_unit='rad')['el1']
    el2 = obs.unpack(['el2'],ang_unit='rad')['el2']

    # read the parallactic angles for each station
    par1 = obs.unpack(['par_ang1'],ang_unit='rad')['par_ang1']
    par2 = obs.unpack(['par_ang2'],ang_unit='rad')['par_ang2']

    # get the observation array info
    tarr = obs.tarr

    # get arrays of station names
    ant1 = obs.data['t1']
    ant2 = obs.data['t2']

    # get multiplicative prefactors for station 1
    f_el1 = np.zeros_like(el1)
    f_par1 = np.zeros_like(par1)
    f_off1 = np.zeros_like(el1)
    for ia, a1 in enumerate(ant1):
        ind1 = (tarr['site'] == a1)
        f_el1[ia] = tarr[ind1]['fr_elev']
        f_par1[ia] = tarr[ind1]['fr_par']
        f_off1[ia] = tarr[ind1]['fr_off']*eh.DEGREE

    # get multiplicative prefactors for station 2
    f_el2 = np.zeros_like(el2)
    f_par2 = np.zeros_like(par2)
    f_off2 = np.zeros_like(el2)
    for ia, a2 in enumerate(ant2):
        ind2 = (tarr['site'] == a2)
        f_el2[ia] = tarr[ind2]['fr_elev']
        f_par2[ia] = tarr[ind2]['fr_par']
        f_off2[ia] = tarr[ind2]['fr_off']*eh.DEGREE

    # combine to get field rotations for each station
    FR1 = (f_el1*el1) + (f_par1*par1) + f_off1
    FR2 = (f_el2*el2) + (f_par2*par2) + f_off2

    # negate the second station to account for conjugation
    FR2 *= -1.0

    # if pre-rotation has been applied, multiply by 2.0
    if ehtim_convention:
        FR1 *= 2.0
        FR2 *= 2.0

    return FR1, FR2



