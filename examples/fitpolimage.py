import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import ehtim as eh
import ehtplot
import pickle
import corner
import os

########################################################################
# inputs

# set random number seed for reproducibility
np.random.seed(12345)

ntrials = 2000
ntuning = 10000
burnin  = 2000
nx = 20

xmin = -30
xmax = 30
ymin = -30
ymax = 30

obsfile = 'hops_lo_3601_M87+zbl-dtcal_selfcal.uvfits'

# Specify the SEFD error budget
# (Note: the ordering of this dictionary determines the
# priority enumeration for reference station selection)
SEFD_error_budget = {'AA':0.10,
                     'AP':0.11,
                     'AZ':0.07,
                     'LM':0.22,
                     'PV':0.10,
                     'SM':0.15,
                     'JC':0.14,
                     'SP':0.07}

########################################################################
# reading in the data and performing some initial bookkeeping

obs = eh.obsdata.load_uvfits(obsfile,polrep='circ')

# Load the array file (to transfer mount type specifications)
eht = eh.io.load.load_array_txt('template_array.txt')
obs.tarr = eht.tarr

"""
# Load the array file (to transfer mount type specifications)
eht = eh.array.load_txt('template_array.txt')

# Fill in the feed parameters for each site
t_obs = list(obs.tarr['site'])
t_eht = list(eht.tarr['site'])
t_conv = {'AA':'ALMA','AP':'APEX','SM':'SMA','JC':'JCMT','AZ':'SMT','LM':'LMT','PV':'PV','SP':'SPT'}
for t in t_conv.keys():
    if t in obs.tarr['site']:
        for key in ['fr_par','fr_elev','fr_off']:
            obs.tarr[key][t_obs.index(t)] = eht.tarr[key][t_eht.index(t_conv[t])]

# Rotate the R-L gains at ALMA (note: this is needed for real EHT data (ER5) but not simulated data!)
print('Correcting the absolute EVPA calibration...')
datadict = {t['site']:np.array([(0.0, 0.0 + 1j*1.0, 1.0 + 1j*0.0)], dtype=eh.DTCAL) for t in obs.tarr}
caltab = eh.caltable.Caltable(obs.ra,obs.dec,obs.rf,obs.bw,datadict,obs.tarr,obs.source,obs.mjd)
obs = caltab.applycal(obs, interp='nearest',extrapolate=True)
"""

# scan average
obs.add_scans()
obs = obs.avg_coherent(0.0,scan_avg=True)

# flag zero-baselines
obs = obs.flag_bl(['AA','AP'])
obs = obs.flag_bl(['AP','AA'])
obs = obs.flag_bl(['SM','JC'])
obs = obs.flag_bl(['JC','SM'])

########################################################################
# initializing the image

ny = nx
npix = nx*ny

x_1d = np.linspace(xmin,xmax,nx)
y_1d = np.linspace(ymin,ymax,ny)
x2d, y2d = np.meshgrid(x_1d,y_1d,indexing='ij')
x = eh.RADPERUAS*x2d.ravel()
y = eh.RADPERUAS*y2d.ravel()

########################################################################
# more bookkeeping

u = np.copy(obs.data['u'])
v = np.copy(obs.data['v'])
rho = np.sqrt((u**2.0) + (v**2.0))

RR_real = np.real(np.copy(obs.data['rrvis']))
RR_imag = np.imag(np.copy(obs.data['rrvis']))
RR_real_err = np.copy(obs.data['rrsigma'])
RR_imag_err = np.copy(obs.data['rrsigma'])

LL_real = np.real(np.copy(obs.data['llvis']))
LL_imag = np.imag(np.copy(obs.data['llvis']))
LL_real_err = np.copy(obs.data['llsigma'])
LL_imag_err = np.copy(obs.data['llsigma'])

RL_real = np.real(np.copy(obs.data['rlvis']))
RL_imag = np.imag(np.copy(obs.data['rlvis']))
RL_real_err = np.copy(obs.data['rlsigma'])
RL_imag_err = np.copy(obs.data['rlsigma'])

LR_real = np.real(np.copy(obs.data['lrvis']))
LR_imag = np.imag(np.copy(obs.data['lrvis']))
LR_real_err = np.copy(obs.data['lrsigma'])
LR_imag_err = np.copy(obs.data['lrsigma'])

ant1 = np.copy(obs.data['t1'])
ant2 = np.copy(obs.data['t2'])

time = np.copy(obs.data['time'])
timestamps = np.unique(time)

# mask out any blank data by giving it enormous uncertainties
mask = ~np.isfinite(obs.data['rrvis'])
RR_real[mask] = 0.0
RR_imag[mask] = 0.0
RR_real_err[mask] = 1000.0
RR_imag_err[mask] = 1000.0
mask_RR = np.where(np.isfinite(obs.data['rrvis']))

mask = ~np.isfinite(obs.data['llvis'])
LL_real[mask] = 0.0
LL_imag[mask] = 0.0
LL_real_err[mask] = 1000.0
LL_imag_err[mask] = 1000.0
mask_LL = np.where(np.isfinite(obs.data['llvis']))

mask = ~np.isfinite(obs.data['rlvis'])
RL_real[mask] = 0.0
RL_imag[mask] = 0.0
RL_real_err[mask] = 1000.0
RL_imag_err[mask] = 1000.0
mask_RL = np.where(np.isfinite(obs.data['rlvis']))

mask = ~np.isfinite(obs.data['lrvis'])
LR_real[mask] = 0.0
LR_imag[mask] = 0.0
LR_real_err[mask] = 1000.0
LR_imag_err[mask] = 1000.0
mask_LR = np.where(np.isfinite(obs.data['lrvis']))

# Determine the total number of gains that need to be solved for
# T_gains is an array of timestamps for each gain parameter
# A_gains is an array of antennas for each gain parameter
N_gains = 0
T_gains = []
A_gains = []
for it, t in enumerate(timestamps):
    ind_here = (time == t)
    N_gains += len(np.unique(np.concatenate((ant1[ind_here],ant2[ind_here]))))
    stations_here = np.unique(np.concatenate((ant1[ind_here],ant2[ind_here])))
    for ii in range(len(stations_here)):
        A_gains.append(stations_here[ii])
        T_gains.append(t)
T_gains = np.array(T_gains)
A_gains = np.array(A_gains)

outfile_array = np.zeros(N_gains, dtype=[('var1', 'U12'), ('var2', float)])
outfile_array['var1'] = A_gains
outfile_array['var2'] = T_gains
np.savetxt('gain_translator.txt', outfile_array, fmt="%-12s %12.6f",delimiter=' ')

# construct design matrices for gain terms
gainamp_design_mat_R1 = np.zeros((len(u),N_gains))
gainamp_design_mat_R2 = np.zeros((len(u),N_gains))
gainamp_design_mat_L1 = np.zeros((len(u),N_gains))
gainamp_design_mat_L2 = np.zeros((len(u),N_gains))

gainphase_design_mat_R1 = np.zeros((len(u),N_gains))
gainphase_design_mat_R2 = np.zeros((len(u),N_gains))
gainphase_design_mat_L1 = np.zeros((len(u),N_gains))
gainphase_design_mat_L2 = np.zeros((len(u),N_gains))

for i in range(len(u)):
    ind1 = ((T_gains == time[i]) & (A_gains == ant1[i]))
    ind2 = ((T_gains == time[i]) & (A_gains == ant2[i]))

    gainamp_design_mat_R1[i,ind1] = 1.0
    gainamp_design_mat_R2[i,ind2] = 1.0
    gainamp_design_mat_L1[i,ind1] = 1.0
    gainamp_design_mat_L2[i,ind2] = 1.0

    gainphase_design_mat_R1[i,ind1] = 1.0
    gainphase_design_mat_R2[i,ind2] = 1.0
    gainphase_design_mat_L1[i,ind1] = 1.0
    gainphase_design_mat_L2[i,ind2] = 1.0

# construct design matrices for D terms
stations = np.unique(np.concatenate((ant1,ant2)))
N_Dterms = len(stations)

Dterm_design_mat_R1 = np.zeros((len(u),N_Dterms))
Dterm_design_mat_R2 = np.zeros((len(u),N_Dterms))
Dterm_design_mat_L1 = np.zeros((len(u),N_Dterms))
Dterm_design_mat_L2 = np.zeros((len(u),N_Dterms))

for istat,station in enumerate(stations):
    ind1 = (ant1 == station)
    ind2 = (ant2 == station)

    Dterm_design_mat_R1[ind1,istat] = 1.0
    Dterm_design_mat_R2[ind2,istat] = 1.0

    Dterm_design_mat_L1[ind1,istat] = 1.0
    Dterm_design_mat_L2[ind2,istat] = 1.0

# construct vectors of field rotation corrections
el1 = obs.unpack(['el1'],ang_unit='rad')['el1']
el2 = obs.unpack(['el2'],ang_unit='rad')['el2']

par1 = obs.unpack(['par_ang1'],ang_unit='rad')['par_ang1']
par2 = obs.unpack(['par_ang2'],ang_unit='rad')['par_ang2']

tarr = obs.tarr

f_el1 = np.zeros_like(el1)
f_par1 = np.zeros_like(par1)
f_off1 = np.zeros_like(el1)
for ia, a1 in enumerate(ant1):
    ind1 = (tarr['site'] == a1)
    f_el1[ia] = tarr[ind1]['fr_elev']
    f_par1[ia] = tarr[ind1]['fr_par']
    f_off1[ia] = tarr[ind1]['fr_off']*eh.DEGREE

f_el2 = np.zeros_like(el2)
f_par2 = np.zeros_like(par2)
f_off2 = np.zeros_like(el2)
for ia, a2 in enumerate(ant2):
    ind2 = (tarr['site'] == a2)
    f_el2[ia] = tarr[ind2]['fr_elev']
    f_par2[ia] = tarr[ind2]['fr_par']
    f_off2[ia] = tarr[ind2]['fr_off']*eh.DEGREE

FR1 = (f_el1*el1) + (f_par1*par1) + f_off1
FR2 = (f_el2*el2) + (f_par2*par2) + f_off2

FR_vec_R1 = 2.0*np.copy(FR1)
FR_vec_R2 = 2.0*np.copy(FR2)
FR_vec_L1 = -2.0*np.copy(FR1)
FR_vec_L2 = -2.0*np.copy(FR2)

# make vector of gain prior means and standard deviations
gainamp_mean = np.ones(N_gains)
gainamp_std = np.ones(N_gains)
for key in SEFD_error_budget.keys():
    index = (A_gains == key)
    gainamp_mean[index] = 1.0
    gainamp_std[index] = SEFD_error_budget[key]
loggainamp_mean = np.log(gainamp_mean)
loggainamp_std = gainamp_std/gainamp_mean

# select reference station with priority given by SEFD budget ordering
gainphase_mu = np.zeros(N_gains)
gainphase_kappa = 0.0001*np.ones(N_gains)
for it, t in enumerate(timestamps):
    index = (T_gains == t)
    ants_here = A_gains[index]
    for key in ['AA']:
        if key in ants_here:
            ind = ((T_gains == t) & (A_gains == key))
            gainphase_kappa[ind] = 10000.0
            break

########################################################################
# constructing Fourier transform matrix

A = np.zeros((len(u),len(x)),dtype='complex')
for i in range(len(u)):
    A[i,:] = np.exp(-2.0*np.pi*(1j)*((u[i]*x) + (v[i]*y)))

# Taking a complex conjugate to account for ehtim internal FT convention
A_real = np.real(A)
A_imag = -np.imag(A)

########################################################################
# setting up the model

# specify the Dirichlet weights; 1=flat, <1=sparse
dirichlet_weights = 1.0*np.ones_like(x)

model = pm.Model()

with model:

    ########################################################################
    # set the priors for the image parameters
    
    # set the total flux prior to be normal around the correct value
    # BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
    # F = BoundedNormal('F',mu=1.2,sd=0.06)
    F = 0.6
    
    # Impose a flat Dirichlet prior on the pixel Stokes I intensities,
    # with summation constraint equal to the total flux
    pix = pm.Dirichlet('pix',dirichlet_weights)
    I = pm.Deterministic('I',pix*F)
    
    # sample the polarization fraction uniformly on [0,1]
    p = pm.Uniform('p',lower=0.0,upper=1.0,shape=npix)

    # sample alpha from a periodic uniform distribution
    alpha = pm.VonMises('alpha',mu=0.0,kappa=0.0001,shape=npix)

    # sample cos(beta) uniformly on [-1,1]
    cosbeta = pm.Uniform('cosbeta',lower=-1.0,upper=1.0,shape=npix)
    sinbeta = pm.math.sqrt(1.0 - (cosbeta**2.0))

    # set the prior on the systematic error term to be uniform on [0,1]
    f = pm.Uniform('f',lower=0.0,upper=1.0)

    ########################################################################
    # set the priors for the gain parameters

    # set the gain amplitude priors to be log-normal around the specified inputs
    # BoundedNormal = pm.Bound(pm.Normal, upper=0.0)

    logg_R = pm.Normal('right_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
    g_R = pm.Deterministic('right_gain_amps',pm.math.exp(logg_R))

    logg_L = pm.Normal('left_logg',mu=loggainamp_mean,sd=loggainamp_std,shape=N_gains)
    g_L = pm.Deterministic('left_gain_amps',pm.math.exp(logg_L))
    
    # set the gain phase priors to be periodic uniform on (-pi,pi)
    theta_R = pm.VonMises('right_gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)
    theta_L = pm.VonMises('left_gain_phases',mu=gainphase_mu,kappa=gainphase_kappa,shape=N_gains)

    ########################################################################
    # set the priors for the leakage parameters
    
    # set the D term amplitude priors to be uniform on [0,1]
    Damp_R = pm.Uniform('right_Dterm_amps',lower=0.0,upper=1.0,shape=N_Dterms,testval=0.01)
    logDamp_R = pm.math.log(Damp_R)

    Damp_L = pm.Uniform('left_Dterm_amps',lower=0.0,upper=1.0,shape=N_Dterms,testval=0.01)
    logDamp_L = pm.math.log(Damp_L)

    # set the D term phase priors to be periodic uniform on (-pi,pi)
    delta_R = pm.VonMises('right_Dterm_phases',mu=0.0,kappa=0.0001,shape=N_Dterms)
    delta_L = pm.VonMises('left_Dterm_phases',mu=0.0,kappa=0.0001,shape=N_Dterms)

    # save the real and imaginary parts for output diagnostics
    D_R_real = pm.Deterministic('right_Dterm_reals',Damp_R*pm.math.cos(delta_R))
    D_R_imag = pm.Deterministic('right_Dterm_imags',Damp_R*pm.math.sin(delta_R))
    D_L_real = pm.Deterministic('left_Dterm_reals',Damp_L*pm.math.cos(delta_L))
    D_L_imag = pm.Deterministic('left_Dterm_imags',Damp_L*pm.math.sin(delta_L))

    ########################################################################
    # compute the polarized Stokes parameters

    Q = pm.Deterministic('Q',I*p*pm.math.cos(alpha)*sinbeta)
    U = pm.Deterministic('U',I*p*pm.math.sin(alpha)*sinbeta)
    V = pm.Deterministic('V',I*p*cosbeta)

    ########################################################################
    # perform the required Fourier transforms
    
    Ireal = pm.math.dot(A_real,I)
    Iimag = pm.math.dot(A_imag,I)
    
    Qreal = pm.math.dot(A_real,Q)
    Qimag = pm.math.dot(A_imag,Q)
    
    Ureal = pm.math.dot(A_real,U)
    Uimag = pm.math.dot(A_imag,U)
    
    Vreal = pm.math.dot(A_real,V)
    Vimag = pm.math.dot(A_imag,V)

    ########################################################################
    # construct the pre-corrupted circular basis model visibilities

    RR_real_pregain = Ireal + Vreal
    RR_imag_pregain = Iimag + Vimag

    LL_real_pregain = Ireal - Vreal
    LL_imag_pregain = Iimag - Vimag

    RL_real_pregain = Qreal - Uimag
    RL_imag_pregain = Qimag + Ureal

    LR_real_pregain = Qreal + Uimag
    LR_imag_pregain = Qimag - Ureal

    ########################################################################
    # compute the corruption terms
    
    gainamp_R1 = pm.math.exp(pm.math.dot(gainamp_design_mat_R1,logg_R))
    gainamp_R2 = pm.math.exp(pm.math.dot(gainamp_design_mat_R2,logg_R))
    gainamp_L1 = pm.math.exp(pm.math.dot(gainamp_design_mat_L1,logg_L))
    gainamp_L2 = pm.math.exp(pm.math.dot(gainamp_design_mat_L2,logg_L))
    
    gainphase_R1 = pm.math.dot(gainphase_design_mat_R1,theta_R)
    gainphase_R2 = pm.math.dot(gainphase_design_mat_R2,theta_R)
    gainphase_L1 = pm.math.dot(gainphase_design_mat_L1,theta_L)
    gainphase_L2 = pm.math.dot(gainphase_design_mat_L2,theta_L)
    
    Damp_R1 = pm.math.exp(pm.math.dot(Dterm_design_mat_R1,logDamp_R))
    Damp_R2 = pm.math.exp(pm.math.dot(Dterm_design_mat_R2,logDamp_R))
    Damp_L1 = pm.math.exp(pm.math.dot(Dterm_design_mat_L1,logDamp_L))
    Damp_L2 = pm.math.exp(pm.math.dot(Dterm_design_mat_L2,logDamp_L))
    
    Dphase_R1_preFR = pm.math.dot(Dterm_design_mat_R1,delta_R)
    Dphase_R2_preFR = pm.math.dot(Dterm_design_mat_R2,delta_R)
    Dphase_L1_preFR = pm.math.dot(Dterm_design_mat_L1,delta_L)
    Dphase_L2_preFR = pm.math.dot(Dterm_design_mat_L2,delta_L)
    
    ########################################################################
    # correct the leakage phases for field rotation
    
    Dphase_R1 = Dphase_R1_preFR + FR_vec_R1
    Dphase_R2 = Dphase_R2_preFR + FR_vec_R2
    Dphase_L1 = Dphase_L1_preFR + FR_vec_L1
    Dphase_L2 = Dphase_L2_preFR + FR_vec_L2
    
    ########################################################################
    # apply corruptions to the model visibilities

    RR_real_model = gainamp_R1*gainamp_R2*((pm.math.cos(gainphase_R1 - gainphase_R2)*RR_real_pregain)
                    - (pm.math.sin(gainphase_R1 - gainphase_R2)*RR_imag_pregain)
                    + (Damp_R1*Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_real_pregain)
                    - (Damp_R1*Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_imag_pregain)
                    + (Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_real_pregain)
                    - (Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_imag_pregain)
                    + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_real_pregain)
                    - (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_imag_pregain))

    RR_imag_model = gainamp_R1*gainamp_R2*((pm.math.sin(gainphase_R1 - gainphase_R2)*RR_real_pregain)
                    + (pm.math.cos(gainphase_R1 - gainphase_R2)*RR_imag_pregain)
                    + (Damp_R1*Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_real_pregain)
                    + (Damp_R1*Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1 - Dphase_R2)*LL_imag_pregain)
                    + (Damp_R2*pm.math.sin(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_real_pregain)
                    + (Damp_R2*pm.math.cos(gainphase_R1 - gainphase_R2 - Dphase_R2)*RL_imag_pregain)
                    + (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_real_pregain)
                    + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_R2 + Dphase_R1)*LR_imag_pregain))

    LL_real_model = gainamp_L1*gainamp_L2*((Damp_L1*Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_real_pregain)
                    - (Damp_L1*Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_imag_pregain)
                    + (pm.math.cos(gainphase_L1 - gainphase_L2)*LL_real_pregain)
                    - (pm.math.sin(gainphase_L1 - gainphase_L2)*LL_imag_pregain)
                    + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_real_pregain)
                    - (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_imag_pregain)
                    + (Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_real_pregain)
                    - (Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_imag_pregain))

    LL_imag_model = gainamp_L1*gainamp_L2*((Damp_L1*Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_real_pregain)
                    + (Damp_L1*Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1 - Dphase_L2)*RR_imag_pregain)
                    + (pm.math.sin(gainphase_L1 - gainphase_L2)*LL_real_pregain)
                    + (pm.math.cos(gainphase_L1 - gainphase_L2)*LL_imag_pregain)
                    + (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_real_pregain)
                    + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_L2 + Dphase_L1)*RL_imag_pregain)
                    + (Damp_L2*pm.math.sin(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_real_pregain)
                    + (Damp_L2*pm.math.cos(gainphase_L1 - gainphase_L2 - Dphase_L2)*LR_imag_pregain))

    RL_real_model = gainamp_R1*gainamp_L2*((Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_real_pregain)
                    - (Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_imag_pregain)
                    + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_real_pregain)
                    - (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_imag_pregain)
                    + (pm.math.cos(gainphase_R1 - gainphase_L2)*RL_real_pregain)
                    - (pm.math.sin(gainphase_R1 - gainphase_L2)*RL_imag_pregain)
                    + (Damp_R1*Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_real_pregain)
                    - (Damp_R1*Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_imag_pregain))

    RL_imag_model = gainamp_R1*gainamp_L2*((Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_real_pregain)
                    + (Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 - Dphase_L2)*RR_imag_pregain)
                    + (Damp_R1*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_real_pregain)
                    + (Damp_R1*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1)*LL_imag_pregain)
                    + (pm.math.sin(gainphase_R1 - gainphase_L2)*RL_real_pregain)
                    + (pm.math.cos(gainphase_R1 - gainphase_L2)*RL_imag_pregain)
                    + (Damp_R1*Damp_L2*pm.math.sin(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_real_pregain)
                    + (Damp_R1*Damp_L2*pm.math.cos(gainphase_R1 - gainphase_L2 + Dphase_R1 - Dphase_L2)*LR_imag_pregain))

    LR_real_model = gainamp_L1*gainamp_R2*((Damp_L1*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_real_pregain)
                    - (Damp_L1*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_imag_pregain)
                    + (Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_real_pregain)
                    - (Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_imag_pregain)
                    + (Damp_L1*Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_real_pregain)
                    - (Damp_L1*Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_imag_pregain)
                    + (pm.math.cos(gainphase_L1 - gainphase_R2)*LR_real_pregain)
                    - (pm.math.sin(gainphase_L1 - gainphase_R2)*LR_imag_pregain))

    LR_imag_model = gainamp_L1*gainamp_R2*((Damp_L1*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_real_pregain)
                    + (Damp_L1*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1)*RR_imag_pregain)
                    + (Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_real_pregain)
                    + (Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 - Dphase_R2)*LL_imag_pregain)
                    + (Damp_L1*Damp_R2*pm.math.sin(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_real_pregain)
                    + (Damp_L1*Damp_R2*pm.math.cos(gainphase_L1 - gainphase_R2 + Dphase_L1 - Dphase_R2)*RL_imag_pregain)
                    + (pm.math.sin(gainphase_L1 - gainphase_R2)*LR_real_pregain)
                    + (pm.math.cos(gainphase_L1 - gainphase_R2)*LR_imag_pregain))

    ########################################################################
    # add in the systematic noise component

    RR_real_err_model = pm.math.sqrt((RR_real_err**2.0) + (f*F)**2.0)
    RR_imag_err_model = pm.math.sqrt((RR_imag_err**2.0) + (f*F)**2.0)

    LL_real_err_model = pm.math.sqrt((LL_real_err**2.0) + (f*F)**2.0)
    LL_imag_err_model = pm.math.sqrt((LL_imag_err**2.0) + (f*F)**2.0)

    RL_real_err_model = pm.math.sqrt((RL_real_err**2.0) + (f*F)**2.0)
    RL_imag_err_model = pm.math.sqrt((RL_imag_err**2.0) + (f*F)**2.0)

    LR_real_err_model = pm.math.sqrt((LR_real_err**2.0) + (f*F)**2.0)
    LR_imag_err_model = pm.math.sqrt((LR_imag_err**2.0) + (f*F)**2.0)

    ########################################################################
    # define the likelihood

    L_real_RR = pm.Normal('L_real_RR',mu=RR_real_model[mask_RR],sd=RR_real_err_model[mask_RR],observed=RR_real[mask_RR])
    L_imag_RR = pm.Normal('L_imag_RR',mu=RR_imag_model[mask_RR],sd=RR_imag_err_model[mask_RR],observed=RR_imag[mask_RR])

    L_real_LL = pm.Normal('L_real_LL',mu=LL_real_model[mask_LL],sd=LL_real_err_model[mask_LL],observed=LL_real[mask_LL])
    L_imag_LL = pm.Normal('L_imag_LL',mu=LL_imag_model[mask_LL],sd=LL_imag_err_model[mask_LL],observed=LL_imag[mask_LL])

    L_real_RL = pm.Normal('L_real_RL',mu=RL_real_model[mask_RL],sd=RL_real_err_model[mask_RL],observed=RL_real[mask_RL])
    L_imag_RL = pm.Normal('L_imag_RL',mu=RL_imag_model[mask_RL],sd=RL_imag_err_model[mask_RL],observed=RL_imag[mask_RL])

    L_real_LR = pm.Normal('L_real_LR',mu=LR_real_model[mask_LR],sd=LR_real_err_model[mask_LR],observed=LR_real[mask_LR])
    L_imag_LR = pm.Normal('L_imag_LR',mu=LR_imag_model[mask_LR],sd=LR_imag_err_model[mask_LR],observed=LR_imag[mask_LR])
    
    ssr_RR = pm.Deterministic('ssr_RR',pm.math.sum((((RR_real_model[mask_RR]-RR_real[mask_RR])/RR_real_err_model[mask_RR])**2.0) + (((RR_imag_model[mask_RR]-RR_imag[mask_RR])/RR_imag_err_model[mask_RR])**2.0)))
    ssr_LL = pm.Deterministic('ssr_LL',pm.math.sum((((LL_real_model[mask_LL]-LL_real[mask_LL])/LL_real_err_model[mask_LL])**2.0) + (((LL_imag_model[mask_LL]-LL_imag[mask_LL])/LL_imag_err_model[mask_LL])**2.0)))
    ssr_RL = pm.Deterministic('ssr_RL',pm.math.sum((((RL_real_model[mask_RL]-RL_real[mask_RL])/RL_real_err_model[mask_RL])**2.0) + (((RL_imag_model[mask_RL]-RL_imag[mask_RL])/RL_imag_err_model[mask_RL])**2.0)))
    ssr_LR = pm.Deterministic('ssr_LR',pm.math.sum((((LR_real_model[mask_LR]-LR_real[mask_LR])/LR_real_err_model[mask_LR])**2.0) + (((LR_imag_model[mask_LR]-LR_imag[mask_LR])/LR_imag_err_model[mask_LR])**2.0)))
    
########################################################################
# define a tuning procedure that adapts the mass matrix

def get_step_for_trace(trace=None, model=None, regularize=True, regular_window=5, regular_variance=1e-3, **kwargs):
    
    model = pm.modelcontext(model)
    
    # If not given, use the trivial metric
    if trace is None:
        potential = pm.step_methods.hmc.quadpotential.QuadPotentialFull(np.eye(model.ndim))
        return pm.NUTS(potential=potential, **kwargs)
    
    # Loop over samples and convert to the relevant parameter space
    div_mask = np.invert(np.copy(trace.diverging))
    samples = np.empty((div_mask.sum() * trace.nchains, model.ndim))
    i = 0
    imask = 0
    for chain in trace._straces.values():
        for p in chain:
            if div_mask[imask]:
                samples[i] = model.bijection.map(p)
                i += 1
            imask += 1
    
    # Compute the sample covariance
    cov = np.cov(samples, rowvar=0)
    
    # Stan uses a regularized estimator for the covariance matrix to
    # be less sensitive to numerical issues for large parameter spaces.
    if regularize:
        N = len(samples)
        cov = cov * N / (N + regular_window)
        cov[np.diag_indices_from(cov)] += regular_variance * regular_window / (N + regular_window)
    
    # Use the sample covariance as the inverse metric
    potential = pm.step_methods.hmc.quadpotential.QuadPotentialFull(cov)
    return pm.NUTS(potential=potential, **kwargs)

########################################################################
# sample

n_start = 25
n_burn = 500
n_tune = 5000
n_window = n_start * 2 ** np.arange(np.floor(np.log2((n_tune - n_burn) / n_start)))

with model:
    start = None
    burnin_trace = None

    for istep, steps in enumerate(n_window):
        step = get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=10,early_max_treedepth=10)
        burnin_trace = pm.sample(start=start, tune=steps, chains=1, step=step,compute_convergence_checks=False, discard_tuned_samples=False)
        
        pickle.dump(burnin_trace,open('tracefile_tuning'+str(istep).zfill(2)+'.p',"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        pm.plots.traceplot(burnin_trace,varnames=['f','I','Q','U','V','right_gain_amps','left_gain_amps','right_gain_phases','left_gain_phases','right_Dterm_reals','left_Dterm_reals','right_Dterm_imags','left_Dterm_imags'])
        plt.savefig('traceplots_tuning'+str(istep).zfill(2)+'.png',dpi=300)
        plt.close()

        start = [t[-1] for t in burnin_trace._straces.values()]

    step = get_step_for_trace(burnin_trace,adapt_step_size=True,max_treedepth=10,early_max_treedepth=10)
    trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=start, chains=1, discard_tuned_samples=False)

# with model:
#     start = [t[-1] for t in trace._straces.values()]
#     step = get_step_for_trace(trace,adapt_step_size=True,max_treedepth=10,early_max_treedepth=10)
#     trace = pm.sample(draws=ntrials, tune=ntuning, step=step, start=start, chains=1, discard_tuned_samples=False)

###############################################################################
# saving the trace file as binary fits

pickle.dump(trace,open('tracefile.p',"wb"),protocol=pickle.HIGHEST_PROTOCOL)
# trace = pickle.load(open("tracefile.p","rb" ))

###############################################################################
#-----------Saving a bunch of summary and diagnostic plots--------------------#
###############################################################################

########################################################
# trace plots

pm.plots.traceplot(trace,varnames=['f','I','Q','U','V','right_gain_amps','left_gain_amps','right_gain_phases','left_gain_phases','right_Dterm_reals','left_Dterm_reals','right_Dterm_imags','left_Dterm_imags'])
plt.savefig('traceplots_with_burnin.png',dpi=300)
plt.close()

########################################################
# image plots

# make edge arrays
xspacing = np.mean(x_1d[1:]-x_1d[0:-1])
x_edges_1d = np.append(x_1d,x_1d[-1]+xspacing) - (xspacing/2.0)
yspacing = np.mean(y_1d[1:]-y_1d[0:-1])
y_edges_1d = np.append(y_1d,y_1d[-1]+yspacing) - (yspacing/2.0)
x_edges, y_edges = np.meshgrid(x_edges_1d,y_edges_1d,indexing='ij')

# remove burnin
I_fit = trace['I'][burnin:]
Q_fit = trace['Q'][burnin:]
U_fit = trace['U'][burnin:]
V_fit = trace['V'][burnin:]
P_fit = np.sqrt((Q_fit**2.0) + (U_fit**2.0) + (V_fit**2.0))/I_fit

# remove divergences and reshape arrays
div_mask = np.invert(np.copy(trace[burnin:].diverging))

std_I = np.std(I_fit[div_mask],axis=0).reshape((nx,ny))
med_I = np.median(I_fit[div_mask],axis=0).reshape((nx,ny))
mean_I = np.mean(I_fit[div_mask],axis=0).reshape((nx,ny))

std_P = np.std(P_fit[div_mask],axis=0).reshape((nx,ny))
med_P = np.median(P_fit[div_mask],axis=0).reshape((nx,ny))
mean_P = np.mean(P_fit[div_mask],axis=0).reshape((nx,ny))

std_Q = np.std(Q_fit[div_mask],axis=0).reshape((nx,ny))
med_Q = np.median(Q_fit[div_mask],axis=0).reshape((nx,ny))
mean_Q = np.mean(Q_fit[div_mask],axis=0).reshape((nx,ny))

std_U = np.std(U_fit[div_mask],axis=0).reshape((nx,ny))
med_U = np.median(U_fit[div_mask],axis=0).reshape((nx,ny))
mean_U = np.mean(U_fit[div_mask],axis=0).reshape((nx,ny))

std_V = np.std(V_fit[div_mask],axis=0).reshape((nx,ny))
med_V = np.median(V_fit[div_mask],axis=0).reshape((nx,ny))
mean_V = np.mean(V_fit[div_mask],axis=0).reshape((nx,ny))

# plot medians
plt.figure()
plt.pcolormesh(x_edges,y_edges,med_I,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesI_median.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,med_P,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Fractional polarization', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesP_median.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,med_Q,vmin=-np.max(np.abs(med_Q)),vmax=np.max(np.abs(med_Q)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesQ_median.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,med_U,vmin=-np.max(np.abs(med_U)),vmax=np.max(np.abs(med_U)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesU_median.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,med_V,vmin=-np.max(np.abs(med_V)),vmax=np.max(np.abs(med_V)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesV_median.png',dpi=300,bbox_inches='tight')
plt.close()

# plot means
plt.figure()
plt.pcolormesh(x_edges,y_edges,mean_I,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesI_mean.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,mean_P,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Fractional polarization', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesP_mean.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,mean_Q,vmin=-np.max(np.abs(mean_Q)),vmax=np.max(np.abs(mean_Q)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesQ_mean.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,mean_U,vmin=-np.max(np.abs(mean_U)),vmax=np.max(np.abs(mean_U)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesU_mean.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,mean_V,vmin=-np.max(np.abs(mean_V)),vmax=np.max(np.abs(mean_V)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesV_mean.png',dpi=300,bbox_inches='tight')
plt.close()

# plot standard deviations
plt.figure()
plt.pcolormesh(x_edges,y_edges,std_I,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesI_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,std_P,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Fractional polarization', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesP_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,std_Q,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesQ_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,std_U,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesU_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,std_V,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Jy per pixel', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesV_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

# plot fractional standard deviations
plt.figure()
plt.pcolormesh(x_edges,y_edges,std_I/mean_I,vmin=0,cmap='afmhot_10us')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Fractional difference', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesI_fractional_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,std_Q/mean_Q,vmin=-np.max(np.abs(std_Q/mean_Q)),vmax=np.max(np.abs(std_Q/mean_Q)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Fractional difference', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesQ_fractional_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,std_U/mean_U,vmin=-np.max(np.abs(std_U/mean_U)),vmax=np.max(np.abs(std_U/mean_U)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Fractional difference', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesU_fractional_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

plt.figure()
plt.pcolormesh(x_edges,y_edges,std_V/mean_V,vmin=-np.max(np.abs(std_V/mean_V)),vmax=np.max(np.abs(std_V/mean_V)),cmap='seismic')
plt.xlim((xmax,xmin))
plt.ylim((ymin,ymax))
plt.xlabel(r'RA ($\mu$as)')
plt.ylabel(r'Dec ($\mu$as)')
cbar = plt.colorbar()
cbar.set_label('Fractional difference', rotation=270, labelpad=20)
plt.axes().set_aspect('equal', 'datalim')
plt.savefig('StokesV_fractional_standard_deviation.png',dpi=300,bbox_inches='tight')
plt.close()

########################################################
# make ehtim image object

im = eh.image.Image(mean_I, psize=xspacing*eh.RADPERUAS, ra=obs.ra, dec=obs.dec, polrep='stokes', rf=obs.rf, source=obs.source, mjd=obs.mjd, time=np.mean(obs.data['time']))

im.ivec = mean_I.T[::-1,::-1].ravel()
im.qvec = mean_Q.T[::-1,::-1].ravel()
im.uvec = mean_U.T[::-1,::-1].ravel()
im.vvec = mean_V.T[::-1,::-1].ravel()

imfig = im.display(plotp=True,cfun='afmhot_10us',show=False)

plt.savefig('polplot.png',bbox_inches='tight',dpi=300)
plt.close()

im.save_fits('image_mean.fits')

########################################################
# gain plots

gain_amps_fit_R = trace['right_gain_amps'][burnin:]
gain_amps_fit_L = trace['left_gain_amps'][burnin:]

std_ga_R = np.std(gain_amps_fit_R[div_mask],axis=0)
med_ga_R = np.median(gain_amps_fit_R[div_mask],axis=0)
std_ga_L = np.std(gain_amps_fit_L[div_mask],axis=0)
med_ga_L = np.median(gain_amps_fit_L[div_mask],axis=0)

plt.figure()
offset = 0.0
ymax = 0.0
for key in SEFD_error_budget.keys():
    index = (A_gains == key)
    plt.plot([-100.0,100.0],[1.0+offset,1.0+offset],'k--',linewidth=0.5,alpha=0.3,zorder=-11)
    plt.plot(T_gains[index],med_ga_R[index]+offset,linewidth=0,marker='o',markersize=3)
    plt.plot(T_gains[index],med_ga_L[index]+offset,linewidth=0,marker='s',markersize=3)
    if index.sum() > 0:
        if np.max(med_ga_R[index]+offset) > ymax:
            ymax = np.max(med_ga_R[index]+offset)
        if np.max(med_ga_L[index]+offset) > ymax:
            ymax = np.max(med_ga_L[index]+offset)
    offset += 1.0

plt.xlim(np.min(T_gains)-0.5,np.max(T_gains)+0.5)
plt.ylim(0,ymax+1)
ylim = plt.gca().get_ylim()

plt.xlabel('Time (hours)')
plt.ylabel('Gain amplitudes')

ax2 = plt.gca().twinx()
ax2.set_yticks(np.arange(1,len(SEFD_error_budget.keys())+1,1))
ax2.set_yticklabels(SEFD_error_budget.keys())
ax2.set_ylim(ylim)

plt.savefig('gain_amplitudes.png',dpi=300,bbox_inches='tight')
plt.close()

########################################################
# gain cornerplots

if not os.path.exists('./gain_phases'):
    os.mkdir('./gain_phases')
if not os.path.exists('./gain_amplitudes'):
    os.mkdir('./gain_amplitudes')

gain_amps_R = trace['right_gain_amps'][burnin:]
gain_amps_L = trace['left_gain_amps'][burnin:]
gain_phases_R = trace['right_gain_phases'][burnin:]
gain_phases_L = trace['left_gain_phases'][burnin:]

count = 0
for it, t in enumerate(timestamps):
    ind_here = (T_gains == t)
    
    ants_here = A_gains[ind_here]

    samples_amps_R = np.ndarray(shape=(len(gain_amps_R[:,count]),len(ants_here)))
    samples_amps_L = np.ndarray(shape=(len(gain_amps_L[:,count]),len(ants_here)))

    samples_phases_R = np.ndarray(shape=(len(gain_phases_R[:,count]),len(ants_here)))
    samples_phases_L = np.ndarray(shape=(len(gain_phases_L[:,count]),len(ants_here)))

    labels_amps = []
    range_amps = []
    labels_phases = []
    range_phases = []
    for ia, ant in enumerate(ants_here):
        samples_amps_R[:,ia] = gain_amps_R[:,count]
        samples_phases_R[:,ia] = gain_phases_R[:,count]

        samples_amps_L[:,ia] = gain_amps_L[:,count]
        samples_phases_L[:,ia] = gain_phases_L[:,count]

        labels_amps.append(r'$|G|_{\rm{'+ant+'}}$')
        labels_phases.append(r'$\theta_{\rm{'+ant+'}}$')

        range_amps.append((0.0,2.0))
        range_phases.append((-np.pi,np.pi))

        count += 1

    fig_amp = corner.corner(samples_amps_R,labels=labels_amps,show_titles=False,title_fmt='.4f',
                        title_kwargs={"fontsize": 12},smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                        plot_density=False,fill_contours=True,range=range_amps,bins=100,color='cornflowerblue')
    corner.corner(samples_amps_L,fig=fig_amp,smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                        plot_density=False,fill_contours=True,range=range_amps,bins=100,color='salmon')
    fig_amp.savefig('./gain_amplitudes/gain_amps_scan'+str(it).zfill(5)+'.png',dpi=300)
    plt.close()

    fig_phase = corner.corner(samples_phases_R,labels=labels_phases,show_titles=False,title_fmt='.4f',
                        title_kwargs={"fontsize": 12},smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                        plot_density=False,fill_contours=True,range=range_phases,bins=100,color='cornflowerblue')
    corner.corner(samples_phases_L,fig=fig_phase,smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                        plot_density=False,fill_contours=True,range=range_phases,bins=100,color='salmon')
    fig_phase.savefig('./gain_phases/gain_phases_scan'+str(it).zfill(5)+'.png',dpi=300)
    plt.close()

########################################################
# D term cornerplots: amplitude and phase

Dterm_amps_R = trace['right_Dterm_amps'][burnin:]
Dterm_amps_L = trace['left_Dterm_amps'][burnin:]
Dterm_phases_R = trace['right_Dterm_phases'][burnin:]
Dterm_phases_L = trace['left_Dterm_phases'][burnin:]

samples_amps_R = np.ndarray(shape=(len(Dterm_amps_R[:,0]),N_Dterms))
samples_amps_L = np.ndarray(shape=(len(Dterm_amps_L[:,0]),N_Dterms))
samples_phases_R = np.ndarray(shape=(len(Dterm_phases_R[:,0]),N_Dterms))
samples_phases_L = np.ndarray(shape=(len(Dterm_phases_L[:,0]),N_Dterms))

labels_amp = []
labels_phase = []
ranges_amp = []
ranges_phase = []
for istat, station in enumerate(stations):
    samples_amps_R[:,istat] = Dterm_amps_R[:,istat]
    samples_amps_L[:,istat] = Dterm_amps_L[:,istat]

    samples_phases_R[:,istat] = Dterm_phases_R[:,istat]
    samples_phases_L[:,istat] = Dterm_phases_L[:,istat]

    labels_amp.append(r'$|D|_{\rm{'+station+'}}$')
    labels_phase.append(r'$\delta_{\rm{'+station+'}}$')
    ranges_amp.append((0.0,1.0))
    ranges_phase.append((-np.pi,np.pi))

fig_amps = corner.corner(samples_amps_R,labels=labels_amp,show_titles=False,title_fmt='.4f',
                    title_kwargs={"fontsize": 12},smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                    plot_density=False,fill_contours=True,range=ranges_amp,bins=100,color='cornflowerblue')
corner.corner(samples_amps_L,fig=fig_amps,smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                    plot_density=False,fill_contours=True,range=ranges_amp,bins=100,color='salmon')
fig_amps.savefig('Dterms_amplitude.png',dpi=300)
plt.close()

fig_phases = corner.corner(samples_phases_R,labels=labels_phase,show_titles=False,title_fmt='.4f',
                    title_kwargs={"fontsize": 12},smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                    plot_density=False,fill_contours=True,range=ranges_phase,bins=100,color='cornflowerblue')
corner.corner(samples_phases_L,fig=fig_phases,smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                    plot_density=False,fill_contours=True,range=ranges_phase,bins=100,color='salmon')
fig_phases.savefig('Dterms_phase.png',dpi=300)
plt.close()

########################################################
# D term cornerplots: real and imaginary

Dterm_reals_R = trace['right_Dterm_reals'][burnin:]
Dterm_reals_L = trace['left_Dterm_reals'][burnin:]
Dterm_imags_R = trace['right_Dterm_imags'][burnin:]
Dterm_imags_L = trace['left_Dterm_imags'][burnin:]

samples_reals_R = np.ndarray(shape=(len(Dterm_reals_R[:,0]),N_Dterms))
samples_reals_L = np.ndarray(shape=(len(Dterm_reals_L[:,0]),N_Dterms))
samples_imags_R = np.ndarray(shape=(len(Dterm_imags_R[:,0]),N_Dterms))
samples_imags_L = np.ndarray(shape=(len(Dterm_imags_L[:,0]),N_Dterms))

labels_real = []
labels_imag = []
ranges = []
for istat, station in enumerate(stations):
    samples_reals_R[:,istat] = Dterm_reals_R[:,istat]
    samples_reals_L[:,istat] = Dterm_reals_L[:,istat]

    samples_imags_R[:,istat] = Dterm_imags_R[:,istat]
    samples_imags_L[:,istat] = Dterm_imags_L[:,istat]

    labels_real.append(r'$\rm{Re}(D_{\rm{'+station+'}})$')
    labels_imag.append(r'$\rm{Im}(D_{\rm{'+station+'}})$')
    ranges.append((-1.0,1.0))

fig_reals = corner.corner(samples_reals_R,labels=labels_real,show_titles=False,title_fmt='.4f',
                    title_kwargs={"fontsize": 12},smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                    plot_density=False,fill_contours=True,range=ranges,bins=100,color='cornflowerblue')
corner.corner(samples_reals_L,fig=fig_reals,smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                    plot_density=False,fill_contours=True,range=ranges,bins=100,color='salmon')
fig_reals.savefig('Dterms_realpart.png',dpi=300)
plt.close()

fig_imags = corner.corner(samples_imags_R,labels=labels_imag,show_titles=False,title_fmt='.4f',
                    title_kwargs={"fontsize": 12},smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                    plot_density=False,fill_contours=True,range=ranges,bins=100,color='cornflowerblue')
corner.corner(samples_imags_L,fig=fig_imags,smooth=0.8,smooth1d=0.8,plot_datapoints=False,
                    plot_density=False,fill_contours=True,range=ranges,bins=100,color='salmon')
fig_imags.savefig('Dterms_imagpart.png',dpi=300)
plt.close()

########################################################
# D term plots for individual stations

if not os.path.exists('./Dterms'):
    os.mkdir('./Dterms')

levels = [1.0-np.exp(-0.5*(1.1775**2.0)),1.0-np.exp(-0.5*(2.146**2.0)),1.0-np.exp(-0.5*(3.035**2.0))]

for istat, station in enumerate(stations):

    fig = plt.figure(figsize=(6,6))

    plt.plot([-10,10],[0,0],'k--',alpha=0.5,zorder=-10)
    plt.plot([0,0],[-10,10],'k--',alpha=0.5,zorder=-10)

    corner.hist2d(Dterm_reals_R[:,istat],Dterm_imags_R[:,istat],fig=fig, levels=levels, color='cornflowerblue', fill_contours=True,plot_datapoints=False,plot_density=False,plot_contours=True,smooth=0.5)
    corner.hist2d(Dterm_reals_L[:,istat],Dterm_imags_L[:,istat],fig=fig, levels=levels, color='salmon', fill_contours=True,plot_datapoints=False,plot_density=False,plot_contours=True,smooth=0.5)
    
    print(station+' R: real = '+str(np.mean(Dterm_reals_R[:,istat]))+',   imag = '+str(np.mean(Dterm_imags_R[:,istat])))
    print(station+' L: real = '+str(np.mean(Dterm_reals_L[:,istat]))+',   imag = '+str(np.mean(Dterm_imags_L[:,istat])))

    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')

    limit = np.max(np.array([np.max(Dterm_amps_R[:,istat]),np.max(Dterm_amps_L[:,istat])]))
    plt.xlim(-1.1*limit,1.1*limit)
    plt.ylim(-1.1*limit,1.1*limit)

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
    plt.text(-1.05*limit,1.05*limit,strhere,ha='left',va='top',fontsize=8)

    plt.savefig('./Dterms/Dterms_'+station+'.png',dpi=300,bbox_inches='tight')
    plt.close()

########################################################
# energy plot

en = np.copy(trace[burnin:].energy)
en_diff = np.diff(en)

en = en[div_mask]
en_diff = en_diff[div_mask[:-1]]

enm = en - en.mean()

xupper = 1.1*max([enm.max(),en_diff.max()])
xlower = 1.1*min([enm.min(),en_diff.min()])

plt.hist(enm,label='energy',alpha=0.5,bins=50,range=(xlower,xupper),density=True)
plt.hist(en_diff,label='energy difference',alpha=0.5,bins=50,range=(xlower,xupper),density=True)

plt.xlabel('Energy')

plt.legend()
plt.savefig('energy.png',dpi=300)
plt.close()

########################################################
# step size plot

plt.plot(trace.get_sampler_stats('step_size'))

plt.semilogy()

plt.ylabel('Step size')
plt.xlabel('Trial number')

plt.savefig('stepsize.png',dpi=300)
plt.close()