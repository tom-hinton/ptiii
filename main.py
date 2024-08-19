# IMPORTS
import os
import pandas as pd
import numpy as np
from utils.load_utils import import_brava_data
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft
from utils.dynsys_utils import \
	calc_dtheta_EVT, \
	embed, calc_lyap_spectrum, \
	calc_dim_Cao1997, \
	_calc_tangent_map


# KEY VARIABLES
main_dir = os.getcwd()
data_dir = main_dir + '/data'
case_study = 'b1378'
filename = data_dir + '/' + case_study + '.txt'
picklename = data_dir + '/' + case_study + '.pickle'


# PARAMS
params = {
    "t_start": 5000,
    "t_end": 8500,
    "dt": 0.001
}

# IMPORT DATA
data = import_brava_data(filename, picklename, params)
print("...data imported.")





# FFT
tau_fft = fft(data["V-LOAD"].to_numpy())
plt.plot(tau_fft)
plt.show()



# # Embedding dimensions to test
# m_list = [3,5,8,10,20,100]
# Nm = len(m_list)

# # Minimum distance between peaks (add 1 for scipy notation), i.e. if you want
# # a minimum distance of 3 use 4
# min_dist = [2,5,10]
# Nmin_dist = len(min_dist)

# tau_delay = []
# tt=-1
# for mm in min_dist:
# 	tt+=1
# 	tau_delay.append(min_dist[tt]-1)
# Ntau_delay = len(tau_delay)

# smoothes = [1.0]

# p_EVT         = 2
# n_neighbors_min_EVT = 100
# dq_thresh_EVT = None
# fit_distr     = False

# LEs_sampling = ['rand',None]
# eps_over_L0 = 0.05

# maxtau = 4000
# mmax = 20
# E1_thresh = 0.9
# E2_thresh = 0.9


# # REMOVE LINEAR SHEAR HARDENING TREND FROM SHEAR STRESS
# data["Time"] = data["Time"]-data["Time"][0]
# p = np.polyfit(data['Time'], data["V-LOAD"], deg=1)
# V_load_notrend = data["V-LOAD"] - (p[0]*data["Time"] + p[1])

# X = np.array([V_load_notrend.to_numpy()]).T
# Nt,Nx = X.shape
# X_norm = np.zeros((Nt,Nx))
# for ii in np.arange(Nx):
#     X_norm[:,ii] = (X[:,ii]-np.min(X[:,ii])) / (np.max(X[:,ii])-np.min(X[:,ii]))


# # CALCULATE LYAPUNOV SPECTRUM
# mhat = []
# LEs        = []
# LEs_mean   = []
# LEs_std    = []
# eps_over_L = np.zeros((Ntau_delay,1))
# mhat       = np.zeros((Ntau_delay,1))

# # BEST EMBEDDING PARAMETERS
# # m FROM CAO (1997) [tau as minimum tau for which m is determined]
# if tau_delay==0:
#     mhat_tmp = [[np.nan]]
#     while np.isnan(mhat_tmp[0][0]):
#         tau_delay+=1
#         mhat_tmp, E1, E2 = calc_dim_Cao1997(X=X_norm, tau=[tau_delay], \
#                     m=np.arange(1,mmax+1,1),E1_thresh=E1_thresh, \
#                     E2_thresh=E2_thresh, qw=None, \
#                     flag_single_tau=False, parallel=False)
    
#     H, tH = embed(X_norm, tau=[tau_delay], \
#                             m=[int(mhat_tmp[0][0])], t=data["Time"])
#     LEs_tmp, LEs_mean_tmp, LEs_std_tmp, eps_over_L_tmp = \
#         calc_lyap_spectrum(H, sampling=LEs_sampling, \
#                 eps_over_L0=eps_over_L0, n_neighbors=20)
#     mhat.append(mhat_tmp)
#     LEs.append(LEs_tmp)
#     LEs_mean.append(LEs_mean_tmp)
#     LEs_std.append(LEs_std_tmp)
#     eps_over_L.append(eps_over_L_tmp)
# else:
#     for tautau in np.arange(Ntau_delay):
#         mhat_tmp, E1, E2 = calc_dim_Cao1997(X=X_norm, \
#                     tau=[tau_delay[tautau]], m=np.arange(1,mmax+1,1), \
#                     E1_thresh=E1_thresh, E2_thresh=E2_thresh, \
#                     qw=None, flag_single_tau=True, parallel=False)
#         mhat[tautau] = mhat_tmp[0][0]
#         if np.isnan(mhat_tmp[0][0]):
#             eps_over_L[tautau] = np.nan
#         else:
#             H, tH = embed(X_norm, tau=[tau_delay[tautau]], \
#                             m=[int(mhat_tmp[0][0])], t=data["Time"])
#             _, eps_over_L[tautau] = _calc_tangent_map(H, \
#                                 n_neighbors=20, eps_over_L0=eps_over_L0)
#     # Define the best embedding parameters as those such that the
#     # tangent map for the calculation of the Lyapunov spectrum is
#     # calculated using the smallest radius in order to have the
#     # required number of neighbors
#     tau_best = int(tau_delay[np.nanargmin(eps_over_L)])
#     mhat_best = int(mhat[np.nanargmin(eps_over_L)])
#     print("Printing best Lyapunov parameters")
#     print("m = %d, tau = %d" %(mhat_best, tau_best))
#     H, tH = embed(X_norm, tau=[tau_best], \
#                             m=[mhat_best], t=data["Time"])
#     LEs, LEs_mean, LEs_std, eps_over_L = \
#                 calc_lyap_spectrum(H, sampling=LEs_sampling, \
#                             eps_over_L0=eps_over_L0, n_neighbors=20)