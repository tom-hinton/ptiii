import numpy as np
from scipy.fft import fft
from scipy.signal.windows import blackman
from utils.dynsys_utils import \
	calc_dtheta_EVT, \
	embed, calc_lyap_spectrum, \
	calc_dim_Cao1997, \
	_calc_tangent_map
import matplotlib.pyplot as plt
from nolitsa.delay import dmi
from scipy.signal import find_peaks


# CALCULATE FFT
def calculate_fft(data):
    print("Applying fft...")
    w = blackman( len(data) )
    y = data["SHEAR STRESS"].to_numpy()
    yf = fft(y*w)
    print("...applied.")
    return yf


# CALCULATE tau BY AMI
max_tau = 10000
min_ami_peak_width = 100
def calculate_tau_ami(data):
    print("Calculating best delay time with AMI (Fraser and Swinney, 1986)...")

    AMI = dmi(data["NORMALISED SHEAR"], maxtau=max_tau)
    minima, _ = find_peaks(-AMI, width=100)
    first_minimum = minima[0]
    print(first_minimum)
    print("... calculated.")

    plt.plot(AMI)
    plt.scatter(minima, AMI[minima], color="red")
    plt.show()

    return first_minimum


# # CALCULATE BEST m, tau BY MINIMISING LYAPUNOV RADIUS
tau_to_try = np.array([5,20,100,500,1000])
m_to_try = np.array([3,5,8,12])
E1_threshold = 0.9
E2_threshold = 0.9
eps_over_L0 = 0.05

def calculate_best_m_tau(data):
    print("Calculating best m, tau by minimising the Lyapunov radius...")
    mhat = np.empty(len(tau_to_try), dtype=np.int8)
    eps_over_L = np.empty(len(tau_to_try))
    for i, tau_i in enumerate(tau_to_try):
        print("Looping through tau values: " + str(i+1) + "/" + str(len(tau_to_try)))
        mhat_i, E1, E2 = calc_dim_Cao1997(X=data["NORMALISED SHEAR"], \
					tau=tau_i, m=m_to_try, \
					E1_thresh=E1_threshold, E2_thresh=E2_threshold, \
					qw=None, flag_single_tau=True, parallel=False)
        print('Results from Cao: mhat_i, E1, E2')
        print(mhat_i)
        print(E1)
        print(E2)
        if ~np.isnan(mhat_i):
            mhat[i] = mhat_i
            H, tH = embed(data["NORMALISED SHEAR"], tau=[tau_i], \
                            m=[int(mhat_i)], t=data["TIME"])
            _, eps_over_L[i] = _calc_tangent_map(H, \
                                n_neighbors=20, eps_over_L0=eps_over_L0)

    print("Loops finished. Eps_over_L:")
    print(eps_over_L)
    best_mhat = mhat[np.nanargmin(eps_over_L)]
    best_tau = tau_to_try[np.nanargmin(eps_over_L)]
    return [best_mhat, best_tau]


# CALCULATE LYAPUNOV
eps_over_L0_ = eps_over_L0
LEs_sampling = ['rand',None]
def calculate_lyapunov_exponents(data, m, tau):
    H, tH = embed(data["NORMALISED SHEAR"], tau=[tau], m=[m], t=data["TIME"])
    LEs, LEs_mean, LEs_std, eps_over_L = calc_lyap_spectrum(H, sampling=LEs_sampling, eps_over_L0=eps_over_L0_, n_neighbors=20)
    return LEs_mean[-1,:]