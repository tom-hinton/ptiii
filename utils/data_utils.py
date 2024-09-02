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
tau_to_try = np.array([1,5,20,50])
m_to_try = np.arange(1,15,1)
E1_threshold = 0.9
E2_threshold = 0.9
eps_over_L0 = 0.05

def calculate_best_m_tau(data):
    print("\nCalculating best m, tau by minimising the Lyapunov radius:")

    m = np.empty(len(tau_to_try))
    eps_over_L = np.empty(len(tau_to_try))
    eps_over_L[:] = np.nan

    # plt.figure()
    # plt.title("E1 & E2")
    # plt.xlabel("Embedding dimension m")
    # plt.ylabel("E value")
    # plt.axhline(y=E1_threshold, color="lightgrey", label="E1 threshold")

    for i, tau_i in enumerate(tau_to_try):
        print("\nLoop ", str(i+1), "/", str(len(tau_to_try)), ": tau = ", str(tau_i), sep="")

        m_i, E1, E2 = calc_dim_Cao1997(X=data["NORMALISED SHEAR"], \
					tau=tau_i, m=m_to_try, \
					E1_thresh=E1_threshold, E2_thresh=E2_threshold, \
					qw=None, flag_single_tau=True, parallel=False)
        m[i] = m_i

        label = "Tau = " + str(tau_i)
        color = "C" + str(i)
        # plt.plot(E1, color=color, label=label)
        # plt.plot(E2, color=color)

        if ~np.isnan(m_i):
            H, tH = embed(data["NORMALISED SHEAR"], tau=[tau_i], \
                            m=[int(m_i)], t=data["TIME"])
            _, eps_over_L[i] = _calc_tangent_map(H, \
                                n_neighbors=20, eps_over_L0=eps_over_L0)

    # plt.legend()

    # plt.figure()
    # plt.title("Eps over L")
    # plt.xlabel("Tau delay")
    # plt.plot(tau_to_try, eps_over_L)

    if np.isnan(eps_over_L).all():
        return [np.nan, np.nan]

    best_m = int(m[np.nanargmin(eps_over_L)])
    best_tau = tau_to_try[np.nanargmin(eps_over_L)]
    return [best_m, best_tau]


# CALCULATE LYAPUNOV
eps_over_L0_ = eps_over_L0
LEs_sampling = ['rand',None]
def calculate_lyapunov_exponents(data, m, tau):
    H, tH = embed(data["NORMALISED SHEAR"], tau=[tau], m=[m], t=data["TIME"])
    LEs_, LEs_mean, LEs_std, eps_over_L = calc_lyap_spectrum(H, sampling=LEs_sampling, eps_over_L0=eps_over_L0_, n_neighbors=20, verbose=False)
    LEs = LEs_mean[-1,:]

    if np.sum(LEs) > 0:
        return [LEs, np.nan]

    i = 1
    while i <= len(LEs) and np.sum(LEs[:i]) > 0:
        i += 1
    
    return [LEs, i]