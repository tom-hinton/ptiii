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
import utils.plot_utils as plot


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


# CALCULATE BEST m, tau BY MINIMISING LYAPUNOV RADIUS
def calculate_best_m_tau(data, params, fig=False, save=False):
    print("Calculating best m, tau by minimising the Lyapunov radius:")
    tau_to_try = params["tau_to_try"]
    m_to_try = params["m_to_try"]

    m = np.empty(len(tau_to_try))
    eps_over_L = np.empty(len(tau_to_try))
    eps_over_L[:] = np.nan
    E1s = np.empty((len(tau_to_try), len(m_to_try)-1))
    E2s = np.empty((len(tau_to_try), len(m_to_try)-1))

    for i, tau_i in enumerate(tau_to_try):
        print("Loop ", str(i+1), "/", str(len(tau_to_try)), ": tau = ", str(tau_i), sep="")

        m_i, E1s[i], E2s[i] = calc_dim_Cao1997(X=data["NORMALISED SHEAR"], \
					tau=tau_i, m=m_to_try, \
					E1_thresh=params["E1_threshold"], E2_thresh=params["E2_threshold"], \
					qw=None, flag_single_tau=True, parallel=False)
        m[i] = m_i

        if ~np.isnan(m_i):
            H, tH = embed(data["NORMALISED SHEAR"], tau=[tau_i], \
                            m=[int(m_i)], t=data["TIME"])
            _, eps_over_L[i] = _calc_tangent_map(H, \
                                n_neighbors=params["n_neighbors"], eps_over_L0=params["eps_over_L0"])
    
    if np.isnan(eps_over_L).all():
        best_m = np.nan
        best_tau = np.nan
    else:
        best_m = int(m[np.nanargmin(eps_over_L)])
        best_tau = tau_to_try[np.nanargmin(eps_over_L)]

    if fig==True:
        plot.summary_calc_m_tau(data, params, E1s, E2s, eps_over_L, best_m, best_tau, save=True)

    return [best_m, best_tau]


# CALCULATE LYAPUNOV
def calculate_lyapunov_exponents(data, params, m, tau):
    H, tH = embed(data["NORMALISED SHEAR"], tau=[tau], m=[m], t=data["TIME"])
    LEs_, LEs_mean, LEs_std, eps_over_L = calc_lyap_spectrum(H, sampling=params["LEs_sampling"], eps_over_L0=params["eps_over_L0"], n_neighbors=params["n_neighbors"], verbose=False)
    LEs = LEs_mean[-1,:]

    if np.sum(LEs) > 0:
        return [LEs, np.nan]

    i = 1
    while i <= len(LEs) and np.sum(LEs[:i]) > 0:
        i += 1
    
    return [LEs, i]