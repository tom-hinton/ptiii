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


# CALCULATE BEST m, tau BY MINIMISING LYAPUNOV RADIUS
def calculate_best_m_tau(data, params, plot=False, save=False):
    print("Calculating best m, tau by minimising the Lyapunov radius:")
    tau_to_try = params["tau_to_try"]
    m_to_try = params["m_to_try"]

    m = np.empty(len(tau_to_try))
    eps_over_L = np.empty(len(tau_to_try))
    eps_over_L[:] = np.nan

    if plot==True:
        fig,axs=plt.subplots(2, 2)
        fig.set_size_inches((11,9))

        axs[0,0].plot(data["TIME"], data["NORMALISED SHEAR"])
        axs[0,0].set_title("Time Series")
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel("Shear Stress")

    for i, tau_i in enumerate(tau_to_try):
        print("Loop ", str(i+1), "/", str(len(tau_to_try)), ": tau = ", str(tau_i), sep="")

        m_i, E1, E2 = calc_dim_Cao1997(X=data["NORMALISED SHEAR"], \
					tau=tau_i, m=m_to_try, \
					E1_thresh=params["E1_threshold"], E2_thresh=params["E2_threshold"], \
					qw=None, flag_single_tau=True, parallel=False)
        m[i] = m_i

        if plot==True:
            label = "Tau = " + str(tau_i)
            color = "C" + str(i)
            axs[0,1].plot(E1, color=color, label=label)
            axs[0,1].plot(E2, color=color)

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

    if plot==True:
        axs[0,1].set_title("E1 & E2")
        axs[0,1].set_xlabel("Embedding dimension m")
        axs[0,1].set_ylabel("E value")
        axs[0,1].axhline(y=params["E1_threshold"], color="lightgrey", label="E1 threshold")
        axs[0,1].legend()

        axs[1,0].scatter(tau_to_try, eps_over_L)
        axs[1,0].set_title("Eps over L")
        axs[1,0].set_xlabel("Tau delay")
        axs[1,0].set_ylabel("Eps over L")

        if ~np.isnan(eps_over_L).all():
            axs[1,1].remove()
            axs[1,1] = fig.add_subplot(2,2,4,projection="3d")
            H, tH = embed(data["NORMALISED SHEAR"], tau=[best_tau], m=[best_m], t=data["TIME"])
            axs[1,1].plot(H[:,0], H[:,1], H[:,2])
            axs[1,1].set_title("First embedding")

        fig.suptitle("Window: m=" + str(best_m) + ", tau=" + str(best_tau))
        if save:
            filename = "/snapshot_window_" + str(params["current_loop"]) + ".png" if "current_loop" in params else "/summary_calc_m_tau.png"
            fig.savefig(params["results_dir"] + filename)


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