import numpy as np
from scipy.fft import fft
from scipy.signal.windows import blackman
from utils.dynsys_utils import \
	calc_dtheta_EVT, \
	embed, calc_lyap_spectrum, \
	_calc_tangent_map
import matplotlib.pyplot as plt
from nolitsa.dimension import afn
from nolitsa.delay import dmi
from scipy.signal import find_peaks
import utils.plot_utils as plot
import time


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


# CALCULATE BEST m, 
def calculate_best_m_tau(data, **opts):
    print("Calculating best m, tau...")
    tau_to_try = opts["tau_to_try"]
    m_to_try = opts["m_to_try"]

    m = np.empty(len(tau_to_try))
    eps_over_L = np.empty(len(tau_to_try)) * np.nan
    E1s = np.empty((len(tau_to_try), len(m_to_try)-1))
    E2s = np.empty((len(tau_to_try), len(m_to_try)-1))

    for i, tau_i in enumerate(tau_to_try):
        print("Loop ", (i+1), "/", len(tau_to_try), ": tau = ", tau_i, sep="")
        tic = time.time()

        m[i], E1s[i], E2s[i] = calc_dim_Cao1997(X=data["NORMALISED SHEAR"], tau=tau_i, m=m_to_try, qw=None, flag_single_tau=True, parallel=False, **opts)

        if ~np.isnan(m[i]):
            H, _ = embed(data["NORMALISED SHEAR"], tau=[tau_i], m=[int(m[i])], t=data["TIME"])
            _, eps_over_L[i] = _calc_tangent_map(H, **opts)
            print("Result: m=", m[i], ", eps/L=", round(eps_over_L[i], 3), ". Calculation time=", round(time.time()-tic, 1), "s", sep="")
        else:
            print("Result: E2 ~ const., time series is stochastic. Calculation time = ", tic-time.time(), "s")
    
    if np.isnan(eps_over_L).all():
        return [np.nan, np.nan, eps_over_L, E1s, E2s]
    best_m = int(m[np.nanargmin(eps_over_L)])
    best_tau = int(tau_to_try[np.nanargmin(eps_over_L)])

    print("...best m = ", best_m, ", tau = ", best_tau, sep="")
    return [best_m, best_tau, eps_over_L, E1s, E2s]


# CALCULATE LYAPUNOV
def calculate_lyapunov_exponents(data, m, tau, **args):
    print("Calculating Lyapunov exponents and Kaplan-Yorke dimension...")
    H, _ = embed(data["NORMALISED SHEAR"], tau=[tau], m=[m], t=data["TIME"])
    LEs_, LEs_mean, LEs_std, eps_over_L = calc_lyap_spectrum(H, verbose=False, **args)
    LEs = LEs_mean[-1,:]

    if np.sum(LEs) > 0:
        return [LEs, np.nan]

    i = 1
    while i <= len(LEs) and np.sum(LEs[:i]) > 0:
        i += 1
    if np.sum(LEs[:-1]) < 0:
        kyd = i + np.sum(LEs[:i])/LEs[i]
    else:
        kyd = np.nan

    print("... KYD=", round(kyd,2), ", LEs: ", LEs, sep="")
    return [LEs, kyd]

# CALCULATE EMBEDDING DIMENSION BY CAO 1997
def calc_dim_Cao1997(X, tau=1, m=np.arange(1,21,1), E1_thresh=0.95, E2_thresh=0.95, mw=2, qw=None, window=10, flag_single_tau=False, parallel=True, **args):
    
    if qw==None:
        qw = _calc_autocorr_time(X)

    E, Es = afn(X, dim=m, tau=tau, maxnum=None, window=int(qw), parallel=parallel)

    E1 = E[1:]/E[:-1]
    E2 = Es[1:]/Es[:-1]
    
    if np.sum(E2<E2_thresh)>0:
        indE1 = np.argmax(E1>=E1_thresh)
        mhat = int(m[indE1])
    else:
        mhat = np.nan

    return mhat, E1, E2

# CALC AUTOCORRELATION TIME
def _calc_autocorr_time(X):
	
    Nt = len(X)

    Nbatches  = int(np.ceil(Nt**(1/3)))
    sizebatch = int(np.ceil(Nt**(2/3)))
    ind_tbatch_start = np.ceil((Nt-sizebatch)*np.random.rand(Nbatches))

    var_x = np.var(X)
    xbatches = np.zeros((Nbatches,sizebatch));
    for bb in np.arange(Nbatches):
        xbatches[bb,:] = X[int(ind_tbatch_start[bb]):int(ind_tbatch_start[bb]+sizebatch)];
    mu_xbatches = np.mean(xbatches,axis=1);
    var_muxbatches = np.var(mu_xbatches);
    autocorr_time = sizebatch*var_muxbatches/var_x;
    return autocorr_time