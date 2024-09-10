import os
import pickle
import time
import numpy as np
import pandas as pd
from operator import itemgetter
from utils.load_utils import normalise_shear
from utils.data_utils import calculate_best_m_tau, calculate_lyapunov_exponents


# LOAD RUN OPTIONS
run_code = "10"
print("Run code:", run_code)
opts_picklepath = os.getcwd() +"/results/" + run_code + "/opts.pickle"
with open(opts_picklepath, "rb") as f:
    opts = pickle.load(f)
picklepath, results_dir, windows, dynsys_params = itemgetter("picklepath", "results_dir", "windows", "dynsys_params")(opts)


# LOAD DATA FROM PICKLE
with open(picklepath, "rb") as f:
    data_long = pickle.load(f)


# PREPARE output.csv
outputpath = results_dir + "/output.pickle"
output = pd.DataFrame(columns=["win_start", "win_end", "t_start", "t_end", "m", "tau", "eps_over_L", "E1s", "E2s", "LEs", "KYD"])


# THE LOOP
for i, win in enumerate(windows):
    print("WINDOW ", (i+1), "/", len(windows))
    tic = time.time()

    # GET WINDOW DATA AND NORMALISE
    data = data_long[win[0]:win[1]].copy()
    data = normalise_shear(data)

    # CALCULATE BEST M, TAU
    m, tau, eps_over_L, E1s, E2s = calculate_best_m_tau(data, **dynsys_params)

    # CALCULATE LYAPUNOV EXPONENTS AND KAPLAN-YORKE DIMENSION
    if ~np.isnan(m):
        LEs, KYD = calculate_lyapunov_exponents(data, m, tau, **dynsys_params)
    else:
        LEs, KYD = [], np.nan

    # PICKLE TO OUTPUT
    output.loc[i] = [win[0], win[1], data.iloc[0]["TIME"], data.iloc[-1]["TIME"], m, tau, eps_over_L, E1s, E2s, LEs, KYD]
    with open(outputpath, 'wb') as f:
        pickle.dump(output, f)
    
    print("Window time taken: ", round(time.time()-tic, 1), "s")
    print("\n" + ("_"*40) + "\n")
    