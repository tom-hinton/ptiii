# IMPORTS
import os
import pandas as pd
import numpy as np
from utils.load_utils import import_brava_data
from utils.data_utils import calculate_fft, calculate_tau_ami, calculate_best_m_tau, calculate_lyapunov_exponents
from utils.dynsys_utils import calc_dim_Cao1997
import matplotlib.pyplot as plt


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
    "resample_rate": 100
}

# IMPORT DATA
data = import_brava_data(filename, picklename, params)
l = len(data)
print("...data imported.")
# print("Data head:")
# print(data.head())
# plt.plot(data["TIME"], data["SHEAR STRESS"])
# plt.show()


# CALCULATE FFT
# fft = calculate_fft(data)
# plt.plot(fft)
# plt.show()


# CALCULATE BEST tau BY AMI
# tau_ami = calculate_tau_ami(data)


# CALCULATE BEST m, tau BY MIN LYAPUNOV RADIUS
m, tau = calculate_best_m_tau(data)
print("Printing m, tau")
print(m)
print(tau)


# CALCULATE LYAPUNOV EXPONENTS
LEs = calculate_lyapunov_exponents(data, m, tau)