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
case_study = 'b1383'
filename = data_dir + '/' + case_study + '.txt'
picklename = data_dir + '/' + case_study + '.pickle'


# PARAMS
params = {
    "t_start": 4900,
    "t_end": 5600,
    "resample_rate": 50
}

# IMPORT DATA
data, _ = import_brava_data(filename, picklename, params)
print("...data imported.")
print(_.head())
plt.plot(_["TIME"], _["SHEAR STRESS"])
plt.plot(data["TIME"], data["SHEAR STRESS"])
plt.show()
print("Data head:")
print(data.head())


# CALCULATE FFT
# fft = calculate_fft(data)
# plt.plot(fft)
# plt.show()


# CALCULATE BEST tau BY AMI
# tau_ami = calculate_tau_ami(data)
# print("printing tau_ami")
# print(tau_ami)


# CALCULATE BEST m, tau BY MIN LYAPUNOV RADIUS
# m, tau = calculate_best_m_tau(data)
# print("Printing m, tau")
# print(m)
# print(tau)


# CALCULATE LYAPUNOV EXPONENTS
# LEs = calculate_lyapunov_exponents(data, m, tau)
# print("LEs:")
# print(LEs)