#########################
###   GENERAL SETUP   ###
#########################

# REDIRECT IMPORTS
import sys
sys.path.append("../")
data_dir = "../data"

# IMPORTS
import os
from utils.load_utils import import_brava_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




#################################
###   EXPERIMENT PARAMETERS   ###
#################################

###   SLOW SERPENTINITE   ###
sampling_freq = 100
datapath = data_dir + "/b1378.txt"
picklepath = data_dir + "/b1378_clipped.pickle"
t_start=12000
t_end=13000
mov_mean_window = 10  # params
event_thresh_derivval = 2.5
event_halfwidth = 100

###   FAST SERPENTINITE   ###
sampling_freq = 100
datapath = data_dir + "/b1385.txt"
picklepath = data_dir + "/b1385_clipped.pickle"
t_start=12000
t_end=13000
mov_mean_window = 100  # params
event_thresh_derivval = 0.5
event_thresh_derivdist = 500
event_halfwidth = 250

###   MIN-U-SIL   ###
sampling_freq = 50
datapath = data_dir + "/b1383.txt"
picklepath = data_dir + "/b1383_50hz.pickle"
t_start = 5800
t_end = 5900
mov_mean_window = 5 # params
event_thresh_derivval = 0.5
event_halfwidth = 20



########################
###   RUN ANALYSIS   ###
########################
from scipy.signal import find_peaks, detrend
from scipy.ndimage import uniform_filter1d
from findiff import FinDiff

# IMPORT AND HAVE A LOOK
data = import_brava_data(datapath, picklepath, downsample_factor=(1000 / sampling_freq), t_start=t_start, t_end=t_end)
data["SHEAR DETRENDED"] = detrend(data["SHEAR STRESS"], bp=np.arange(0, len(data), 1000*sampling_freq))
# plt.scatter(data["TIME"], data["SHEAR STRESS"])
# plt.show()

# CALCULATE MOVING MEAN DERIVATIVE
# movmean = uniform_filter1d(data["SHEAR STRESS"], mov_mean_window, mode="nearest")
# movmean_deriv = FinDiff(0, 1/sampling_freq, 1)(movmean)

# IDENTIFY EVENTS
# peaks, _ = find_peaks(-movmean_deriv, height=event_thresh_derivval, distance=event_thresh_derivdist)

# ALT EVENT IDENTIFIER
peaks = (data["SHEAR DETRENDED"]<0) & (data["SHEAR DETRENDED"].shift(-50)<0) & (data.shift(1)["SHEAR DETRENDED"]>0) & (data.shift(50)["SHEAR DETRENDED"]>0) & (data["TIME"] > 3790.5)
peaks_i = np.arange(0, len(data), 1)[peaks]
diff = np.empty(len(peaks_i))
diff[0] = np.inf
diff[1:] = np.diff(peaks_i)
peaks_i = peaks_i[diff > 100]
print(peaks_i)
# print(peaks_i[diff < 20])

fig, ax = plt.subplots()
ax.plot(data["TIME"], data["SHEAR DETRENDED"], zorder=1)
ax.scatter(data["TIME"][peaks_i], data["SHEAR DETRENDED"][peaks_i], color="orange", zorder=2)
plt.show()

# fig, ax1 = plt.subplots()
# # ax1.scatter(data["TIME"], data["SHEAR STRESS"], label="shear stress", s=1., color="g")
# ax1.plot(data["TIME"], movmean, label="movmean")
# # ax1.scatter(data["TIME"][peaks], movmean[peaks], color="orange")
# ax2 = ax1.twinx()
# ax2.plot(data["TIME"], movmean_deriv, color="red", label="mov mean deriv")
# ax2.scatter(data["TIME"][peaks], movmean_deriv[peaks], color="orange")
# fig.legend()
# plt.show()
events_list=[]
fig, ax = plt.subplots()
for i, peak in enumerate(peaks):
    if (peak-event_halfwidth)<0 or (peak+event_halfwidth) > len(data):
        continue
    datum = data[peak-event_halfwidth:peak+event_halfwidth].copy()
    
    max_stress = np.nanmax(datum["SHEAR STRESS"])
    min_stress = np.nanmin(datum["SHEAR STRESS"])
    stress_drop = max_stress-min_stress
    if i < 1:
        time_to_previous = (peak - peaks[i-1])/sampling_freq
    else:
        time_to_previous = np.nan
    if (i+1) < len(peaks):
        time_to_next = (peaks[i+1]-peak)/sampling_freq
    else:
        time_to_next = np.nan
    events_list.append({
        "peak_loc": peak,
        "peak_time": data[peak]["TIME"],
        "max_stress": max_stress,
        "min_stress": min_stress,
        "stress_drop": stress_drop,
        "time_to_previous": time_to_previous,
        "time_to_next": time_to_next
    })
    ax.plot(range(2*event_halfwidth), datum["SHEAR STRESS"])
plt.show()

df = pd.DataFrame(events_list)