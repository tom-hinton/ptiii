import os
import pickle
import pandas as pd
import numpy as np
from scipy.signal import resample, convolve, find_peaks
from findiff import FinDiff
import json
from datetime import datetime

def import_brava_data(datapath, picklepath, downsample_factor=None, t_start=None, t_end=None, ROIs=None, LVDT_readjustments=None, LVDT_method=None):
    print("Importing BRAVA data...")

    # TRY READING PICKLE
    if os.path.isfile(picklepath):
        print('...reading pickle...')
        with open(picklepath, 'rb') as f:
            data = pickle.load(f)
        print("... data imported")
        return data
        
    # READ DATA
    print('...reading data from file...')
    data = pd.read_table(datapath, skiprows=[1])

    # RENAME COLUMNS
    data.rename(columns={
        "V-LOAD": "SHEAR STRESS",
        "H-LOAD": "NORMAL STRESS",
        "Time": "TIME",
        " LVDT": "ON BOARD LVDT"
    }, inplace=True)

    # DELETE COLUMNS
    data = data.drop(columns="Record")

    # UNIT CONVERSION
    data["SHEAR STRESS"] = (data["SHEAR STRESS"]+0.044) / 5
    data["NORMAL STRESS"] *= 0.2
    data["ON BOARD LVDT"] *= 1546
    data["V-LVDT"] *= -1

    # CLIP DATA
    if t_start is not None and t_end is not None:
        data = data[ data["TIME"].between(t_start, t_end) ]
    data = data.reset_index(drop=True)

    # RESAMPLE DATA
    if downsample_factor is not None:
        print("...resampling...", end=" ")
        large_data = data
        num_rows = int(data.shape[0] // downsample_factor)
        data = pd.DataFrame(columns=data.columns)
        for column_name, column in large_data.items():
            if column_name == "TIME":
                data["TIME"] = np.linspace(column.iloc[0], column.iloc[-1], num_rows)
            else:
                data[column_name] = resample(column, num_rows)
        data = data.iloc[1:] # avoid resampling issue of first item
        print("resampled...")

    # ADD NORMALISED V LOAD COLUMN
    p = np.polyfit(data["TIME"], data["SHEAR STRESS"], deg=1)
    X = data["SHEAR STRESS"] - (p[0]*data["TIME"] + p[1])
    data["NORMALISED SHEAR"]  = (X - np.min(X)) / (np.max(X) - np.min(X))
    
    # IDENTIFY LVDT JUMPS
    data["LVDT NO JUMPS"] = data.loc[:,"ON BOARD LVDT"]
    freq = int(1/(data["TIME"][2] - data["TIME"][1]))
    jumps = []
    if LVDT_method == "sta/lta":
        print("...finding LVDT peaks...")
        rolling_window_length = 5*freq
        rolling_mean = convolve(data["LVDT NO JUMPS"], np.ones(rolling_window_length)/rolling_window_length, "same")
        lvdt_detrended = data["LVDT NO JUMPS"] - rolling_mean
        peaks, _ = find_peaks(lvdt_detrended, height=30, distance=4*freq)
        jumps = [(int(p-2*freq), int(p+freq)) for p in peaks]
    elif LVDT_method == "clusters":
        print("...finding LVDT peaks...")
        peaks, _ = find_peaks(data["LVDT NO JUMPS"], width=(0.019*freq, 0.25*freq), prominence=1)
        troughs, _ = find_peaks(-data["LVDT NO JUMPS"], width=(0.019*freq, 0.25*freq), prominence=1)
        peaks = np.sort(np.concatenate((peaks, troughs)))
        i = 0
        while i < (len(peaks) - 1):
            peak = peaks[i]
            if (peaks[i+1] - peak) < freq:
                cluster = [p for p in peaks if peak <= p <= (peak+2*freq) ]
                jumps.append((int(cluster[0]-freq), int(cluster[-1]+0.5*freq)))
                i = i + len(cluster)
            i += 1
    elif LVDT_method == "negativevelocity":
        dt = data["TIME"][2] - data["TIME"][1]
        d_dt = FinDiff(0, dt)
        vel = d_dt(data["LVDT NO JUMPS"])
        peaks, _ = find_peaks(vel, height=200)
        i = 0
        while i < len(peaks):
            cluster = [p for p in peaks if peaks[i] <= p <= (peaks[i]+3*freq) ]
            jumps.append((int(cluster[0]-freq), int(cluster[-1]+0.5*freq)))
            i += len(cluster)
    elif LVDT_method == "negativevelocitysensitive":
        dt = data["TIME"][2] - data["TIME"][1]
        d_dt = FinDiff(0, dt)
        vel = d_dt(data["LVDT NO JUMPS"])
        peaks, _ = find_peaks(vel, height=50)
        i = 0
        while i < len(peaks):
            cluster = [p for p in peaks if peaks[i] <= p <= (peaks[i]+3*freq) ]
            jumps.append((int(cluster[0]-freq), int(cluster[-1]+0.5*freq)))
            i += len(cluster)


    # REMOVE LVDT READJUSTMENTS
    if LVDT_readjustments is not None:
        for readj in LVDT_readjustments:
            data.loc[data["TIME"].between(readj[0], readj[1]), "LVDT NO JUMPS"] = np.nan
    
    # REMOVE LVDT JUMPS
    for jump in jumps:
        if(jump[0] < 0):
            data.loc[0:jump[1], "LVDT NO JUMPS"] = np.nan
        else:
            data.loc[jump[0]:jump[1], "LVDT NO JUMPS"] = np.nan
    
    # COMPUTE LVDT VELOCITY
    dt = data["TIME"][2] - data["TIME"][1]
    d_dt = FinDiff(0, dt)
    data["LVDT VELOCITY"] = -d_dt(data["LVDT NO JUMPS"])

    # SAVE TO PICKLE
    with open(picklepath, 'wb') as f:
        pickle.dump(data, f)

    # RETURN DATA
    print("... data imported")
    return data

def get_exp_meta(exp_code):
    match exp_code:
        case "b1383":
            return {
                "t_start": 1940, "t_end": 5922, "sampling_freq": 1000,
                "ROIs": [ (3790.5, 4614.3, "LPV=3µm/s"), (4614.3, 4853.6, "LPV=10µm/s"), (4853.6, 5677.6, "LPV=3µm/s"), (5677.6, 5922, "LPV=10µm/s") ],
                "LVDT_readjustments": [(1940,2900), (3675,3705), (4648.7,4649.7), (5555,5570), (5590,5592)], "LVDT_method": "negativevelocity"
            }
        case "b1385":
            return {
                "t_start": 2320, "t_end": 12247,
                "sampling_freq": 1000,
                "ROIs": [ (9758, 12247, "PID=13"), (12247, 14735, "PID=8"), (14735, 17219, "PID=4"), (17219, 19710, "PID=2") ],
                "LVDT_readjustments": [(2320,2326), (11803,11810)], "LVDT_method": "negativevelocity"
            }
        case "b1378":
            return { "t_start": 2466, "t_end": 19570, "sampling_freq": 1000, "LVDT_method": "negativevelocitysensitive" }
        case "b1380":
            return {
                "t_start": 2492, "t_end": 23895, "sampling_freq": 1000,
                "ROIs": [ (2492., 10710.5, "LPV=0.3µm/s"), (10710.5, 13182., "LPV=1.0µm/s"), (13182., 21432., "LPV=0.3µm/s"), (21432., 23895., "LPV=1.0µm/s") ],
                "LVDT_readjustments": [(2492, 2972), (14836, 14839), (23894,23895)], "LVDT_method": "negativevelocitysensitive"
            }
        case "b1384":
            return {
                "t_start": 3462.6, "t_end": 4954, "sampling_freq": 1000,
                "ROIs": [ (3461.6, 3710.8, "PID=20"), (3710.8, 3959.6, "PID=16"), (3959.6, 4208.3, "PID=12"), (4208.3, 4457.9, "PID=8"), (4457.9, 4706.35, "PID=4"), (4706.35, 4954, "PID=2") ],
                "LVDT_method": "negativevelocity"
            }
        case "b1382":
            return { "t_start": 1895, "t_end": 4346, "sampling_freq": 1000, "LVDT_readjustments": [(2700, 3312), (3968, 3972)], "LVDT_method": "negativevelocity" }
        case _:
            return { "sampling_freq": 1000 }
        
def normalise_shear(data):
    p = np.polyfit(data["TIME"], data["SHEAR STRESS"], deg=1)
    X = data["SHEAR STRESS"] - (p[0]*data["TIME"] + p[1])
    data["NORMALISED SHEAR"]  = (X - np.min(X)) / (np.max(X) - np.min(X))
    return data

class JsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.strftime("%c")
        return json.JSONEncoder.default(self, obj)