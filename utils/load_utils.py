import os
import pickle
import pandas as pd
import numpy as np
from scipy.signal import resample
import json
from datetime import datetime

def import_brava_data(datapath, picklepath, downsample_factor=None, t_start=None, t_end=None, ROIs=None):
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

    # ADD METADATA
    data.attrs = { "ROIs": ROIs }

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
                "ROIs": [ (3790.5, 4614.3), (4614.3, 4853.6), (4853.6, 5677.6), (5677.6, 5922) ]
            }
        case "b1385":
            return { "t_start": 2320, "t_end": 12232, "sampling_freq": 1000 }
        case "b1378":
            return { "t_start": 2466, "t_end": 19570, "sampling_freq": 1000 }
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