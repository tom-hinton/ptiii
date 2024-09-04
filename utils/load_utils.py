import os
import pickle
import pandas as pd
import numpy as np
from scipy.signal import resample
import json
from datetime import datetime

def import_brava_data(params):
    print("Importing BRAVA data...")

    # TRY READING PICKLE
    if os.path.isfile(params["data_picklename"]):
        print('...reading pickle...')
        with open(params["data_picklename"], 'rb') as f:
            data = pickle.load(f)
        return data
        
    # READ DATA
    print('...reading data from file...')
    data = pd.read_table(params["data_filename"], skiprows=[1])

    # RENAME COLUMNS
    data.rename(columns={
        "V-LOAD": "SHEAR STRESS",
        "H-LOAD": "NORMAL STRESS",
        "Time": "TIME",
        " LVDT": "ON BOARD LVDT"
    }, inplace=True)

    # UNIT CONVERSION
    data["SHEAR STRESS"] = (data["SHEAR STRESS"]+0.044) / 5
    data["NORMAL STRESS"] *= 0.2
    data["ON BOARD LVDT"] *= 1546
    data["V-LVDT"] *= -1

    # CLIP DATA
    if "t_start" in params["exp_meta"] and "t_end" in params["exp_meta"]:
        data = data[ data["TIME"].between(params["exp_meta"]["t_start"], params["exp_meta"]["t_end"]) ]
    data = data.reset_index(drop=True)
    # data["TIME"] = data["TIME"]-data["TIME"][0]

    # RESAMPLE DATA
    if "downsample_factor" in params:
        large_data = data
        num_rows = data.shape[0] // params["downsample_factor"]
        data = pd.DataFrame(columns=data.columns)
        for column_name, column in large_data.items():
            if column_name == "TIME":
                data["TIME"] = np.linspace(column.iloc[0], column.iloc[-1], num_rows)
            else:
                print("Resampling column: " + column_name)
                data[column_name] = resample(column, num_rows)

    # ADD NORMALISED V LOAD COLUMN
    p = np.polyfit(data["TIME"], data["SHEAR STRESS"], deg=1)
    X = data["SHEAR STRESS"] - (p[0]*data["TIME"] + p[1])
    data["NORMALISED SHEAR"]  = (X - np.min(X)) / (np.max(X) - np.min(X))

    # ADD METADATA
    data.attrs = params["exp_meta"]

    # SAVE TO PICKLE
    with open(params["data_picklename"], 'wb') as f:
        pickle.dump(data, f)

    # RETURN DATA
    return data

def get_exp_meta(params):
    match params["exp_code"]:
        case "b1383":
            return {
                "t_start": 1940, "t_end": 5922,
                "ROIs": [ (3790.5, 4614.3), (4614.3, 4853.6), (4853.6, 5677.6), (5677.6, 5922) ]
            }
        case _:
            return {}
        
def normalise_shear(data):
    p = np.polyfit(data["TIME"], data["SHEAR STRESS"], deg=1)
    X = data["SHEAR STRESS"] - (p[0]*data["TIME"] + p[1])
    data["NORMALISED SHEAR"]  = (X - np.min(X)) / (np.max(X) - np.min(X))
    return data

class NumpyEncoder(json.JSONEncoder):
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