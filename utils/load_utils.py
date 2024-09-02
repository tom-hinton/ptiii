import os
import pickle
import pandas as pd
import numpy as np
from scipy.signal import resample

def import_brava_data(filename, picklename, params={}):
    print("Importing BRAVA data...")

    # TRY READING PICKLE
    if os.path.isfile(picklename):
        print('...reading pickle...')
        with open(picklename, 'rb') as f:
            data = pickle.load(f)
        return data
        
    # READ DATA
    print('...reading data from file...')
    data = pd.read_table(filename, skiprows=[1])

    # RENAME COLUMNS
    data.rename(columns={
        "V-LOAD": "SHEAR STRESS",
        "H-LOAD": "NORMAL STRESS",
        "Time": "TIME",
        " LVDT": "ON BOARD LVDT"
    }, inplace=True)

    # CLIP DATA
    if "t_start" in params and "t_end" in params:
        data = data[ data["TIME"].between(params["t_start"], params["t_end"]) ]
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

    # UNIT CONVERSION
    data["SHEAR STRESS"] = (data["SHEAR STRESS"]+0.044) / 5
    data["NORMAL STRESS"] *= 0.2
    data["ON BOARD LVDT"] *= 1546
    data["V-LVDT"] *= -1

    # SAVE TO PICKLE
    with open(picklename, 'wb') as f:
        pickle.dump(data, f)

    # RETURN DATA
    return data

def get_import_params(exp_name):
    match exp_name:
        case "b1383":
            return { "t_start": 1940, "t_end": 5922 }
        case _:
            return {}