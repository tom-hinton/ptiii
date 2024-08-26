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
        "Time": "TIME"
    }, inplace=True)


    # FILTER DATA
    if params["t_start"] and params["t_end"]:
        data = data[ data["TIME"].between(params["t_start"], params["t_end"]) ]
    data = data.reset_index(drop=True)
    data["TIME"] = data["TIME"]-data["TIME"][0]

    # RESAMPLE DATA
    if "resample_rate" in params:
        large_data = data
        num_rows = data.shape[0] // params["resample_rate"]
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

    # SAVE TO PICKLE
    with open(picklename, 'wb') as f:
        pickle.dump(data, f)

    # RETURN DATA
    return data

    