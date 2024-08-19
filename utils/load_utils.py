import os
import pickle
import pandas as pd

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

    # FILTER DATA
    if params["t_start"] and params["t_end"]:
        data = data[ data["Time"].between(params["t_start"], params["t_end"]) ]
    data = data.reset_index(drop=True)

    # SAVE TO PICKLE
    with open(picklename, 'wb') as f:
        pickle.dump(data, f)

    # RETURN DATA
    return data

    