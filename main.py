from utils.load_utils import get_exp_meta

# EXPERIMENT META 
exp_code = "b1382"
desired_sampling_freq = 50
exp_meta = get_exp_meta(exp_code)

import os

# FILENAMES
working_dir = os.getcwd()
data_dir = working_dir + "/data"
datapath = data_dir + "/" + exp_code + ".txt"
picklepath = data_dir + '/' + exp_code + "_" + str(desired_sampling_freq) + "hz.pickle"


from utils.load_utils import import_brava_data
import matplotlib.pyplot as plt
import numpy as np

# IMPORT DATA
args = dict((k, exp_meta[k]) for k in ["t_start", "t_end", "ROIs", "LVDT_readjustments", "LVDT_method"] if k in exp_meta)
data = import_brava_data(datapath, picklepath, downsample_factor=(exp_meta["sampling_freq"] / desired_sampling_freq), **args)


# PLOT DATA
print("\n", data.head())
plt.close("all")
plt.plot(data["TIME"], data["ON BOARD LVDT"], color="lightgrey")
plt.plot(data["TIME"], data["LVDT NO JUMPS"])
plt.plot(data["TIME"], data["LVDT VELOCITY"])
plt.show()