import matplotlib.pyplot as plt
import numpy as np

def dimensionality_vs_time(data, results, params, save=False):
    fig, ax1 = plt.subplots()
    ax1.set_title("Dimension vs time")

    ax1.plot(results["t"], results["m"], label="m")
    ax1.plot(results["t"], results["KYD"], label="KYD")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Dimension")
    ax1.legend()

    ax2 = ax1.twinx()
    if "ROIs" in data.attrs:
        for ROI in data.attrs["ROIs"][::2]:
            ax2.axvspan(ROI[0], ROI[1], facecolor="whitesmoke")
        # for i in range(int(np.ceil(len(import_params["key_points"])/2))):
        #     xmin = import_params["key_points"][2*i]
        #     if len(import_params["key_points"]) >= (2*(i+1)):
        #         xmax = import_params["key_points"][(2*i+1)]
        #     else:
        #         xmax = data[-1].TIME
        #     ax2.axvspan(xmin, xmax, facecolor="whitesmoke")
    ax2.plot(data["TIME"], data["SHEAR STRESS"], color="0.93")

    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)

    if save:
        fig.savefig(params["results_dir"] + "/dim_v_time.png")

def tau_vs_time(data, results, params, save=False):
    fig, ax1 = plt.subplots()
    ax1.set_title("Tau delay vs time")

    ax1.plot(results["t"], results["tau"], label="tau")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Tau delay")
    ax1.legend()

    ax2 = ax1.twinx()
    if "ROIs" in data.attrs:
        for ROI in data.attrs["ROIs"][::2]:
            ax2.axvspan(ROI[0], ROI[1], facecolor="whitesmoke")
    ax2.plot(data["TIME"], data["SHEAR STRESS"], color="0.93")

    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)

    if save:
        fig.savefig(params["results_dir"] + "/tau_v_time.png")