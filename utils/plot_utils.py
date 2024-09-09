import matplotlib.pyplot as plt
import numpy as np
from utils.dynsys_utils import embed

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

def summary_calc_m_tau(data, tau_to_try, E1_thresh, E1s, E2s, eps_over_L, best_m, best_tau, save=False):
    fig,axs=plt.subplots(2, 2)
    fig.set_size_inches((11,9))

    axs[0,0].plot(data["TIME"], data["NORMALISED SHEAR"])
    axs[0,0].set_title("Time Series")
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("Shear Stress")

    for i, E1 in enumerate(E1s):
        color = "C" + str(i)
        axs[0,1].plot(E1, color=color, label=("Tau = " + str(tau_to_try[i])))
        axs[0,1].plot(E2s[i], color=color)
    axs[0,1].set_title("E1 & E2")
    axs[0,1].set_xlabel("Embedding dimension m")
    axs[0,1].set_ylabel("E value")
    axs[0,1].axhline(y=E1_thresh, color="lightgrey", label="E1 threshold")
    axs[0,1].legend()

    axs[1,0].scatter(tau_to_try, eps_over_L)
    axs[1,0].set_title("Eps over L")
    axs[1,0].set_xlabel("Tau delay")
    axs[1,0].set_ylabel("Eps over L")

    if ~np.isnan(eps_over_L).all() and best_m >= 3:
        axs[1,1].remove()
        axs[1,1] = fig.add_subplot(2,2,4,projection="3d")
        H, tH = embed(data["NORMALISED SHEAR"], tau=[best_tau], m=[best_m], t=data["TIME"])
        axs[1,1].plot(H[:,0], H[:,1], H[:,2])
        axs[1,1].set_title("First embedding")

    fig.suptitle("Window: m=" + str(best_m) + ", tau=" + str(best_tau))
    # if save:
    #     filename = "/snapshot_window_" + str(params["current_loop"]) + ".png" if "current_loop" in params else "/summary_calc_m_tau.png"
    #     fig.savefig(params["results_dir"] + filename)