import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA/Scripts/multilaterate')
# from math import sqrt, radians
# import math
# from joblib import Parallel, delayed
import multiprocessing


import sys
sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA')

import bella.multilaterate.bella_plotter as bplot
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import solarmap
# from scipy.stats import gaussian_kde
# from scipy.ndimage import median_filter
# from scipy.optimize import curve_fit
# import scipy.io
from scipy.ndimage import gaussian_filter

from astropy.constants import R_sun, e

# import bayes_positioner as bp

# import matplotlib
# import matplotlib.patheffects as PathEffects
# from matplotlib import cm
#
# from matplotlib.animation import FuncAnimation, writers
# from matplotlib.ticker import FormatStrFormatter,LogFormatter
#
# import os


# import astropy.units as u

# import datetime as dt


# from contextlib import contextmanager
# import sys, os
# import logging
# import csv
plt.rcParams.update({'font.size': 18})
plt.rcParams["font.family"] = "Times New Roman"



if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    string =f'Running on PyMC3 v{pm.__version__}'
    print(string)

    ncores_cpu = multiprocessing.cpu_count()

    showbgmap = True
    showtracked = True
    showscatter = False
    showparkerspirals = True
    savefigs = False

    # Spacecraft:
    windsc = True
    steasc = True
    stebsc = False
    solosc = True
    pspsc  = True
    mexsc  = True

    # spacecraft = ["stereo_a", "solo", "psp", "mex"]
    spacecraft = []
    spacecraft_labels = []
    if windsc:
        spacecraft.append("wind")
        spacecraft_labels.append("Wind")
    if steasc:
        spacecraft.append("stereo_a")
        spacecraft_labels.append("Stereo A")
    if stebsc:
        spacecraft.append("stereo_b")
        spacecraft_labels.append("Stereo B")
    if solosc:
        spacecraft.append("solo")
        spacecraft_labels.append("SolO")
    if pspsc:
        spacecraft.append("psp")
        spacecraft_labels.append("PSP")
    if mexsc:
        spacecraft.append("mex")
        spacecraft_labels.append("MEX")

    # generate some test data
    # np.random.seed(1)
    day = 4
    month = 12
    year = 2021
    date_str = f"{year}_{month:02d}_{day:02d}"

    sol_rot = 24.47#27.2753
    omega = 2*3.1416/(sol_rot*24*60*60)


    solarsystem = solarmap.get_HEE_coord(date=[year, month, day], objects=spacecraft)
    stations_rsun = np.array(solarsystem.locate_simple())

    N_STATIONS = 4#len(stations_rsun)
    stations = stations_rsun*R_sun.value

    # Making grid
    xrange = [-310,310]
    xres = 10
    yrange = [-310, 310]
    yres = xres
    xmapaxis = np.arange(xrange[0], xrange[1], xres)
    ymapaxis = np.arange(yrange[0], yrange[1], yres)
    cadence = 60 # for the filename
    bg_limits = [25, 80] # BG map histogram 0-1

    # FILENAMES
    # BELLA MAP
    filename_BELLA_MAP_4sc = "./Data/BG_Maps/2021_12_04/results_-310_310_-310_310_10_10_4stations_WIND_STEA_SOLO_PSP_60s.pkl"


    # TRACKED TYPE III
    burst_files = {
        "burst1": f"./Data/Multilaterated/{date_str}/TRACKING_{date_str}_results_4stations_WIND_STEA_SOLO_PSP_LE_Freqs_0.5_3_BURST_A.pkl",
        "burst2": f"./Data/Multilaterated/{date_str}/TRACKING_{date_str}_results_4stations_WIND_STEA_SOLO_PSP_LE_Freqs_0.4_10_BURST_B.pkl",
    }
    # LEGEND LABELS
    burst_labels = {
        "burst1": "BurstA",
        "burst2": "BurstB",
    }
    # # TRACKED TYPE III
    burst_markers = {
        "burst1": "o",
        "burst2": "^",
    }

    tracked_cmap = "turbo"
    trackedfile_scatter = ""


    # Figures
    fname = f"/bayes_positioner_map_median_tracked_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}_AvsB.jpg"


    # ################################################################################## #
    # ##########          PLOT 4 SPACECRAFT                     ######################## #
    # ################################################################################## #
    filename_BELLA_MAP = filename_BELLA_MAP_4sc

    """  ------------------------ Load BELLA MAP Data ----------------------------------   """
    if showbgmap:
        # xmapaxis, ymapaxis, delta_obs, stations_rsun = bplot.loaddata(filename_BELLA_MAP)
        xmapaxis, ymapaxis, delta_obs, _ = bplot.loaddata(filename_BELLA_MAP)
        """  ---------------------------------------------------------------------   """
        # """  ------------------------ BELLA MAP PLOT W/ MEDIAN FILTER ---------------------------   """
        # from scipy.ndimage import median_filter as medfil
        delta_obs2 = delta_obs*2
        median_filter_image = bplot.medfil(delta_obs2, size=(6,6))

    # Load and process data dynamically
    tracked_bursts = {}
    for burst_key, burst_file in burst_files.items():
        if burst_file:
            tracked_data = bplot.loadtrackedtypeiii(burst_file)
            tracked_bursts[burst_key] = {
                "data": bplot.typeIII(tracked_data[:, 0], tracked_data[:, 1], tracked_data[:, 2]),
                "times": tracked_data[:, 3],
                "label": burst_labels[burst_key],
                "markers": burst_markers[burst_key]
            }


    if showscatter:
        tracked_scatter = bplot.loadtrackedtypeiii(trackedfile_scatter)
        trackedtypeIII_scatter = bplot.typeIII(tracked_scatter[:,0], tracked_scatter[:,1],tracked_scatter[:,2])
        timestracked_scatter = tracked_scatter[:, 3]


    # BELLA PLOTTER
    # UNCERTAINTY MAP
    if showbgmap:
        delta_obs2 = delta_obs*2
        median_filter_image = bplot.medfil(delta_obs2, size=(6,6))

    fig, ax = plt.subplots(1,1,figsize=(11,11))
    plt.subplots_adjust(top=1, bottom=0)
    if showbgmap:
        fig, ax = bplot.plot_bella_map(fig,ax, median_filter_image, xmapaxis, ymapaxis, stations_rsun,
                                 vmin=bg_limits[0], vmax=bg_limits[1],
                                 date_str=date_str, spacecraft_labels=spacecraft_labels,
                                 objects =["earth_orbit", "earth", "sun", "spacecraft"])


        Z_smoothed = gaussian_filter(median_filter_image, sigma=1)
        CS = ax.contour(xmapaxis, ymapaxis,Z_smoothed.T,  colors='black')
        ax.clabel(CS, inline=True, fontsize=12)

    # TRACKED TYPE III
    if showtracked and tracked_bursts:
        # Plot each burst using a loop
        for burst_info in tracked_bursts.values():
            try:
                fig, ax = bplot.plot_tracked_typeIII(fig, ax, burst_info["data"], confidence=True, marker=burst_info["markers"], s=50, label=burst_info["label"], cmap=tracked_cmap)
            except Exception as e:
                print(f"Error plotting burst {burst_info['label']}: {e}")


    if showscatter:
        fig, ax = bplot.plot_tracked_typeIII(fig, ax, trackedtypeIII_scatter, confidence=True, showcolorbar=False)

    # PARKER SPIRALS
    # fig, ax = plot_parker_spiral(fig, ax, v_sw=310, phi_sw=30)
    if showparkerspirals:
        v_sw = 300
        pp = 90
        fig, ax = bplot.plot_parker_spiral(fig, ax, v_sw=v_sw+20, phi_sw=pp, omega=omega, color='black', linestyle="-.")
        fig, ax = bplot.plot_parker_spiral(fig, ax, v_sw=v_sw, phi_sw=pp, omega=omega, color='black', linestyle="-")
        fig, ax = bplot.plot_parker_spiral(fig, ax, v_sw=v_sw-20, phi_sw=pp, omega=omega, color='black', linestyle="-.")

        v_sw = 600
        pp = 90
        fig, ax = bplot.plot_parker_spiral(fig, ax, v_sw=v_sw + 20, phi_sw=pp, omega=omega, color='black', linestyle="-.")
        fig, ax = bplot.plot_parker_spiral(fig, ax, v_sw=v_sw, phi_sw=pp, omega=omega, color='black', linestyle="-")
        fig, ax = bplot.plot_parker_spiral(fig, ax, v_sw=v_sw - 20, phi_sw=pp, omega=omega, color='black', linestyle="-.")
    # REFRESH LEGEND
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True, ncol=2, fontsize=18)
    # data1_handle = legend.legend_handles[3]
    # data1_handle.set_sizes([300])
    # data1_handle = legend.legend_handles[4]
    # data1_handle.set_sizes([300])
    fig.subplots_adjust(left=0.098, bottom=0.080, right=0.9, top=0.845, wspace=0.2, hspace=0.2)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.show(block = False)

    if savefigs is True:
        dir = bplot.mkdirectory("./Figures/")
        plt.savefig(dir+'BELLA_map_4spacecraft.png', dpi=300)
