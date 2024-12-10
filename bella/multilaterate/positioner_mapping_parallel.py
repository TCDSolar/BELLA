import os
import math
import logging
import argparse
import datetime as dt
import multiprocessing
from math import radians
from contextlib import contextmanager

import arviz as az
import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.io
import solarmap
from joblib import Parallel, delayed
from matplotlib import cm
from scipy import interpolate
from scipy.ndimage import median_filter
from termcolor import colored

from astropy.constants import R_sun, au, c

# caution: path[0] is reserved for script path (or '' in REPL)
# IF running from a different directory, point to the multilaterate directory with bayes_positioner.py file
#sys.path.insert(1, 'PATH/TO/Multilateratefolder')

import sys
sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA')

import bella.multilaterate.bayes_positioner as bp


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def mkdirectory(directory):
    dir = directory
    isExist = os.path.exists(dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir)
        print("The new directory is created!")

def parallel_pos_map(x1,y1,stations,xrange,yrange,xres,yres,t_cadence=60, cores=1, traceplotsave=False, figdir="./traceplots", date_str="date"):
    tloop0 = dt.datetime.now()
    try:
        mesh = {}
        k=0
        xmapaxis = np.arange(xrange[0], xrange[1], xres)
        ymapaxis = np.arange(yrange[0], yrange[1], yres)
        for i in xmapaxis:
            for j in ymapaxis:
                mesh[i,j]=k
                k += 1


        currentloop = f"x, y : {x1},{y1}   ||  loop {mesh[x1,y1]}/{len(mesh)}"
        print(currentloop)
        logginfo = f"{dt.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} : START: {currentloop}"
        logging.info(logginfo)

        x_true = np.array([x1 * R_sun.value, y1 * R_sun.value])  # true source position (m)
        v_true = c.value  # speed of light (m/s)
        d_true = np.linalg.norm(stations - x_true, axis=1)
        t1_true = d_true / v_true  # true time of flight values
        # t_obs = t1_true-t0_true# true time difference of arrival values
        t_obs = t1_true
        # np.random.seed(1)
        # t_obs = t_obs+0.05*np.random.randn(*t_obs.shape)# noisy observations

        # Make sure to use cores=1 when using parallel loops. cores in triangulate function refers to number of cores
        # used by the MCMC solver.
        results_out = bp.triangulate(stations, t_obs, t_cadence=t_cadence, cores=1, progressbar=False, report=0, plot=0)

        mu = results_out[0]
        sd = results_out[1]
        results_out[2]
        trace = results_out[3]
        results_out[4]
        results_out[5]
        results_out[6]





        # report
        # print(summary)
        # print(f"t0_true: {t0_true}")
        # print(f"t_obs:    {t_obs}")
        # print(f"t1_true: {t1_true}")
        # print(f"t1_pred: {t1_pred}")
        # print(f"stations: {stations}")

        if traceplotsave is True:
            traceplotpath = f"./traceplots/{date_str}/traceplot_{xrange[0]}_{xrange[1]}_{yrange[0]}_{yrange[1]}_{xres}_{yres}"
            mkdirectory(traceplotpath)

            # trace plot
            ax, ay = az.plot_trace(trace, compact=False)[1:3, 0]
            ax.hlines(0.6065 * ax.get_ylim()[1], mu[0] - sd[0], mu[0] + sd[0])  # add mu, sigma lines to x,y plots
            ay.hlines(0.6065 * ay.get_ylim()[1], mu[1] - sd[1], mu[1] + sd[1])

            figname = traceplotpath + f"/bayes_positioner_traceplot_{x1}_{y1}.jpg"
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.01, dpi=100)
            # plt.close()
            # pm.autocorrplot(trace)
            # pm.plot_posterior(trace)

        # difference detected observed
        x_true = x_true / R_sun.value
        mu[0] / R_sun.value, mu[1] / R_sun.value

        #delta_obs = sqrt((xy[0] - x_true[0]) ** 2 + (xy[1] - x_true[1]) ** 2)           # Deltaobs = xy - x_true Deprecated
        delta_obs =  np.amax([sd[0]/R_sun.value , sd[1]/R_sun.value])               # Bayesian uncertainty. Pick the largest one.
        print(f"delta_obs: {delta_obs}")
        res = np.array([delta_obs, np.nan, np.nan])
        logginfo = f"{dt.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} : END  {currentloop}"
        logging.info(logginfo)
        tloop1 = dt.datetime.now()
        print(f" P({x1},{y1}),  {colored('SUCCESS', 'green')}:  {res}")


    except Exception as e:
        delta_obs = 600
        print(f"SIM ERROR: {e}")
        print(f"SIM {colored('FAILED', 'red')} at P({x1},{y1}), SKIPPED")
        res = np.array([delta_obs, x1, y1])
        tloop1 = dt.datetime.now()
        pass

    tloopinseconds = (tloop1 - tloop0).total_seconds()
    print(f"Time Loop : {tloop1 - tloop0}   :   {tloopinseconds}s  \n  loop {mesh[x1,y1]}/{len(mesh)} DONE ")

    estimated_t_left = tloopinseconds*(len(mesh) - mesh[x1,y1])/(multiprocessing.cpu_count())
    estimated_t_left_timedelta = dt.timedelta(seconds=estimated_t_left)
    print(f"{colored('Estimated time left', 'yellow')}: {estimated_t_left_timedelta} ")

    return res

def plot_map_simple(delta_obs, xmapaxis, ymapaxis, stations,vmin=0,vmax=30, savefigure=False, showfigure=True, title="",figdir="./MapFigures", date_str="date", filename="fig.jpg"):
    xmapaxis[1]-xmapaxis[0]
    ymapaxis[1]-ymapaxis[0]
    len(stations)

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    plt.subplots_adjust(top=1, bottom=0)

    im_0 = ax.pcolormesh(xmapaxis, ymapaxis, delta_obs.T, cmap='jet', vmin=vmin, vmax=vmax)

    earth_orbit = plt.Circle((0, 0), au/R_sun, color='k', linestyle="dashed", fill=None)
    ax.add_patch(earth_orbit)

    ax.scatter(stations[:,0], stations[:,1],color = "w", marker="^",edgecolors="k", s=80, label="Spacecraft")

    ax.set_aspect('equal')

    ax.set_xlim(xmapaxis[0], xmapaxis[-1])
    ax.set_ylim(ymapaxis[0], ymapaxis[-1])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.01, 0.5])
    fig.colorbar(im_0, cax=cbar_ax)
    cbar_ax.set_ylabel('Triangulation uncertainty (Rsun)', fontsize=22)

    ax.plot(au / R_sun, 0, 'bo', label="Earth")
    ax.plot(0, 0, 'yo', label="Sun")

    ax.legend(loc=1)

    ax.set_xlabel(r"'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel(r"'HEE - Y / $R_{\odot}$'", fontsize=22)
    ax.set_title(title, fontsize=22)

    if savefigure is True:
        figdir = f"{figdir}/{date_str}"
        mkdirectory(figdir)
        plt.savefig(figdir+filename, bbox_inches='tight', pad_inches=0.01, dpi=300)

    if showfigure is True:
        plt.show(block=False)
    else:
        plt.close(fig)

def pol2cart(r, phi):
    x = r* np.cos(phi)
    y = r* np.sin(phi)
    return(x, y)

def cart2pol(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    return(r,theta)

def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return (r, theta)

def plot_map_simple_withTracked(delta_obs, trackedtypeIII, xmapaxis, ymapaxis, stations,
                                vmin=0,vmax=30,parker_spiral=False, v_sw=400,phi_sw=0, dphi=0,
                                savefigure=False, showfigure=True,spacecraft=[], confidence=False,
                                title="",figdir="./MapFigures", date_str="date", filename="fig_Tracked.jpg"):


    xy_vals = np.array(list(trackedtypeIII.xy))
    xy = xy_vals/R_sun.value
    tracked_freqs = trackedtypeIII.freq
    sd_vals = np.array(list(trackedtypeIII.sd))
    sd = sd_vals/R_sun.value


    xmapaxis[1]-xmapaxis[0]
    ymapaxis[1]-ymapaxis[0]
    len(stations)


    # PARKER SPIRAL
    phi0 = phi_sw
    parkerphi = []
    parkerphi_plus = []
    parkerphi_minus = []
    parkerphi120 = []
    parkerphi240 = []
    parkerend = 600
    for r in range(0, parkerend):
        parkerphi.append(parkerSpiral(r, phi0, v_sw=v_sw))
        parkerphi_plus.append(parkerSpiral(r, phi0+dphi, v_sw=v_sw))
        parkerphi_minus.append(parkerSpiral(r, phi0-dphi, v_sw=v_sw))
        parkerphi120.append(parkerSpiral(r, phi0+120, v_sw=v_sw))
        parkerphi240.append(parkerSpiral(r, phi0+240, v_sw=v_sw))
    x_parker, y_parker = pol2cart(np.arange(0,parkerend),parkerphi)
    x_parker_plus, y_parker_plus = pol2cart(np.arange(0,parkerend),parkerphi_plus)
    x_parker_minus, y_parker_minus = pol2cart(np.arange(0,parkerend),parkerphi_minus)

    x_parker120, y_parker120 = pol2cart(np.arange(0,parkerend),parkerphi120)
    x_parker240, y_parker240 = pol2cart(np.arange(0,parkerend),parkerphi240)


    fig, ax = plt.subplots(1,1,figsize=(11,11))
    plt.subplots_adjust(top=1, bottom=0)

    im_0 = ax.pcolormesh(xmapaxis, ymapaxis, delta_obs.T, cmap='plasma', shading='gouraud', vmin=vmin, vmax=vmax)

    # Uncomment to see where simulation failed.
    # im_fail = ax.pcolormesh(xmapaxis, ymapaxis, np.ma.masked_values(delta_obs,200).T, cmap='Greys', vmin=vmin, vmax=vmax)

    earth_orbit = plt.Circle((0, 0), au/R_sun + 5, color='k', linestyle="dashed", fill=None)
    ax.add_patch(earth_orbit)

    ax.scatter(stations[:,0], stations[:,1],color = "w", marker="^",edgecolors="k", s=180, label="Spacecraft")

    i = 0
    for sc in spacecraft:
        sclab = ax.text(stations[i, 0], stations[i, 1], sc, color="w", fontsize=22)
        sclab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
        i+=1


    if parker_spiral is True:
        ax.plot(x_parker,y_parker,"k--")
        ax.plot(x_parker_plus,y_parker_plus,"k--", markersize=0.5)
        ax.plot(x_parker_minus,y_parker_minus,"k--")
        # ax.plot(x_parker120, y_parker120,"k--")
        # ax.plot(x_parker240, y_parker240,"k--")

    mat = scipy.io.loadmat('Data/2012_06_07/SEMP_spiral_results.mat')
    x_zhang = mat["x_ParkerSpiral_Cart"]
    y_zhang = mat["y_ParkerSpiral_Cart"]
    ax.plot(x_zhang[0], y_zhang[0], "r--",label="Zhang et al. 2019")



    # # ELLIPSES
    colors = cm.turbo(list(np.linspace(0,1.0,len(xy))))
    if confidence is True:
        i = 0
        ell_track_uncertainty = matplotlib.patches.Ellipse(xy=(xy[i, 0], xy[i, 1]),
                                                           width=2 * sd[i, 0], height=2 * sd[i, 1],
                                                           angle=0., edgecolor=colors[i], lw=1.5)
        for i in range(1, len(xy)):
            ell_track_uncertainty =matplotlib.patches.Ellipse(xy=(xy[i,0], xy[i,1]),
                      width=4*sd[i,0], height=4*sd[i,1],
                      angle=0.,edgecolor=colors[i], lw=1.5)

            ell_track_uncertainty.set_facecolor('none')


            ax.add_patch(ell_track_uncertainty)

    im_track = ax.scatter(xy[:,0], xy[:,1],c = tracked_freqs, cmap="turbo", marker=".",edgecolors="w", s=100, label="TrackedBeam", norm=matplotlib.colors.LogNorm(),zorder=1000)

    # psplab = ax.text(stations[0,0], stations[0,1], "PSP", color="w",fontsize=22)
    # psplab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
    # sololab = ax.text(stations[1,0], stations[1,1], "SoLO", color="w",fontsize=22)
    # sololab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
    # stelab = ax.text(stations[2,0], stations[2,1], "StereoA", color="w",fontsize=22)
    # stelab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
    # windlab = ax.text(stations[3,0], stations[3,1], "WIND", color="w",fontsize=22)
    # windlab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])


    ax.set_aspect('equal')

    ax.set_xlim(xmapaxis[0], xmapaxis[-1])
    ax.set_ylim(ymapaxis[0], ymapaxis[-1])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.30])
    fig.colorbar(im_0, cax=cbar_ax)
    cbar_ax.set_ylabel('Triangulation uncertainty (Rsun)', fontsize=18)

    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.30])
    fig.colorbar(im_track, cax=cbar_ax2)
    cbar_ax2.set_ylabel('Tracked beam freq (MHz)', fontsize=18)

    ax.plot(au / R_sun+5, 0, 'bo', label="Earth", markersize=10)
    ax.plot(0, 0, 'yo', label="Sun", markersize=10, markeredgecolor ='k')


    # ax.plot(x_spiral,y_spiral, "k-", label="Zhang et al. 2019 ")


    ax.legend(loc=1)

    ax.set_xlabel(r"'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel(r"'HEE - Y / $R_{\odot}$'", fontsize=22)
    ax.set_title(title, fontsize=22)

    if savefigure is True:
        figdir = f"{figdir}/{date_str}"
        mkdirectory(figdir)
        plt.savefig(figdir+filename, bbox_inches='tight', pad_inches=0.01, dpi=300)

    if showfigure is True:
        plt.show(block=False)
    else:
        plt.close(fig)

    return fig,ax

def plot_map_simple_withTracked_zoom(delta_obs, trackedtypeIII, xmapaxis, ymapaxis, stations,
                                vmin=0,vmax=30,v_sw=400,phi_sw=0, dphi=0,
                                savefigure=False, showfigure=True,
                                title="",figdir="./MapFigures", date_str="date", filename="fig_Tracked.jpg"):

    xy_vals = np.array(list(trackedtypeIII.xy))
    xy = xy_vals/R_sun.value
    tracked_freqs = trackedtypeIII.freq
    sd_vals = np.array(list(trackedtypeIII.sd))
    sd_vals/R_sun.value


    xmapaxis[1]-xmapaxis[0]
    ymapaxis[1]-ymapaxis[0]
    len(stations)


    # PARKER SPIRAL
    phi0 = phi_sw
    parkerphi = []
    parkerphi_plus = []
    parkerphi_minus = []
    parkerphi120 = []
    parkerphi240 = []
    parkerend = 600
    for r in range(0, parkerend):
        parkerphi.append(parkerSpiral(r, phi0, v_sw=v_sw))
        parkerphi_plus.append(parkerSpiral(r, phi0+dphi, v_sw=v_sw))
        parkerphi_minus.append(parkerSpiral(r, phi0-dphi, v_sw=v_sw))
        parkerphi120.append(parkerSpiral(r, phi0+120, v_sw=v_sw))
        parkerphi240.append(parkerSpiral(r, phi0+240, v_sw=v_sw))
    x_parker, y_parker = pol2cart(np.arange(0,parkerend),parkerphi)
    x_parker_plus, y_parker_plus = pol2cart(np.arange(0,parkerend),parkerphi_plus)
    x_parker_minus, y_parker_minus = pol2cart(np.arange(0,parkerend),parkerphi_minus)

    x_parker120, y_parker120 = pol2cart(np.arange(0,parkerend),parkerphi120)
    x_parker240, y_parker240 = pol2cart(np.arange(0,parkerend),parkerphi240)

    # zhangresults = np.load('spiral.npz')
    # x_spiral = zhangresults['x_spiral']
    # y_spiral = zhangresults['y_spiral']
    #

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    plt.subplots_adjust(top=1, bottom=0)

    im_0 = ax.pcolormesh(xmapaxis, ymapaxis, delta_obs.T, cmap='jet', vmin=vmin, vmax=vmax)

    # Uncomment to see where simulation failed.
    # im_fail = ax.pcolormesh(xmapaxis, ymapaxis, np.ma.masked_values(delta_obs,200).T, cmap='Greys', vmin=vmin, vmax=vmax)

    earth_orbit = plt.Circle((0, 0), au/R_sun + 5, color='k', linestyle="dashed", fill=None)
    ax.add_patch(earth_orbit)

    ax.scatter(stations[:,0], stations[:,1],color = "w", marker="^",edgecolors="k", s=180, label="Spacecraft")
    psplab = ax.text(stations[0,0], stations[0,1], "PSP", color="w",fontsize=22)
    psplab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
    sololab = ax.text(stations[1,0], stations[1,1], "SoLO", color="w",fontsize=22)
    sololab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
    stelab = ax.text(stations[2,0], stations[2,1], "StereoA", color="w",fontsize=22)
    stelab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
    windlab = ax.text(stations[3,0], stations[3,1], "WIND", color="w",fontsize=22)
    windlab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])

    im_track = ax.scatter(xy[:,0], xy[:,1],c = tracked_freqs, marker="s",edgecolors="w",cmap='plasma', s=250, label="TrackedBeam")

    # ell_track_uncertainty = matplotlib.patches.Ellipse(xy=(xy[i, 1], xy[i, 0]),
    #                                                    width=4 * sd[i, 1], height=4 * sd[i, 0],
    #                                                    angle=0., edgecolor=tracked_freqs, cmap='plasma', label="Posterior ($2\sigma$)", lw=1.5)
    # for i in range(1, len(xy)):
    #     ell_track_uncertainty =matplotlib.patches.Ellipse(xy=(xy[i,1], xy[i,0]),
    #               width=4*sd[i,1], height=4*sd[i,0],
    #               angle=0., edgecolor="k", lw=1.5)
    #
    #     ell_track_uncertainty.set_facecolor('none')
    #
    #
    #     ax.add_patch(ell_track_uncertainty)

    ax.set_aspect('equal')

    ax.set_xlim(xmapaxis[0], xmapaxis[-1])
    ax.set_ylim(ymapaxis[0], ymapaxis[-1])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.30])
    fig.colorbar(im_0, cax=cbar_ax)
    cbar_ax.set_ylabel('Triangulation uncertainty (Rsun)', fontsize=18)

    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.30])
    fig.colorbar(im_track, cax=cbar_ax2)
    cbar_ax2.set_ylabel('Tracked beam freq (MHz)', fontsize=18)

    ax.plot(au / R_sun+5, 0, 'bo', label="Earth", markersize=10)
    ax.plot(x_parker,y_parker,"k--")
    ax.plot(x_parker_plus,y_parker_plus, "k--", markersize=0.5)
    ax.plot(x_parker_minus,y_parker_minus,"k--")
    # ax.plot(x_parker120, y_parker120,"k--")
    # ax.plot(x_parker240, y_parker240,"k--")
    ax.plot(0, 0, 'yo', label="Sun", markersize=10)


    # ax.plot(x_spiral,y_spiral, "k-", label="Zhang et al. 2019 ")

    ax.legend(loc=1)

    ax.set_xlabel(r"'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel(r"'HEE - Y / $R_{\odot}$'", fontsize=22)
    ax.set_title(title, fontsize=22)

    axins = ax.inset_axes([0.01, 0.01, 0.5, 0.5])
    axins.pcolormesh(xmapaxis, ymapaxis, delta_obs.T, cmap='jet', vmin=vmin, vmax=vmax)
    axins.scatter(xy[:,0], xy[:,1],c = tracked_freqs, marker="s",edgecolors="w",cmap='plasma', s=250)
    axins.plot(x_parker,y_parker,"k--")
    axins.plot(x_parker_plus,y_parker_plus, "k--", markersize=0.5)
    axins.plot(x_parker_minus,y_parker_minus,"k--")
    axins.set_aspect('equal')
    # sub region of the original image

    x1, y1, x2, y2 = xy[:,0].min()-3, xy[:,1].max()+3, xy[:,0].max()+3, xy[:,1].min()-3
    # x1,y1, x2, y2 = -32, 72, -8, 50
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.invert_yaxis()
    ax.indicate_inset_zoom(axins, edgecolor="black")


    if savefigure is True:
        figdir = f"{figdir}/{date_str}"
        mkdirectory(figdir)
        plt.savefig(figdir+filename, bbox_inches='tight', pad_inches=0.01, dpi=300)

    if showfigure is True:
        plt.show(block=False)
    else:
        plt.close(fig)


def savedata(delta_obs, xmapaxis, ymapaxis, stations, filename="output.pkl"):
    import pickle

    # Split the path into directory and filename
    directory, _ = os.path.split(filename)

    mkdirectory(directory)


    xmapaxis[1]-xmapaxis[0]
    ymapaxis[1]-ymapaxis[0]
    len(stations)

    results  = [xmapaxis, ymapaxis, delta_obs, stations]
    with open(filename, 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()
    print(f"saved data : {filename}")

def loaddata(filenamef):
    import pickle

    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)

    xmapaxis = results[0]
    ymapaxis = results[1]
    delta_obs = results[2]
    stations = results[3]
    inp.close()
    print(f"data loaded : {filenamef}")

    return xmapaxis, ymapaxis, delta_obs, stations

def savepickle(results, filename):
    import pickle
    with open(filename, 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()
    print(f"saved data : {filename}")


def loadtrackedtypeiii(filenamef):
    import pickle
    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)

    inp.close()
    return results


def interpolate_map(delta_obs, xmapaxis, ymapaxis, scale_factor=10, kind="linear"):


    f = interpolate.interp2d(xmapaxis, ymapaxis, delta_obs.T, kind=kind)

    xnew = np.linspace(xmapaxis[0], xmapaxis[-1], xmapaxis.shape[0] * scale_factor)
    ynew = np.linspace(ymapaxis[0], ymapaxis[-1], ymapaxis.shape[0] * scale_factor)
    znew = f(xnew, ynew)

    return xnew,ynew, znew.T

def simulation_report(title="",xrange=[], yrange=[], xres=0, yres=0, pixels=0, coresused=0, tstart=dt.datetime.now(), tfinal=dt.datetime.now(),writelog=True):
    SIMREPORT = f"""
    -------------------REPORT---------------------
    {title}
    Grid: X{xrange}, Y{yrange}
    xres: {xres}
    yres: {yres}
    totalpixels: {pixels}


    cores: {coresused}
    Computational cost: {tfinal - tstart}

    ----------------END-REPORT---------------------
    """
    logging.info(SIMREPORT)
    print(SIMREPORT)

def medfil(*args, **kwargs):
    new_image = median_filter(*args, **kwargs)
    return new_image

def parkerSpiral(r,phi0,v_sw=400, omega=2.662e-6, theta=0):
    # http://www.physics.usyd.edu.au/~cairns/teaching/2010/lecture8_2010.pdf page 6
    # r-r0 = -(v_sw/(omega*sin(theta)))(phi(r)*phi0)
    phi0 = np.radians(phi0)
    theta = np.radians(theta+90)
    r_sun2km = 695700     #
    r = r * r_sun2km
    b=v_sw/(omega*np.sin(theta))
    r0= 1.0*r_sun2km
    buff = 1/b*(r-r0)

    phi = phi0 - buff

    return phi

def get_parkerSpiral(v_sw, phi0):
    parkerend=32
    length_data = 100
    r_vals = np.linspace(0, parkerend, length_data)
    parkerphi = []
    for r in r_vals:
        parkerphi.append(parkerSpiral(r, phi0, v_sw=v_sw))

    x_parker, y_parker = pol2cart(r_vals, parkerphi)
    return x_parker, y_parker


class typeIII:
  def __init__(self, freq, xy, sd):
    self.freq = freq
    self.xy = xy
    self.sd = sd




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process spacecraft names.')
    parser.add_argument('-s','--spacecraft', nargs='+', help='Spacecraft name(s)')
    parser.add_argument('-d','--date', type=str, help='Observation date in YYYY-MM-DD format')
    parser.add_argument('-l','--limits', nargs=4, type=int, help='Limits in the format minX maxX minY maxY')
    parser.add_argument('-r','--res', nargs=1, type=int, help='Resolution')
    parser.add_argument('-c','--cadence', nargs=1, type=int, help='Cadence')


    # Parse command line arguments
    args = parser.parse_args()
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    string =f'Running on PyMC3 v{pm.__version__}'
    print(string)

    ncores_cpu = multiprocessing.cpu_count()

    # generate some test data
    # np.random.seed(1)
    try:
        observation_date = dt.datetime.strptime(args.date, "%Y-%m-%d")
        day = observation_date.day
        month = observation_date.month
        year = observation_date.year
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD format.")
    except Exception as e:
        print(f"Warning. {e}, Date assigned to manual")
        day = 4
        month = 12
        year = 2021
        print( f"Date assigned: {year}-{month:02d}-{day:02d}")


    date_str = f"{year}_{month:02d}_{day:02d}"
    print(f"Observation date: {date_str}")
    # date_str = f"surround"
    spacecraft = []
    sc_buff = []
    if not args.spacecraft:
        # Manually inputing Spacecraft:
        windsc = False
        steasc = True
        stebsc = False
        solosc = True
        pspsc = True
        mexsc = True

        if windsc:
            spacecraft.append("wind")
        if steasc:
            spacecraft.append("stereo_a")
        if stebsc:
            spacecraft.append("stereo_b")
        if solosc:
            spacecraft.append("solo")
        if pspsc:
            spacecraft.append("psp")
        if mexsc:
            spacecraft.append("mex")
    else:
        for sc in args.spacecraft:
            spacecraft.append(sc.lower())

        if "wind" in spacecraft:
            windsc=True
            sc_buff.append("wind")
        else:
            windsc=False

        if "stereo_a" in spacecraft:
            steasc=True
            sc_buff.append("stereo_a")
        else:
            steasc=False

        if "stereo_b" in spacecraft:
            stebsc=True
            sc_buff.append("stereo_b")
        else:
            stebsc=False

        if "solo" in spacecraft:
            solosc=True
            sc_buff.append("solo")
        else:
            solosc=False

        if "psp" in spacecraft:
            pspsc=True
            sc_buff.append("psp")
        else:
            pspsc=False

        if "mex" in spacecraft:
            mexsc=True
            sc_buff.append("mex")
        else:
            mexsc=False

        spacecraft = sc_buff
        print(f"Spacecraft selected: {spacecraft}")

    sc_str = ""
    if windsc:
        sc_str += '_WIND'
    if steasc:
        sc_str += '_STEA'
    if stebsc:
        sc_str += '_STEB'
    if solosc:
        sc_str += '_SOLO'
    if pspsc:
        sc_str += '_PSP'
    if mexsc:
        sc_str += '_MEX'

    if date_str == "surround":
        # SURROUND
        #############################################################################
        theta_sc = int(sys.argv[1])

        print(f"theta_sc:    {theta_sc}")
        L1 = [0.99*(au/R_sun), 0]
        L4 = [(au/R_sun)*np.cos(radians(60)), (au/R_sun)*np.sin(radians(60))]
        L5 = [(au/R_sun)*np.cos(radians(60)), -(au/R_sun)*np.sin(radians(60))]
        # ahead = [(au/R_sun)*np.cos(radians(theta_sc)),(au/R_sun)*np.sin(radians(theta_sc))]
        # behind = [(au/R_sun)*np.cos(radians(theta_sc)),-(au/R_sun)*np.sin(radians(theta_sc))]

        dh = 0.01
        # theta_AB_deg = 90
        theta_AB = np.radians(theta_sc)
        ahead =  pol2cart((1-dh)*(au / R_sun), theta_AB)
        behind = pol2cart((1+dh)*(au / R_sun),-theta_AB)



        stations_rsun = np.array([L1, ahead, behind])
        #############################################################################
    elif date_str == "test":
        stations_rsun = np.array([[200, 200], [-200, -200], [-200, 200], [200, -200]])
    elif date_str == "manual":
        stations_rsun = np.array([[45.27337378, 9.90422281], [-24.42715218, -206.46280171], [212.88183411, 0.]])
        date_str = f"{year}_{month:02d}_{day:02d}"
    else:
        solarsystem = solarmap.get_HEE_coord(date=[year, month, day], objects=spacecraft)
        stations_rsun = np.array(solarsystem.locate_simple())


    N_STATIONS = len(stations_rsun)
    stations = stations_rsun*R_sun.value

    # Making grid
    if not args.limits:
        xrange = [-310,310]
        yrange = [-310, 310]
    else:
        xrange = [args.limits[0], args.limits[1]]
        yrange = [args.limits[2], args.limits[3]]

    if args.res is None:  # Check if 'res' argument is provided
        xres = 10
        yres = xres
    else:
        xres = args.res[0]
        yres = xres

    xmapaxis = np.arange(xrange[0], xrange[1], xres)
    ymapaxis = np.arange(yrange[0], yrange[1], yres)
    if args.cadence is None:  # Check if 'res' argument is provided
        cadence = 60
    else:
        cadence = args.cadence[0]


    pixels0 = len(xmapaxis)*len(ymapaxis)
    print(f" Pixels : {pixels0}")

    # Change these numbers based on your experience.
    tpl_l = 1   # time per loop in mins low end
    tpl_h = 3     # time per loop in mins high end.
    est_low = pixels0*tpl_l/ncores_cpu
    est_high= pixels0*tpl_h/ncores_cpu

    print(f" Estimated Simulation time: {est_low} - {est_high} mins")
    print(f" Estimated Simulation time: {est_low/60:.2f} - {est_high/60:.2f} hours")
    print(f" Estimated Simulation time: {est_low/(60*24):.2f} - {est_high/(60*24):.2f} days")


    """  ---------------------------------------------------------------------   """
    """  -------------------------  ON / OFF ---------------------------------   """
    ###############################################################
    # Check if running in server or local. Change these names to your server names.
    machinenames = ['SERVER1', 'SERVER2','SERVER3', 'SERVER4', 'SERVER5']
    if os.uname().nodename in machinenames:
        runserver = True
    else:
        runserver = False
    ###############################################################
    if runserver is True:
        # Settings for server (Make sure runserver=True)
        runsimulation = True
        run_failed_again = False  # Keep False. NEEDS DEBUGGING> FAILS
        run_savedata = True
        run_loaddata = False
        run_plotdata = False
        run_median_filter = False
        run_median_filter_TrackedRay = False
        run_datainterpolate = False

    else:
        # Settings for Local machine. (Make sure runserver = False)
        runsimulation = True
        run_failed_again = False   # Keep False. NEEDS DEBUGGING> FAILS
        run_savedata = True
        run_loaddata = True
        run_plotdata = True
        run_median_filter = True
        run_median_filter_TrackedRay = False
        run_datainterpolate = False


    yesanswer = ["y".casefold(), "ye".casefold(), "yes".casefold()]
    noanswer = ["n".casefold(), "no".casefold()]


    if date_str == "surround":
        filename = f"./Data/BG_Maps/{date_str}/results_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}stations{sc_str}_{cadence}s_thetasc{theta_sc}.pkl"
    else:
        filename = f"./Data/BG_Maps/{date_str}/results_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}stations{sc_str}_{cadence}s.pkl"

    # Does not make sense to save data or run flagged points if simulation is off.
    if runsimulation is False:
        run_savedata = False
        run_failed_again = False
        if run_loaddata is True:
            #Check if data exists
            isExist = os.path.exists(filename)
            if not isExist:
                # If data does not exist run simulation?
                runsimans = input("The data you are looking for does not exist. Run simulation? y/n:   ")
                if runsimans.casefold() in yesanswer:
                    runsimulation = True
                    runsimflagsans = input("Do you want to run failed points a second time? y/n:    ")
                    if runsimflagsans.casefold() in yesanswer:
                        run_failed_again = True
                    runsavedataans = input("Would you like to save the data? y/n:     ")
                    if runsavedataans.casefold() in yesanswer:
                        run_savedata = True
                run_loaddata = False

    if runsimulation is True:
        # Check if data exists
        isExist = os.path.exists(filename)
        if isExist:
            # If data exists run simulation?
            runsimans = input("There is data for this simulation. Are you sure you want to run? y/n:   ")
            if runsimans.casefold() in noanswer:
                runsimulation = False
                run_failed_again = False
                run_savedata = False
                run_loaddata = True
                run_median_filter = True


    """  ---------------------------------------------------------------------   """




    """  ---------------------------------------------------------------------   """
    """  ------------------------- Simulation --------------------------------   """
    """  ---------------------------------------------------------------------   """
    if runsimulation is True:
        compcost0 = dt.datetime.now()
        coresforloop = multiprocessing.cpu_count()

        coresthres = 20
        if coresforloop >= coresthres:
            # use all cores except 1 for every 20 cores.
            # This is done as a factor of safety. Sometimes the server runs out of cores and the simulation gets stuck
            # reduce coresthres if this keeps happening. Recommended at 20.
            # if running on local machine, no need to change this. keep above number of cores on local machine.
            coresforloop = coresforloop - int(coresforloop/coresthres)


        results = Parallel(n_jobs=coresforloop, verbose=100)(delayed(parallel_pos_map)(i, j, stations=stations, xrange=xrange,
                                                                                       yrange=yrange, xres=xres, yres=yres, t_cadence=cadence, figdir="./traceplots",
                                                                                       date_str=date_str) for i in xmapaxis for j in ymapaxis)
        delta_obs=np.array(results)[:,0]
        flaggedpoints = np.array(results)[:,1:]
        compcost1 = dt.datetime.now()


        simulation_report(title="Positioner Mapping Parallel", xrange=xrange, yrange=yrange, xres=xres, yres=yres, pixels=pixels0,
                          coresused=coresforloop, tstart=compcost0, tfinal=compcost1)

        delta_obs = np.reshape(delta_obs, (len(xmapaxis), len(ymapaxis)))

    """  ---------------------------------------------------------------------   """
    """  ------------------- End - of - Simulation ---------------------------   """
    """  ---------------------------------------------------------------------   """



    """  ------------------------ Save Data ----------------------------------   """
    if run_savedata is True:
        savedata(delta_obs, xmapaxis, ymapaxis, stations_rsun, filename=filename)
    """  ---------------------------------------------------------------------   """


    """  ------------------------ Load Data ----------------------------------   """
    if run_loaddata is True:
        xmapaxis, ymapaxis, delta_obs, stations_rsun = loaddata(filename)
    """  ---------------------------------------------------------------------   """



    """  ------------------------ PLOT Data ----------------------------------   """
    if run_plotdata is True:
        fname = f"/bayes_positioner_map_{xmapaxis[0]}_{xmapaxis[-1]}_{ymapaxis[0]}_{ymapaxis[-1]}_{xres}_{yres}_{N_STATIONS}_30s.jpg"
        plot_map_simple(delta_obs*2, xmapaxis, ymapaxis, stations_rsun, vmin=np.min(delta_obs*2), vmax=np.max(delta_obs*2), showfigure=True, savefigure=True, date_str=date_str, filename=fname)
    """  ---------------------------------------------------------------------   """

    """  ------------------------ Median Filter ---------------------------   """
    if run_median_filter is True:
    # from scipy.ndimage import median_filter as medfil
        delta_obs2 = delta_obs*2   # 2std for 95% confidence intervalm
        median_filter_image = medfil(delta_obs2, size=(6,6))
        median_filter_image = medfil(median_filter_image, size=(6,6))
        fname = f"/bayes_positioner_map_median_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}.jpg"
        plot_map_simple(median_filter_image, xmapaxis, ymapaxis, stations_rsun,vmin=np.min(median_filter_image), vmax=np.max(median_filter_image), savefigure=True, date_str=date_str,filename=fname)
    """  ---------------------------------------------------------------------   """
