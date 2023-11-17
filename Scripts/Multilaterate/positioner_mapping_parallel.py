from bayes_positioner import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import cm

from matplotlib.animation import FuncAnimation, writers

import os

import pymc3 as pm
from scipy.stats import gaussian_kde
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit


import arviz as az
from astropy.constants import c, m_e, R_sun, e, eps0, au
import solarmap
import datetime as dt

from math import sqrt, radians
import math
from joblib import Parallel, delayed
import multiprocessing

from contextlib import contextmanager
import sys, os
import logging
from termcolor import colored

# import huxt as H



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

def parallel_pos_map(x1,y1,stations,xrange,yrange,xres,yres,t_cadence=60, cores=1, traceplotsave=False, figdir=f"./traceplots", date_str="date"):
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
        t0_true = 100  # source time. can be any constant, as long as it is within the uniform distribution prior on t0
        d_true = np.linalg.norm(stations - x_true, axis=1)
        t1_true = d_true / v_true  # true time of flight values
        # t_obs = t1_true-t0_true# true time difference of arrival values
        t_obs = t1_true
        # np.random.seed(1)
        # t_obs = t_obs+0.05*np.random.randn(*t_obs.shape)# noisy observations

        # Make sure to use cores=1 when using parallel loops. cores in triangulate function refers to number of cores
        # used by the MCMC solver.
        results_out = triangulate(stations, t_obs, t_cadence=t_cadence, cores=1, progressbar=False, report=0, plot=0)

        mu = results_out[0]
        sd = results_out[1]
        t1_pred = results_out[2]
        trace = results_out[3]
        summary = results_out[4]
        t_emission = results_out[5]
        v_analysis = results_out[6]





        # report
        # print(summary)
        # print(f"t0_true: {t0_true}")
        # print(f"t_obs:    {t_obs}")
        # print(f"t1_true: {t1_true}")
        # print(f"t1_pred: {t1_pred}")
        # print(f"stations: {stations}")

        if traceplotsave == True:
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
        xy = mu[0] / R_sun.value, mu[1] / R_sun.value

        #delta_obs = sqrt((xy[0] - x_true[0]) ** 2 + (xy[1] - x_true[1]) ** 2)           # Deltaobs = xy - x_true Deprecated
        delta_obs =  np.amax([sd[0]/R_sun.value , sd[1]/R_sun.value])               # Bayesian uncertainty. Pick the largest one.
        print(f"delta_obs: {delta_obs}")
        res = np.array([delta_obs, np.nan, np.nan])
        logginfo = f"{dt.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} : END  {currentloop}"
        logging.info(logginfo)
        tloop1 = dt.datetime.now()
        print(f" P({x1},{y1}),  {colored('SUCCESS', 'green')}:  {res}")


    except:
        delta_obs = 600
        print(f"SIM {colored('FAILED', 'red')} at P({x1},{y1}), SKIPPED")
        res = np.array([delta_obs, x1, y1])
        tloop1 = dt.datetime.now()
        pass

    tloopinseconds = (tloop1 - tloop0).total_seconds()
    print(f"Time Loop : {tloop1 - tloop0}   :   {tloopinseconds}s  \n  loop {mesh[x1,y1]}/{len(mesh)} DONE ")

    estimated_t_left = tloopinseconds*((len(mesh) - mesh[x1,y1]))/(multiprocessing.cpu_count())
    estimated_t_left_timedelta = dt.timedelta(seconds=estimated_t_left)
    print(f"{colored('Estimated time left', 'yellow')}: {estimated_t_left_timedelta} ")

    return res

def plot_map_simple(delta_obs, xmapaxis, ymapaxis, stations,vmin=0,vmax=30, savefigure=False, showfigure=True, title="",figdir=f"./MapFigures", date_str="date", filename="fig.jpg"):
    xres = xmapaxis[1]-xmapaxis[0]
    yres = ymapaxis[1]-ymapaxis[0]
    N_STATIONS = len(stations)

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

    ax.set_xlabel("'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel("'HEE - Y / $R_{\odot}$'", fontsize=22)
    ax.set_title(title, fontsize=22)

    if savefigure == True:
        figdir = f"{figdir}/{date_str}"
        mkdirectory(figdir)
        plt.savefig(figdir+filename, bbox_inches='tight', pad_inches=0.01, dpi=300)

    if showfigure == True:
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
                                title="",figdir=f"./MapFigures", date_str="date", filename="fig_Tracked.jpg"):


    xy_vals = np.array(list(trackedtypeIII.xy))
    xy = xy_vals/R_sun.value
    tracked_freqs = trackedtypeIII.freq
    sd_vals = np.array(list(trackedtypeIII.sd))
    sd = sd_vals/R_sun.value


    xres = xmapaxis[1]-xmapaxis[0]
    yres = ymapaxis[1]-ymapaxis[0]
    N_STATIONS = len(stations)


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


    if parker_spiral == True:
        ax.plot(x_parker,y_parker,"k--")
        ax.plot(x_parker_plus,y_parker_plus,"k--", markersize=0.5)
        ax.plot(x_parker_minus,y_parker_minus,"k--")
        # ax.plot(x_parker120, y_parker120,"k--")
        # ax.plot(x_parker240, y_parker240,"k--")

    import scipy.io
    mat = scipy.io.loadmat('Data/2012_06_07/SEMP_spiral_results.mat')
    x_zhang = mat["x_ParkerSpiral_Cart"]
    y_zhang = mat["y_ParkerSpiral_Cart"]
    ax.plot(x_zhang[0], y_zhang[0], "r--",label="Zhang et al. 2019")



    # # ELLIPSES
    colors = cm.turbo(list(np.linspace(0,1.0,len(xy))))
    if confidence ==True:
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

    ax.set_xlabel("'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel("'HEE - Y / $R_{\odot}$'", fontsize=22)
    ax.set_title(title, fontsize=22)

    if savefigure == True:
        figdir = f"{figdir}/{date_str}"
        mkdirectory(figdir)
        plt.savefig(figdir+filename, bbox_inches='tight', pad_inches=0.01, dpi=300)

    if showfigure == True:
        plt.show(block=False)
    else:
        plt.close(fig)

    return fig,ax


def plot_map_simple_withTracked_zoom(delta_obs, trackedtypeIII, xmapaxis, ymapaxis, stations,
                                vmin=0,vmax=30,v_sw=400,phi_sw=0, dphi=0,
                                savefigure=False, showfigure=True,
                                title="",figdir=f"./MapFigures", date_str="date", filename="fig_Tracked.jpg"):

    xy_vals = np.array(list(trackedtypeIII.xy))
    xy = xy_vals/R_sun.value
    tracked_freqs = trackedtypeIII.freq
    sd_vals = np.array(list(trackedtypeIII.sd))
    sd = sd_vals/R_sun.value


    xres = xmapaxis[1]-xmapaxis[0]
    yres = ymapaxis[1]-ymapaxis[0]
    N_STATIONS = len(stations)


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

    i = 0
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

    ax.set_xlabel("'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel("'HEE - Y / $R_{\odot}$'", fontsize=22)
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


    if savefigure == True:
        figdir = f"{figdir}/{date_str}"
        mkdirectory(figdir)
        plt.savefig(figdir+filename, bbox_inches='tight', pad_inches=0.01, dpi=300)

    if showfigure == True:
        plt.show(block=False)
    else:
        plt.close(fig)




def savedata(delta_obs, xmapaxis, ymapaxis, stations,dir=f"./Data", date_str="date",filename="output.pkl"):
    import pickle
    mkdirectory(dir+f"/{date_str}")


    xres = xmapaxis[1]-xmapaxis[0]
    yres = ymapaxis[1]-ymapaxis[0]
    N_STATIONS = len(stations)

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
    from scipy import interpolate


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
    r_sun2km = 695700;     #
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


def plot_HUXt(model, time, save=False, tag='', fighandle=np.nan, axhandle=np.nan,
         minimalplot=False, plotHCS=True):
    """
    Make a contour plot on polar axis of the solar wind solution at a specific time.
    Args:
        model: An instance of the HUXt class with a completed solution.
        time: Time to look up closet model time to (with an astropy.unit of time).
        save: Boolean to determine if the figure is saved.
        tag: String to append to the filename if saving the figure.
        fighandle: Figure handle for placing plot in existing figure.
        axhandle: Axes handle for placing plot in existing axes.
        minimalplot: Boolean, if True removes colorbar, planets, spacecraft, and labels.
        plotHCS: Boolean, if True plots heliospheric current sheet coordinates
    Returns:
        fig: Figure handle.
        ax: Axes handle.
    """
    # Get plotting data
    lon_arr, dlon, nlon = H.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)
    mymap = mpl.cm.viridis
    v_sub = model.v_grid.value[id_t, :, :].copy()
    plotvmin = 200
    plotvmax = 810
    dv = 10
    ylab = "Solar Wind Speed (km/s)"

    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.NaN
        if model.lon.size != 1:
            for i, lo in enumerate(model.lon):
                id_match = np.argwhere(lon_arr == lo)[0][0]
                v[:, id_match] = v_sub[:, i]
        else:
            print('Warning: Trying to contour single radial solution will fail.')
    else:
        v = v_sub

    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(plotvmin, plotvmax + dv, dv)

    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    else:
        fig = fighandle
        ax = axhandle

    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    # Set edge color of contours the same, for good rendering in PDFs
    for c in cnt.collections:
        c.set_edgecolor("face")

    # Add on CME boundaries
    cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
    for j, cme in enumerate(model.cmes):
        cid = np.mod(j, len(cme_colors))
        cme_lons = cme.coords[id_t]['lon']
        cme_r = cme.coords[id_t]['r'].to(u.solRad)
        if np.any(np.isfinite(cme_r)):
            # Pad out to close the profile.
            cme_lons = np.append(cme_lons, cme_lons[0])
            cme_r = np.append(cme_r, cme_r[0])
            ax.plot(cme_lons, cme_r, '-', color=cme_colors[cid], linewidth=3)

    ax.set_ylim(0, model.r.value.max())
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if not minimalplot:
        # determine which bodies should be plotted
        plot_observers = zip(['EARTH', 'VENUS', 'MERCURY', 'STA', 'STB'],
                             ['ko', 'mo', 'co', 'rs', 'y^'])
        if model.r[0] > 200 * u.solRad:
            plot_observers = zip(['EARTH', 'MARS', 'JUPITER', 'SATURN'],
                                 ['ko', 'mo', 'ro', 'cs'])

        # Add on observers
        for body, style in plot_observers:
            obs = model.get_observer(body)
            deltalon = 0.0 * u.rad
            if model.frame == 'sidereal':
                earth_pos = model.get_observer('EARTH')
                deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]

            obslon = H._zerototwopi_(obs.lon[id_t] + deltalon)
            ax.plot(obslon, obs.r[id_t], style, markersize=16, label=body)

        # Add on a legend.
        fig.legend(ncol=5, loc='lower center', frameon=False, handletextpad=0.2, columnspacing=1.0)

        ax.patch.set_facecolor('slategrey')
        fig.subplots_adjust(left=0.05, bottom=0.16, right=0.95, top=0.99)

        # Add color bar
        pos = ax.get_position()
        dw = 0.005
        dh = 0.045
        left = pos.x0 + dw
        bottom = pos.y0 - dh
        wid = pos.width - 2 * dw
        cbaxes = fig.add_axes([left, bottom, wid, 0.03])
        cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
        cbar1.set_label(ylab)
        cbar1.set_ticks(np.arange(plotvmin, plotvmax, dv * 10))

        # Add label
        label = "   Time: {:3.2f} days".format(model.time_out[id_t].to(u.day).value)
        label = label + '\n ' + (model.time_init + time).strftime('%Y-%m-%d %H:%M')
        fig.text(0.70, pos.y0, label, fontsize=16)

        label = "HUXt2D \nLat: {:3.0f} deg".format(model.latitude.to(u.deg).value)
        fig.text(0.175, pos.y0, label, fontsize=16)

        # plot any tracked streaklines
        if model.track_streak:
            nstreak = len(model.streak_particles_r[0, :, 0, 0])
            for istreak in range(0, nstreak):
                # construct the streakline from multiple rotations
                nrot = len(model.streak_particles_r[0, 0, :, 0])
                streak_r = []
                streak_lon = []
                for irot in range(0, nrot):
                    streak_lon = streak_lon + model.lon.value.tolist()
                    streak_r = streak_r + (
                                model.streak_particles_r[id_t, istreak, irot, :] * u.km.to(u.solRad)).value.tolist()

                    # add the inner boundary postion too
                mask = np.isfinite(streak_r)
                plotlon = np.array(streak_lon)[mask]
                plotr = np.array(streak_r)[mask]
                # only add the inner boundary if it's in the HUXt longitude grid
                foot_lon = H._zerototwopi_(model.streak_lon_r0[id_t, istreak])
                dlon_foot = abs(model.lon.value - foot_lon)
                if dlon_foot.min() <= model.dlon.value:
                    plotlon = np.append(plotlon, foot_lon + model.dlon.value / 2)
                    plotr = np.append(plotr, model.r[0].to(u.solRad).value)

                ax.plot(plotlon, plotr, 'k')

        # plot any HCS that have been traced
        if plotHCS and hasattr(model, 'b_grid'):
            for i in range(0, len(model.hcs_particles_r[:, 0, 0, 0])):
                r = model.hcs_particles_r[i, id_t, 0, :] * u.km.to(u.solRad)
                lons = model.lon
                ax.plot(lons, r, 'w.')

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_frame_{:03d}.png".format(cr_num, tag, id_t)
        filepath = os.path.join(model._figure_dir_, filename)
        fig.savefig(filepath)

    return fig, ax


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    string ='Running on PyMC3 v{}'.format(pm.__version__)
    print(string)

    ncores_cpu = multiprocessing.cpu_count()

    # generate some test data
    # np.random.seed(1)
    day = 7
    month = 6
    year = 2012
    date_str = f"{year}_{month:02d}_{day:02d}"
    # date_str = f"surround"


    if date_str == "surround":
        # SURROUND
        #############################################################################
        theta_sc = int(sys.argv[1])

        print(f"theta_sc:    {theta_sc}")
        L1 = [0.99*(au/R_sun),0]
        L4 = [(au/R_sun)*np.cos(radians(60)),(au/R_sun)*np.sin(radians(60))]
        L5 = [(au/R_sun)*np.cos(radians(60)),-(au/R_sun)*np.sin(radians(60))]
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
        stations_rsun = np.array([[45.27337378, 9.90422281],[-24.42715218,-206.46280171],[ 212.88183411,0.]])
        date_str = f"{year}_{month:02d}_{day:02d}"
    else:
        solarsystem = solarmap.get_sc_coord(date=[year, month, day], objects=["stereo_b", "stereo_a", "earth"])
        stations_rsun = np.array(solarsystem.locate_simple())

    # Wind ephemeris not available for this date So 99% of Sun-Earth distance.
    # ValueError: Horizons Error: No ephemeris for target "Wind (spacecraft)" prior to A.D. 2019-OCT-08 00:01:09.1823 TD
    spacecraft = ["stereo_b", "stereo_a", "wind"]    # redefining wind as the name of the spacecraft
    stations_rsun[2][0] = 0.99 * stations_rsun[2][0]


    N_STATIONS = len(stations_rsun)
    stations = stations_rsun*R_sun.value

    # Making grid
    xrange = [-250,250]
    xres = 10
    yrange = [-250, 250]
    yres = xres
    xmapaxis = np.arange(xrange[0], xrange[1], xres)
    ymapaxis = np.arange(yrange[0], yrange[1], yres)
    cadence = 30

    pixels0 = len(xmapaxis)*len(ymapaxis)
    print(f" Pixels : {pixels0}")

    # Change these numbers based on your experience.
    tpl_l = 0.5   # time per loop in mins low end
    tpl_h = 5     # time per loop in mins high end.
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
    if runserver == True:
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
        filename = f"./Data/{date_str}/results_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}stations_{cadence}s_thetasc{theta_sc}.pkl"
    else:
        filename = f"./Data/{date_str}/results_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}stations_{cadence}s.pkl"

    #Doesnt make sense to save data or run flagged points if simulation is off.
    if runsimulation == False:
        run_savedata = False
        run_failed_again = False
        if run_loaddata == True:
            #Check if data exists
            isExist = os.path.exists(filename)
            if not isExist:
                # If data doesnt exist run simulation?
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

    if runsimulation == True:
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
    if runsimulation == True:
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
                                                                                       yrange=yrange,xres=xres,yres=yres,t_cadence = cadence, figdir=f"./traceplots",
                                                                                       date_str=date_str) for i in xmapaxis for j in ymapaxis)
        delta_obs=np.array(results)[:,0]
        flaggedpoints = np.array(results)[:,1:]
        compcost1 = dt.datetime.now()


        if (run_failed_again == True) and np.isfinite(flaggedpoints).any():
            print("RUNNING FAILED POINTS AGAIN")
            flagindex = []
            flagPx = []
            flagPy = []
            for i in range(0, len(flaggedpoints)):
                if ~np.isnan(flaggedpoints[i][0]) and ~np.isnan(flaggedpoints[i][1]):
                    flagindex.append(i)
                    flagPx.append([flaggedpoints[i][0]])
                    flagPy.append([flaggedpoints[i][1]])

            results_failed = Parallel(n_jobs=coresforloop, verbose=100)(delayed(parallel_pos_map)(flagPx[i], flagPy[i], stations=stations, xrange=xrange, yrange=yrange, xres=xres, yres=yres) for i in range(0,len(flagindex)))
            delta_obs[flagindex] = np.array(results_failed,dtype=object)[:,0]
            flaggedpoints = np.array(results_failed, dtype=object)[:, 1]







        simulation_report(title="Positioner Mapping Parallel", xrange=xrange, yrange=yrange, xres=xres, yres=yres, pixels=pixels0,
                          coresused=coresforloop, tstart=compcost0, tfinal=compcost1)

        delta_obs = np.reshape(delta_obs, (len(xmapaxis), len(ymapaxis)))

    """  ---------------------------------------------------------------------   """
    """  ------------------- End - of - Simulation ---------------------------   """
    """  ---------------------------------------------------------------------   """



    """  ------------------------ Save Data ----------------------------------   """
    if run_savedata == True:
        savedata(delta_obs, xmapaxis, ymapaxis, stations_rsun, dir=f"./Data", date_str=date_str, filename=filename)
    """  ---------------------------------------------------------------------   """


    """  ------------------------ Load Data ----------------------------------   """
    if run_loaddata == True:
        xmapaxis, ymapaxis, delta_obs, stations_rsun = loaddata(filename)
    """  ---------------------------------------------------------------------   """



    """  ------------------------ PLOT Data ----------------------------------   """
    if run_plotdata == True:
        fname = f"/bayes_positioner_map_{xmapaxis[0]}_{xmapaxis[-1]}_{ymapaxis[0]}_{ymapaxis[-1]}_{xres}_{yres}_{N_STATIONS}_30s.jpg"
        plot_map_simple(delta_obs*2, xmapaxis, ymapaxis, stations_rsun, vmin=np.min(delta_obs*2), vmax=np.max(delta_obs*2), showfigure=True, savefigure=True, date_str=date_str, filename=fname)
    """  ---------------------------------------------------------------------   """

    """  ------------------------ Median Filter ---------------------------   """
    if run_median_filter == True:
    # from scipy.ndimage import median_filter as medfil
        delta_obs2 = delta_obs*2   # 2std for 95% confidence intervalm
        median_filter_image = medfil(delta_obs2, size=(6,6))
        median_filter_image = medfil(median_filter_image, size=(6,6))
        fname = f"/bayes_positioner_map_median_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}.jpg"
        plot_map_simple(median_filter_image, xmapaxis, ymapaxis, stations_rsun,vmin=np.min(median_filter_image), vmax=np.max(median_filter_image), savefigure=True, date_str=date_str,filename=fname)
    """  ---------------------------------------------------------------------   """

    if date_str != "surround":
        """  ------------------------ Median w/ Tracked data ---------------------------   """
        if run_median_filter_TrackedRay == True:
            # from scipy.ndimage import median_filter as medfil
            delta_obs2 = delta_obs*2
            median_filter_image = medfil(delta_obs2, size=(6,6))



            fname = f"/bayes_positioner_map_median_tracked_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}.jpg"


            #median_filter_image[np.where(abs(xmapaxis) >= 220),:] = 300
            trackedfile = "./Data/2020_07_11/Tracking/TRACKING_2020_07_11_results_4stations.pkl"
            trackedfile = "/Users/canizares/OneDrive/Work/0_PhD/Projects/dynSpectra/Data/2020_07_21/Tracking/TRACKING_2020_07_21_results_4stations.pkl"
            trackedfile = "/Users/canizares/OneDrive/Work/0_PhD/Projects/BELLA/scripts/Data/2012_06_07/Tracking/TRACKING_2012_06_07_results_3stations_LE.pkl"
            trackedfile = "/Users/canizares/OneDrive/Work/0_PhD/Projects/BELLA/scripts/Data/2012_06_07/Tracking/TRACKING_2012_06_07_results_3stations_LE_Freqs_0.15_13.pkl"
            trackedfile = "/Users/canizares/OneDrive/Work/0_PhD/Projects/BELLA/scripts/Data/2012_06_07/Tracking/TRACKING_2012_06_07_results_3stations_LE_Freqs_0.15_2.pkl"
            tracked = loadtrackedtypeiii(trackedfile)

            trackedtypeIII = typeIII(tracked[:,0], tracked[:,1],tracked[:,2])
            timestracked  =  tracked[:, 3]





            fig_1, ax_1 = plot_map_simple_withTracked(median_filter_image,trackedtypeIII, xmapaxis, ymapaxis, stations_rsun,
                                        vmin=np.min(median_filter_image), vmax=np.max(median_filter_image),
                                        parker_spiral=True, v_sw=420, phi_sw=20, dphi=3,
                                        confidence=True,
                                        savefigure=True, date_str=date_str,spacecraft=["stereo_b", "stereo_a", "wind"], filename=fname)


        """  ---------------------------------------------------------------------   """










        """  ------------------------ Interpolate Data ---------------------------   """
        """ GENERATES ARTIFACTS. USE MEDIAN FILTER AFTER"""
        if run_datainterpolate == True:
            delta_obs2 = delta_obs*2
            factor = 3   # Recommended factor of 2.
            xnew,ynew, znew = interpolate_map(delta_obs2, xmapaxis, ymapaxis, scale_factor=factor, kind="quintic")
            for i in range(0,factor):
               znew = medfil(znew, size=(6,6))
            tracked = loadtrackedtypeiii(trackedfile)
            plot_map_simple_withTracked(znew,trackedtypeIII, xnew, ynew, stations_rsun,vmin=np.min(median_filter_image), vmax=150,v_sw=300,
                                        phi_sw=300, dphi=20, savefigure=True, date_str=date_str, filename=fname)
            #savedata(znew,xnew,ymapaxis,stations_rsun,dir=f"./InterpData/", date_str=date_str)
        """  ---------------------------------------------------------------------   """

        #
        # xy = np.array(list(trackedtypeIII.xy))
        # r_tIII = []
        # theta_tIII = []
        # for i in range(0, len(xy)):
        #     r_buff, theta_buff = cartesian_to_polar(xy[i,0],xy[i,1])
        #     r_tIII.append(r_buff)
        #     theta_tIII.append(theta_buff)
        #
        # r_tIII = np.array(r_tIII)
        # theta_tIII = np.array(theta_tIII)
        #
        # theta_tIII_deg = theta_tIII * (180/math.pi)
        #
        #
        #
        # threshold = 0
        # idx = np.where(theta_tIII<threshold)
        #
        # x_data = theta_tIII[idx]
        # y_data = r_tIII[idx]
        #
        # from scipy.stats import linregress
        #
        # slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        # line = slope * x_data + intercept
        # x_intercept = -intercept/slope
        # x_fit = np.linspace(x_data.min(), x_intercept, num=100)
        # line_fit = slope*x_fit + intercept
        #
        # coefficients = np.polyfit(x_data, y_data, 1)
        # y_poly = np.polyval(coefficients, x_data)
        # plt.figure()
        # plt.plot(theta_tIII,r_tIII/R_sun.value, "r.")
        # plt.plot(x_data, line/R_sun.value, 'k--')
        # plt.plot(x_fit, line_fit/R_sun.value, 'k--')
        # # plt.plot(x_data, y_poly/R_sun.value, 'b--')
        # plt.xlabel("Theta [radians]")
        # plt.ylabel("r  [Rsun]")
        # plt.show(block=False)
        #
        #
        # residuals = line-y_data
        # residuals2 = y_poly-y_data
        # plt.figure()
        # plt.plot(residuals, "r.")
        # plt.ylabel('resiudals')
        # plt.show(block=False)
        #
        # omega = 2.662e-06
        # v_sw = -slope*omega/1000      # km/s
        # print(f"v_sw: {v_sw}")
        #
        #
        #

        # QUARANTINE DELETE I
        # SOLAR WIND MODELLING
        #
        # filename_vsw = "/Users/canizares/OneDrive/Work/0_PhD/Projects/2012_06_07_WSTASTB/SolarWind/2012_06_07_MASv_huxt.csv"
        #
        # xy_vals = np.array(list(trackedtypeIII.xy))
        # xy = xy_vals / R_sun.value
        # tracked_freqs = trackedtypeIII.freq
        # sd_vals = np.array(list(trackedtypeIII.sd))
        # sd = sd_vals / R_sun.value
        #
        #
        #
        #
        # # Load data from CSV file
        # data_vsw = np.genfromtxt(filename_vsw, delimiter=',')
        #
        # r = np.linspace(5, 240.5, num=158)
        # phi = np.linspace(0.0254,6.2586, num=128)  # rad
        # # phi = np.linspace(0,2*math.pi, num=128)  # rad
        #
        # R, PHI = np.meshgrid(r, phi)
        # Z = data_vsw.T
        #
        # # Convert to Cartesian coordinates
        # X, Y = R * np.cos(PHI), R * np.sin(PHI)
        #
        # phi0 = 20
        # parkerphi = []
        # parkerend = 215
        # v_sw = 420
        # omega = 2.66e-6
        #
        # for r in range(0, parkerend):
        #     parkerphi.append(parkerSpiral(r, phi0, v_sw=v_sw, theta=0))
        #
        # x_parker, y_parker = pol2cart(np.arange(0, parkerend), parkerphi)
        #
        # # Create the plot
        # fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        # plt.subplots_adjust(top=1, bottom=0)
        #
        # lon, rad = np.meshgrid(phi, r)
        # plotvmin = 200
        # plotvmax = 810
        # dv = 10
        # levels = np.arange(plotvmin, plotvmax + dv, dv)
        #
        # # im_sw = ax.pcolormesh(X, Y, Z, cmap='plasma')#,shading='gouraud')
        # # im_sw = ax.pcolormesh(X, Y, Z, cmap='plasma')
        # # im_sw = ax.pcolormesh(X, Y, Z, cmap='plasma', vmin=Z.min(), vmax=Z.max())#,shading='gouraud'
        # im_sw = ax.contourf(X, Y, Z, cmap='plasma',levels=levels, extend='both')
        #
        #
        #
        # im_contour = ax.contour(X, Y, Z,  colors='black')
        # ax.clabel(im_contour, inline=True, fontsize=8)
        # cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.30])
        # fig.colorbar(im_sw, cax=cbar_ax)
        #
        # # earth_orbit = plt.Circle((0, 0), au / R_sun + 5, color='k', linestyle="dashed", fill=None)
        # # ax.add_patch(earth_orbit)
        #
        # ax.scatter(stations[:, 0]/R_sun.value, stations[:, 1]/R_sun.value, color="w", marker="^", edgecolors="k", s=180, label="Spacecraft")
        #
        # i = 0
        # for sc in spacecraft:
        #     sclab = ax.text(stations[i, 0]/R_sun.value, stations[i, 1]/R_sun.value, sc, color="w", fontsize=22)
        #     sclab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
        #     i += 1
        #
        # colors = cm.jet(list(np.linspace(0, 1.0, len(xy))))
        #
        # im_track = ax.scatter(xy[:, 0], xy[:, 1], c=tracked_freqs, cmap="jet", marker=".", edgecolors="w", s=100,
        #                       label="TrackedBeam",zorder = 1000)
        # ax.plot(x_parker, y_parker, "r--")
        #
        #
        # cbar_ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.30])
        # fig.colorbar(im_track, cax=cbar_ax2)
        # cbar_ax2.set_ylabel('Tracked beam freq (MHz)', fontsize=18)
        #
        # ax.plot(au / R_sun + 5, 0, 'bo', label="Earth", markersize=10)
        # ax.plot(0, 0, 'yo', label="Sun", markersize=10, markeredgecolor='k')
        #
        # cbar_ax.set_ylabel('Vsw (km/s)', fontsize=18)
        # ax.set_title('V solar wind 2012-06-07')
        # ax.set_aspect('equal')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.show(block=False)

