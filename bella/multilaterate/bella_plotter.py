import os
import math
import logging
import datetime as dt
import multiprocessing
from contextlib import contextmanager

import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import solarmap
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LogFormatter
# import pymc3 as pm
from scipy.ndimage import median_filter

import astropy.units as u
# import arviz as az
from astropy.constants import R_sun, au

import sys
# sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA')
# from bella.multilaterate.bayes_positioner import *

plt.rcParams.update({'font.size': 18})
plt.rcParams["font.family"] = "Times New Roman"

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def pol2cart(r, phi):
    x = r* np.cos(phi)
    y = r* np.sin(phi)
    return(x, y)

def cart2pol(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    return(r,theta)

def cartesian_to_polar(x, y,x0y0 = [0,0]):
    x0 = x0y0[0]
    y0 = x0y0[1]
    r = math.sqrt((x0-x)**2 + (y0-y)**2)
    theta = math.atan2(y, x)
    return (r, theta)





def mkdirectory(directory):
    dir = directory
    isExist = os.path.exists(dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir)
        print("The new directory is created!")
    return dir


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

    ax.set_xlabel(r"HEE - X (R$_\odot$)", fontsize=22)
    ax.set_ylabel(r"HEE - Y (R$_\odot$)", fontsize=22)
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



def plot_bella_map(fig,ax,delta_obs, xmapaxis, ymapaxis, stations,
                   vmin=0,vmax=30,
                   spacecraft_labels=[],cmap='plasma',
                   title="", figdir="./MapFigures", date_str="date",showcolorbar=True,showlegend=True, cbar_axes =[0.91, 0.5, 0.01, 0.35], objects =["earth_orbit", "earth", "sun", "spacecraft"],
                   linecolor = "k"):


    xmapaxis[1]-xmapaxis[0]
    ymapaxis[1]-ymapaxis[0]
    len(stations)

    # fig, ax = plt.subplots(1,1,figsize=(11,11))
    # plt.subplots_adjust(top=1, bottom=0)

    im_0 = ax.pcolormesh(xmapaxis, ymapaxis, delta_obs.T, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)

    # Uncomment to see where simulation failed.
    # im_fail = ax.pcolormesh(xmapaxis, ymapaxis, np.ma.masked_values(delta_obs,200).T, cmap='Greys', vmin=vmin, vmax=vmax)

    if "earth_orbit" in objects:
        earth_orbit = plt.Circle((0, 0), au/R_sun + 5, color=linecolor, linestyle="dashed", fill=None)
        ax.add_patch(earth_orbit)


    if "spacecraft" in objects:
        ax.scatter(stations[:,0], stations[:,1], color = "w", marker="^", edgecolors="k", s=180, label="Spacecraft")

    if spacecraft_labels != []:
        i = 0
        for sc in spacecraft_labels:
            sclab = ax.text(stations[i, 0], stations[i, 1], sc, color="w", fontsize=18)
            sclab.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
            i+=1

    ax.set_aspect('equal')

    ax.set_xlim(xmapaxis[0], xmapaxis[-1])
    ax.set_ylim(ymapaxis[0], ymapaxis[-1])

    # fig.subplots_adjust(right=0.9)
    if showcolorbar is True:
        cbar_ax = fig.add_axes(cbar_axes) # [left, bottom, width, height]
        fig.colorbar(im_0, cax=cbar_ax)
        cbar_ax.set_ylabel(r'BELLA uncertainty (R$_\odot$)', fontsize=18)
        cbar_ax.tick_params(labelsize=15)

    if "earth" in objects:
        ax.plot(au / R_sun+5, 0, 'bo', label="Earth", markersize=10)
    if "sun" in objects:
        ax.plot(0, 0, 'yo', label="Sun", markersize=10, markeredgecolor ='k')

    if showlegend is True:
        ax.legend(loc=1)

    ax.set_xlabel(r"HEE - X  (R$_\odot$)", fontsize=20)
    ax.set_ylabel(r"HEE - Y  (R$_\odot$)", fontsize=20)
    ax.set_title(title, fontsize=22)


    return fig,ax



def plot_tracked_typeIII(fig, ax, trackedtypeIII, confidence=False, showcolorbar=True, cbar_sep_ax=True, cbar_axes=[0.91, 0.1, 0.01, 0.35], marker=".",cmap="turbo", label="TOA BELLA",s=100, edgecolors="w",norm=[],zorder=1000):
    xy_vals = np.array(list(trackedtypeIII.xy))
    xy = xy_vals/R_sun.value
    tracked_freqs = trackedtypeIII.freq
    sd_vals = np.array(list(trackedtypeIII.sd))
    sd = sd_vals/R_sun.value

    if norm == []:
        norm = matplotlib.colors.LogNorm()

    # Access the colormap function dynamically
    cmap_func = getattr(cm, cmap)
    # Generate colors
    colors = cmap_func(np.linspace(0, 1.0, len(xy)))

    # colors = cm.turbo(list(np.linspace(0,1.0,len(xy))))


    if confidence is True:
        i = 0
        ell_track_uncertainty = matplotlib.patches.Ellipse(xy=(xy[i, 0], xy[i, 1]),
                                                           width=2*sd[i, 0], height=2*sd[i, 1],
                                                           angle=0., edgecolor=colors[i], lw=1.5)
        for i in range(1, len(xy)):
            ell_track_uncertainty =matplotlib.patches.Ellipse(xy=(xy[i,0], xy[i,1]),
                      width=2*sd[i,0], height=2*sd[i,1],
                      angle=0.,edgecolor=colors[i], lw=1.5, zorder=zorder-1)

            ell_track_uncertainty.set_facecolor('none')


            ax.add_patch(ell_track_uncertainty)

    im_track = ax.scatter(xy[:,0], xy[:,1],c = tracked_freqs, cmap=cmap, marker=marker, edgecolors=edgecolors, s=s, label=label, norm=norm,zorder=zorder)

    if showcolorbar is True:
        if cbar_sep_ax:
            cbar_ax2 = fig.add_axes(cbar_axes) # [left, bottom, width, height]
            formatter = LogFormatter(10, labelOnlyBase=False)
            fig.colorbar(im_track, cax=cbar_ax2, format=formatter)
            cbar_ax2.set_ylabel('Tracked beam freq (MHz)', fontsize=18)

            cbar_ax2.tick_params(axis='y', which='minor',labelsize=15)
            cbar_ax2.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

            cbar_ax2.tick_params(labelsize=15)
        else:
            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar = fig.colorbar(im_track, ax=ax, format=formatter)
            cbar.set_label('Tracked beam freq (MHz)', fontsize=18)
            cbar.ax.tick_params(axis='y', which='minor', labelsize=15)
            cbar.ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
            cbar.ax.tick_params(labelsize=15)

    return fig, ax
def plot_typeIII_sim(fig, ax, trackedtypeIII, confidence=False, showcolorbar=True, showtruesources=True, cbar_axes=[0.91, 0.1, 0.01, 0.35], marker=".",cmap="turbo", label="TOA BELLA",s=100, edgecolors="w",norm=[],zorder=1000):

    xy_vals = np.vstack(trackedtypeIII.xy_detected[0])
    xy = xy_vals/R_sun.value
    sd_vals = np.vstack(trackedtypeIII.sd_detected[0])
    sd = sd_vals/R_sun.value
    xy_vals_true = np.vstack(trackedtypeIII.xy_true)
    xy_true = xy_vals_true/R_sun.value

    if norm == []:
        norm = matplotlib.colors.LogNorm()

    # Access the colormap function dynamically
    cmap_func = getattr(cm, cmap)
    # Generate colors
    colors = cmap_func(np.linspace(0, 1.0, len(xy)))

    # colors = cm.turbo(list(np.linspace(0,1.0,len(xy))))
    if confidence is True:
        i = 0
        ell_track_uncertainty = matplotlib.patches.Ellipse(xy=(xy[i, 0], xy[i, 1]),
                                                           width=2*sd[i, 0], height=2*sd[i, 1],
                                                           angle=0., edgecolor=colors[i], lw=1.5)
        ax.add_patch(ell_track_uncertainty)
        ell_track_uncertainty.set_facecolor('none')

        for i in range(1, len(xy)):
            ell_track_uncertainty = matplotlib.patches.Ellipse(xy=(xy[i,0], xy[i,1]),
                      width=2*sd[i,0], height=2*sd[i,1],
                      angle=0.,edgecolor=colors[i], lw=1.5)

            ell_track_uncertainty.set_facecolor('none')


            ax.add_patch(ell_track_uncertainty)

    if showtruesources:
        ax.scatter(xy_true[:,0], xy_true[:,1], cmap=cmap, marker="o", edgecolors=edgecolors, c="yellow", s=s, label="True Sources", norm=norm,zorder=zorder-1)
    im_track = ax.scatter(xy[:,0], xy[:,1], cmap=cmap, marker=marker, edgecolors=edgecolors, s=s, label=label, norm=norm,zorder=zorder)

    if showcolorbar is True:
        cbar_ax2 = fig.add_axes(cbar_axes) # [left, bottom, width, height]
        formatter = LogFormatter(10, labelOnlyBase=False)
        fig.colorbar(im_track, cax=cbar_ax2, format=formatter)
        cbar_ax2.set_ylabel('Tracked beam freq (MHz)', fontsize=18)

        cbar_ax2.tick_params(axis='y', which='minor',labelsize=15)
        cbar_ax2.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

        cbar_ax2.tick_params(labelsize=15)

    return fig, ax

def plot_parker_spiral(fig, ax, v_sw=400,phi_sw=0, omega=2.662e-6, theta=0, **kwargs):
    # PARKER SPIRAL
    phi0 = phi_sw
    parkerphi = []
    parkerend = 600
    for r in range(0, parkerend):
        parkerphi.append(parkerSpiral(r, phi0, v_sw=v_sw, omega=omega, theta=theta))
    x_parker, y_parker = pol2cart(np.arange(0, parkerend), parkerphi)

    ax.plot(x_parker,y_parker, **kwargs)
    return fig, ax


def loaddata(filenamef):
    import pickle

    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)

    xmapaxis = results[0]
    ymapaxis = results[1]
    delta_obs = results[2]
    stations = results[3]
    inp.close()
    return xmapaxis, ymapaxis, delta_obs, stations

def savepickle(results, filename):
    import pickle
    with open(filename, 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()



def loadtrackedtypeiii(filenamef):
    import pickle
    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)
    inp.close()
    return results

def loadpickle(filenamef):
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
    r_sun2km = R_sun.value/1000     #
    r = r * r_sun2km
    b=v_sw/(omega*np.sin(theta))
    r0= 1.0*r_sun2km
    buff = 1/b*(r-r0)

    phi = phi0 - buff
    return phi

def archimedeanSpiral(r,phi0,v_sw=400, omega=2.662e-6, theta=0):
    # http://www.physics.usyd.edu.au/~cairns/teaching/2010/lecture8_2010.pdf page 6
    # r-r0 = -(v_sw/(omega*sin(theta)))(phi(r)*phi0)
    phi0 = np.radians(phi0)
    theta = np.radians(theta+90)
    r_sun2km = R_sun.value/1000     #
    r = r * r_sun2km
    b=v_sw/(omega*np.sin(theta))
    r0= 0
    buff = 1/b*(r-r0)

    phi = phi0 - buff
    return phi


def fit_parkerSpiral(r,phi0=0,v_sw=400, sol_rot=24.47):
    # http://www.physics.usyd.edu.au/~cairns/teaching/2010/lecture8_2010.pdf page 6
    # r-r0 = -(v_sw/(omega*sin(theta)))(phi(r)*phi0)
    omega=(2. * np.pi) / (sol_rot * (24 * 3600))
    theta = 0
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

def parker_streamline(phi_0=0.0,sol_rot=24.47, v_sw=360e3,r0=1.0,rmax = 315,sampling = 100, coord = "cartesian" ):
    # Output r :: Rs, phi :: deg
    w_s = (2. * np.pi) / (sol_rot * (24 * 3600))
    r = np.linspace(r0,rmax,sampling)*u.R_sun
    r0 = r0*u.R_sun
    phi = np.degrees(-w_s/v_sw*(r.to("m").value-r0.to("m").value)) + phi_0
    if coord == "polar":
        return r.value, phi
    elif coord == "cartesian":
        return np.cos(phi* u.deg) * r.value, np.sin(phi * u.deg) * r.value
    else:
        print("specify coordinate system. coord = 'polar' || coord = 'cartesian'")

def find_intersection_point(sc1, angle1_degrees, sc2, angle2_degrees):
    # Convert angles to radians
    angle1_radians = math.radians(angle1_degrees)
    angle2_radians = math.radians(angle2_degrees)

    x1, y1 = sc1
    x2, y2 = sc2

    # Calculate slopes
    m1 = math.tan(angle1_radians)
    m2 = math.tan(angle2_radians)

    # Calculate y-intercepts
    b1 = y1 - m1 * x1
    b2 = y2 - m2 * x2

    # Calculate intersection point
    x_intersection = (b2 - b1) / (m1 - m2)
    y_intersection = m1 * x_intersection + b1

    return x_intersection, y_intersection

def find_near_dt_idx(datetime_array, target_datetime):
    return min(range(len(datetime_array)), key=lambda i: abs(datetime_array[i] - target_datetime))

def find_nearest_value(array, target_value):
    """
    Find the nearest value in the array to the target value.

    Parameters:
    - array: NumPy array
    - target_value: Value to find the nearest match

    Returns:
    - nearest_value: Nearest value in the array
    """

    # Find the index of the nearest value
    index_nearest = np.abs(array - target_value).argmin()

    # Get the nearest value
    nearest_value = array[index_nearest]

    return nearest_value



class typeIII:
  def __init__(self, freq, xy, sd):
    self.freq = freq
    self.xy = xy
    self.sd = sd

class sim_burst:
  def __init__(self, xy_true, xy_detected, sd_detected):
    self.xy_true = xy_true
    self.xy_detected = xy_detected
    self.sd_detected = sd_detected




if __name__ == "__main__":

    ncores_cpu = multiprocessing.cpu_count()

    showbgmap = True
    showtracked = True
    showscatter = True
    showparkerspirals = True
    savefigs = True

    # generate some test data
    # np.random.seed(1)
    day = 7
    month = 6
    year = 2012
    date_str = f"{year}_{month:02d}_{day:02d}"
    # date_str = f"surround"

    sol_rot = 24.47#27.2753
    omega = 2*3.1416/(sol_rot*24*60*60)


    solarsystem = solarmap.get_sc_coord(date=[year, month, day], objects=["stereo_b", "stereo_a", "earth"])
    stations_rsun = np.array(solarsystem.locate_simple())

    spacecraft = ["stereo_b", "stereo_a", "wind"]  # LABELS


    N_STATIONS = len(stations_rsun)
    stations = stations_rsun*R_sun.value

    # Making grid
    xrange = [-250,250]
    xres = 10
    yrange = [-250, 250]
    yres = xres
    xmapaxis = np.arange(xrange[0], xrange[1], xres)
    ymapaxis = np.arange(yrange[0], yrange[1], yres)
    cadence = 60 # for the filename


    # FILENAMES
    # BELLA MAP
    filename_BELLA_MAP = "./Data/2012_06_07/results_-250_250_-250_250_10_10_3stations_60s.pkl"
    # TRACKED TYPE III
    trackedfile = "./Data/TRACKING_2012_06_07_results_3stations_LE_Freqs_0.15_2_HR.pkl"
    trackedfile_scatter = "./Data/TRACKING_2012_06_07_results_3stations_LE_Freqs_0.15_2_HR_SCATTER.pkl"


    # Figures
    fname = f"/bayes_positioner_map_median_tracked_{xrange[0]}_{xrange[-1]}_{yrange[0]}_{yrange[-1]}_{xres}_{yres}_{N_STATIONS}.jpg"




    """  ------------------------ Load BELLA MAP Data ----------------------------------   """
    if showbgmap:
        xmapaxis, ymapaxis, delta_obs, stations_rsun = loaddata(filename_BELLA_MAP)
        """  ---------------------------------------------------------------------   """
        # """  ------------------------ BELLA MAP PLOT W/ MEDIAN FILTER ---------------------------   """
        # from scipy.ndimage import median_filter as medfil
        delta_obs2 = delta_obs*2
        median_filter_image = medfil(delta_obs2, size=(6,6))

    if showtracked:
        tracked = loadtrackedtypeiii(trackedfile)
        trackedtypeIII = typeIII(tracked[:, 0], tracked[:, 1], tracked[:, 2])
        timestracked = tracked[:, 3]

    if showscatter:
        tracked_scatter = loadtrackedtypeiii(trackedfile_scatter)
        trackedtypeIII_scatter = typeIII(tracked_scatter[:,0], tracked_scatter[:,1],tracked_scatter[:,2])
        timestracked_scatter = tracked_scatter[:, 3]


    # BELLA PLOTTER
    # UNCERTAINTY MAP
    if showbgmap:
        delta_obs2 = delta_obs*2
        median_filter_image = medfil(delta_obs2, size=(6,6))

    fig, ax = plt.subplots(1,1,figsize=(11,11))
    plt.subplots_adjust(top=1, bottom=0)
    if showbgmap:
        fig, ax = plot_bella_map(fig,ax, median_filter_image, xmapaxis, ymapaxis, stations_rsun,
                                 vmin=np.min(median_filter_image), vmax=np.max(median_filter_image),
                                 date_str=date_str, spacecraft_labels=["stereo_b", "stereo_a", "wind"])

        ax.contour(xmapaxis, ymapaxis,median_filter_image,[10],  colors='black')


    # TRACKED TYPE III
    if showtracked:
        fig, ax = plot_tracked_typeIII(fig, ax, trackedtypeIII, confidence=True )

    if showscatter:
        fig, ax = plot_tracked_typeIII(fig, ax, trackedtypeIII_scatter, confidence=True, showcolorbar=False)

    # PARKER SPIRALS
    # fig, ax = plot_parker_spiral(fig, ax, v_sw=310, phi_sw=30)
    if showparkerspirals:
        pp = 30
        fig, ax = plot_parker_spiral(fig, ax, v_sw=400, phi_sw=pp, omega=omega, color='black', linestyle="-.")
        fig, ax = plot_parker_spiral(fig, ax, v_sw=380, phi_sw=pp, omega=omega, color='black', linestyle="-")
        fig, ax = plot_parker_spiral(fig, ax, v_sw=360, phi_sw=pp, omega=omega, color='black', linestyle="-.")
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
        dir = mkdirectory("./Figures/")
        if trackedfile[-11:-4] != 'SCATTER':
            plt.savefig(dir+'BELLA_map0.png', dpi=300)
        else:
            plt.savefig(dir+'BELLA_map0_SCATTER.png', dpi=300)
        #
        # ax.get_legend().remove()
        # xmin, xmax = -5, 155
        # ymin, ymax = -40, 40
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        # plt.draw()
        # if trackedfile[-11:-4] != 'SCATTER':
        #     plt.savefig('BELLA_map1.png', dpi=300)
        # else:
        #     plt.savefig('BELLA_map1_SCATTER.png', dpi=300)




    #
    # # BELLA PLOTTER
    # # UNCERTAINTY MAP
    # delta_obs2 = delta_obs*2
    # median_filter_image = medfil(delta_obs2, size=(6,6))
    #
    # fig = plt.figure(figsize=(21,10))#layout="constrained",
    # ax = fig.subplot_mosaic("AB;AC")
    # plt.subplots_adjust(left=0.14, bottom=0.11, right=0.916, top=0.856, wspace=0.151, hspace=0.231)
    #
    #
    # fig, ax['A'] = plot_bella_map(fig,ax['A'], median_filter_image, xmapaxis, ymapaxis, stations_rsun,
    #                          vmin=np.min(median_filter_image), vmax=np.max(median_filter_image),
    #                          date_str=date_str, spacecraft_labels=["stereo_b", "stereo_a", "wind"], showcolorbar=True, cbar_axes=[0.02,0.2, 0.01, 0.6]) #[left, bottom, width, height]
    #
    # fig, ax['B'] = plot_bella_map(fig,ax['B'], median_filter_image, xmapaxis, ymapaxis, stations_rsun,
    #                          vmin=np.min(median_filter_image), vmax=np.max(median_filter_image),
    #                          date_str=date_str, spacecraft_labels=["stereo_b", "stereo_a", "wind"], showlegend=False, showcolorbar=False)
    #
    # fig, ax['C'] = plot_bella_map(fig,ax['C'], median_filter_image, xmapaxis, ymapaxis, stations_rsun,
    #                          vmin=np.min(median_filter_image), vmax=np.max(median_filter_image),
    #                          date_str=date_str, spacecraft_labels=["stereo_b", "stereo_a", "wind"], showlegend=False, showcolorbar=False)
    #
    #
    #
    #
    # fig, ax['A'] = plot_tracked_typeIII(fig, ax['A'], trackedtypeIII, confidence=True, showcolorbar=True, cbar_axes=[0.92,0.2, 0.01, 0.6]) #[left, bottom, width, height]
    # fig, ax['B'] = plot_tracked_typeIII(fig, ax['B'], trackedtypeIII, confidence=True, showcolorbar=False  )
    # fig, ax['C'] = plot_tracked_typeIII(fig, ax['C'], trackedtypeIII_scatter, confidence=True, showcolorbar=False  )
    #
    #
    # # fig, ax = plot_tracked_typeIII(fig, ax, trackedtypeIII_scatter, confidence=True )
    # # fig, ax = plot_tracked_typeIII(fig, ax, trackedtypeIIIHR, confidence=True )
    #
    # # PARKER SPIRALS
    # # fig, ax = plot_parker_spiral(fig, ax, v_sw=310, phi_sw=30)
    # pp = 30
    # vsw = 400
    # dvsw = 20
    # fig, ax['A'] = plot_parker_spiral(fig, ax['A'], v_sw=vsw+dvsw, phi_sw=pp, omega=omega, color='black', linestyle="-.")
    # fig, ax['A'] = plot_parker_spiral(fig, ax['A'], v_sw=vsw, phi_sw=pp, omega=omega, color='black', linestyle="-")
    # fig, ax['A'] = plot_parker_spiral(fig, ax['A'], v_sw=vsw-dvsw, phi_sw=pp, omega=omega, color='black', linestyle="-.")
    #
    # fig, ax['B'] = plot_parker_spiral(fig, ax['B'], v_sw=vsw+dvsw, phi_sw=pp, omega=omega, color='black', linestyle="-.")
    # fig, ax['B'] = plot_parker_spiral(fig, ax['B'], v_sw=vsw, phi_sw=pp, omega=omega, color='black', linestyle="-")
    # fig, ax['B'] = plot_parker_spiral(fig, ax['B'], v_sw=vsw-dvsw, phi_sw=pp, omega=omega, color='black', linestyle="-.")
    #
    # fig, ax['C'] = plot_parker_spiral(fig, ax['C'], v_sw=vsw+dvsw, phi_sw=pp, omega=omega, color='black', linestyle="-.")
    # fig, ax['C'] = plot_parker_spiral(fig, ax['C'], v_sw=vsw, phi_sw=pp, omega=omega, color='black', linestyle="-")
    # fig, ax['C'] = plot_parker_spiral(fig, ax['C'], v_sw=vsw-dvsw, phi_sw=pp, omega=omega, color='black', linestyle="-.")
    #
    # # REFRESH LEGEND
    # legend = ax['A'].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True, ncol=2, fontsize=18)
    # data1_handle = legend.legend_handles[3]
    # data1_handle.set_sizes([300])
    # data1_handle = legend.legend_handles[4]
    # data1_handle.set_sizes([300])
    # # fig.subplots_adjust(left=0.098, bottom=0.080, right=0.9, top=0.845, wspace=0.2, hspace=0.2)
    # ax['A'].tick_params(axis='both', which='major', labelsize=18)
    #
    # xmin, xmax = -5, 155
    # ymin, ymax = -40, 40
    # ax['B'].set_xlim(xmin, xmax)
    # ax['B'].set_ylim(ymin, ymax)
    # ax['C'].set_xlim(xmin, xmax)
    # ax['C'].set_ylim(ymin, ymax)
    #
    #
    # ax['A'].text(200, 210, '(a)', horizontalalignment='center', verticalalignment='center', c="white")
    # ax['B'].text(xmax-10,ymax-10, '(b)', horizontalalignment='center', verticalalignment='center', c="white")
    # ax['C'].text(xmax-10,ymax-10,  '(c)', horizontalalignment='center', verticalalignment='center', c="white")
    #
    #
    # plt.show(block = False)
    #
    # savefigs = True
    # if savefigs is True:
    #     plt.savefig('/Users/canizares/OneDrive/Work/0_PhD/Projects/2012_06_07_WSTASTB/BELLA_map_mosaic.png', dpi=300)
    #
    #
