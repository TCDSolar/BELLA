
# Standard Library imports
import os
# General
import logging
import datetime as dt
# Parallel processing imports
import multiprocessing
from math import sqrt, radians
from contextlib import contextmanager

# Third Party imports
# BAYESIAN IMPORTS
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import solarmap
from scipy.ndimage import median_filter

from astropy.constants import R_sun, au, c

# Local imports
import sys
sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA')
from bella.multilaterate.bayes_positioner import triangulate


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
    return dir

def parallel_pos_map(x1,y1,stations,xrange,yrange,xres,yres, cores=1, traceplotsave=False, figdir="./traceplots", date_str="date"):
    tloop0 = dt.datetime.now()
    try:
        currentloop = f"""x, y : {x1},{y1}"""
        print(currentloop)

        x_true = np.array([x1 * R_sun.value, y1 * R_sun.value])  # true source position (m)
        v_true = c.value  # speed of light (m/s)
        d_true = np.linalg.norm(stations - x_true, axis=1)
        t1_true = d_true / v_true  # true time of flight values
        # t_obs = t1_true-t0_true# true time difference of arrival values
        t_obs = t1_true
        np.random.seed(1)
        # t_obs = t_obs+0.05*np.random.randn(*t_obs.shape)# noisy observations

        # Make sure to use cores=1 when using parallel loops. cores in triangulate function refers to number of cores
        # used by the MCMC solver.
        mu, sd, t1_pred, trace, summary = triangulate(stations, t_obs, cores=1, progressbar=False, report=0, plot=0)

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
        xy = mu[0] / R_sun.value, mu[1] / R_sun.value

        delta_obs = sqrt((xy[0] - x_true[0]) ** 2 + (xy[1] - x_true[1]) ** 2)
        print(f"delta_obs: {delta_obs}")
        res = np.array([delta_obs, np.nan, np.nan])
        tloop1 = dt.datetime.now()


    except Exception as e:
        print(f"Error at P({x1},{y1}): {str(e)}")
        res = np.array([600, x1, y1])  # Fallback error handling
        tloop1 = dt.datetime.now()

    tloopinseconds = (tloop1 - tloop0).total_seconds()
    print(f"Time Loop : {tloop1 - tloop0}   :   {tloopinseconds}s   ")

    return res


def parallel_tracker(freq,t_obs, stations ):
    tloop0 = dt.datetime.now()
    print("parallel_tracker(freq,t_obs, stations )")
    print(f"parallel_tracker({freq},"
          f"{t_obs}, "
          f"{stations} )")
    try:
        currentloop = f"""Freq : {freq} MHz"""
        print(currentloop)

        mu, sd, t1_pred, trace, summary = triangulate(stations, t_obs, cores=1, progressbar=False, report=0, plot=0)


        xy = mu[0] / R_sun.value, mu[1] / R_sun.value

        print(f"observed position: {xy}")
        res = np.array([xy, np.nan, np.nan])

        tloop1 = dt.datetime.now()
    except Exception as e:
        xy = 0
        print(f"ERROR: {e}")
        print(f"SIM FAILED at Freq = {freq} MHz, SKIPPED")
        res = np.array([xy, freq])
        tloop1 = dt.datetime.now()



    tloopinseconds = (tloop1 - tloop0).total_seconds()
    print(f"Time Loop : {tloop1 - tloop0}   :   {tloopinseconds}s   ")



    return res


def plot_map_simple(delta_obs, xmapaxis, ymapaxis, stations,vmin=0,vmax=30, savefigure=False, showfigure=True, title="",figdir="./MapFigures", date_str="date"):
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

    ax.set_xlabel(r"'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel(r"'HEE - Y / $R_{\odot}$'", fontsize=22)
    ax.set_title(title, fontsize=22)

    if savefigure is True:
        figdir = f"{figdir}/{date_str}"
        mkdirectory(figdir)
        plt.savefig(figdir+f"/bayes_positioner_map_{xmapaxis[0]}_{xmapaxis[-1]}_{ymapaxis[0]}_{ymapaxis[-1]}_{xres}_{yres}_{N_STATIONS}.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)

    if showfigure is True:
        plt.show(block=False)
    else:
        plt.close(fig)

def savedata(delta_obs, xmapaxis, ymapaxis, stations,dir="./Data", date_str="date"):
    import pickle
    mkdirectory(dir+f"/{date_str}")


    xres = xmapaxis[1]-xmapaxis[0]
    yres = ymapaxis[1]-ymapaxis[0]
    N_STATIONS = len(stations)

    results  = [xmapaxis, ymapaxis, delta_obs, stations]
    with open(dir+f"/{date_str}"+f'/results_{xmapaxis[0]}_{xmapaxis[-1]}_{ymapaxis[0]}_{ymapaxis[-1]}_{xres}_{yres}_{N_STATIONS}stations.pkl', 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)

def savetrackedtypeiii(results, dir="./Data/", date_str="date", profile="PROFILE",N_STATIONS=0, title=""):
    import pickle

    if title=="":
       title = f"TRACKING_{date_str}_results_{N_STATIONS}stations_{profile}.pkl"


    mkdirectory(dir)

    stringpath = dir+f'{title}'
    with open(stringpath, 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)

    print(f"Results saved to:  {stringpath}")


def loaddata(filenamef):
    import pickle

    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)

    xmapaxis = results[0]
    ymapaxis = results[1]
    delta_obs = results[2]
    stations = results[3]

    return xmapaxis, ymapaxis, delta_obs, stations

def loadtypeiii(filenamef, spacecraft):
    import pickle

    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)

    typeIII = {}
    freqs = results["Freqs"]
    loadedtxt = 'freqs'
    typeIII['freqs'] = freqs
    if 'wind' in spacecraft:
        windtypeiii = results["WindTime"]
        typeIII['wind'] = windtypeiii
        loadedtxt = loadedtxt + ', wind'

    if 'stereo_a' in spacecraft:
        stetypeiii = results['StereoATime']
        typeIII['stereo_a'] = stetypeiii
        loadedtxt = loadedtxt + ', stereo a'

    if 'stereo_b' in spacecraft:
        stetypeiii = results['StereoBTime']
        typeIII['stereo_b'] = stetypeiii
        loadedtxt = loadedtxt + ', stereo b'

    if 'solo' in spacecraft:
        solotypeiii = results['SoloTime']
        typeIII['solo'] = solotypeiii
        loadedtxt = loadedtxt + ', solo'

    if 'psp' in spacecraft:
        psptypeiii = results['PSPTime']
        typeIII['psp'] = psptypeiii
        loadedtxt = loadedtxt + ', psp'

    if 'mex' in spacecraft:
        mextypeiii = results['MEXTime']
        typeIII['mex'] = mextypeiii
        loadedtxt = loadedtxt + ', mex'

    print(f"loaded type iii data from {loadedtxt} ")

    return typeIII

def loadtrackedtypeiii(filenamef):
    import pickle
    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)
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



if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    string =f'Running on PyMC3 v{pm.__version__}'
    print(string)

    #DATA
    #sys.path.insert(1, '../2020_07_11_SOLO_PSP_STEREOA_WIND')

    ncores_cpu = multiprocessing.cpu_count()

    # Date of the observation
    day = 11
    month = 7
    year = 2020
    date_str = f"{year}_{month:02d}_{day:02d}"
    spacecraft = ["wind", "stereo", "psp", "solo"]


    mkdirectory("./logs")
    logging.basicConfig(filename=f'logs/sim_{date_str}.log', level=logging.INFO)
    logging.info(string)
    logging.info(f'Date of Observation: {date_str}')



    if date_str == "surround":
        # SURROUND
        #############################################################################
        L1 = [0.99*(au/R_sun),0]
        L4 = [(au/R_sun)*np.cos(radians(60)),(au/R_sun)*np.sin(radians(60))]
        L5 = [(au/R_sun)*np.cos(radians(60)),-(au/R_sun)*np.sin(radians(60))]
        ahead = [0, (au/R_sun)]
        behind = [0, -(au/R_sun)]

        stations_rsun = np.array([L1, L4, L5, ahead, behind])
        #############################################################################
    elif date_str == "test":
        stations_rsun = np.array([[200, 200], [-200, -200], [-200, 200], [200, -200]])

    elif date_str == "manual":
        stations_rsun = np.array([[45.27337378, 9.90422281],[-24.42715218,-206.46280171],[ 212.88183411,0.]])
        date_str = f"{year}_{month:02d}_{day:02d}"
    else:
        solarsystem = solarmap.get_sc_coord(date=[year, month, day], objects=["psp", "stereo_a", "wind", "solo"])
        stations_rsun = np.array(solarsystem.locate_simple())



    N_STATIONS = len(stations_rsun)
    stations = stations_rsun*R_sun.value




    """  ---------------------------------------------------------------------   """
    """  ---------------------- LOAD TYPE III --------------------------------   """
    """  ---------------------------------------------------------------------   """
    typeiiifile = "../2020_07_11_SOLO_PSP_STEREOA_WIND/typeIII_2020_7_21_2_50_WIND_STEREO_PSP_SOLO.pkl"
    typeIII = loadtypeiii(typeiiifile, spacecraft)

    typeIII_times = {}
    i = 0
    t0_norm = dt.datetime(year,month,day, 0,0,0)
    for each in typeIII['freqs']:
        windnorm = typeIII['wind'][i] - t0_norm
        stenorm = typeIII['stereo'][i] - t0_norm
        pspnorm = typeIII['psp'][i] - t0_norm
        solonorm = typeIII['solo'][i] - t0_norm

        typeIII_times[f"{each}"] = [windnorm.seconds,stenorm.seconds ,pspnorm.seconds ,solonorm.seconds ]
        i = i + 1



    """  ---------------------------------------------------------------------   """
    """  --------------------------- ON/OFF ----------------------------------   """
    """  ---------------------------------------------------------------------   """
    runtracker = True
    run_failed_again = True


    """  ---------------------------------------------------------------------   """
    """  ------------------------- Tracking ----------------------------------   """
    """  ---------------------------------------------------------------------   """
    results = []
    for i_freq in list(typeIII_times):
        currentloop = f"""Freq : {i_freq} MHz"""
        print(currentloop)

        mu, sd, t1_pred, trace, summary = triangulate(stations, typeIII_times[i_freq], cores=ncores_cpu, progressbar=False, report=0, plot=0)


        xy = mu[0] / R_sun.value, mu[1] / R_sun.value


        print(f"observed position: {xy}")
        results.append([float(i_freq), mu,sd,t1_pred,trace,summary])

        tloop1 = dt.datetime.now()

    results = np.array(results)

    savetrackedtypeiii(results, date_str=date_str)
