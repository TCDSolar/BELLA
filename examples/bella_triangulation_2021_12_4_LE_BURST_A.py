# caution: path[0] is reserved for script path (or '' in REPL)

# Point to the multilaterate directory if running from different directory
#sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA/Scripts/multilaterate')

# Local imports
# from bayesian_tracker import *
# from bayes_positioner import *

import argparse
import datetime
import datetime as dt
# Parallel processing imports
import multiprocessing

import sys
sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA')

import bella.multilaterate.bayes_positioner as bp
import bella.multilaterate.bayesian_tracker as btrack

import numpy as np
# Third Party imports
# BAYESIAN IMPORTS
import pymc3 as pm
# General
import solarmap

# Standard Library imports
from astropy.constants import R_sun, c

    # if "wind" in spacecraft:
    # if "stereo_a" in spacecraft:
    # if "stereo_b" in spacecraft:
    # if "solo" in spacecraft:
    # if "psp" in spacecraft:
    # if "mex" in spacecraft:

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process spacecraft names.')
    parser.add_argument('--spacecraft', nargs='+', help='Spacecraft name(s)')
    parser.add_argument('--profile', type=str, help='Profile type (e.g., "LE" or "BB")')
    parser.add_argument('--freqs', type=float, nargs=2, help='Freqs start end ')
    parser.add_argument('--note', type=str, help='Custom note')

    # usage:
    # python bella_triangulation_2021_12_4_LE_BURST_A.py --spacecraft wind stereo_a solo psp --profile LE --note BURST_A



    # Parse command line arguments
    args = parser.parse_args()

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    string =f'Running on PyMC3 v{pm.__version__}'
    print(string)
    # Date of the observation
    day = 4
    month = 12
    year = 2021
    observation_start = "1300"  # needed for the data file
    date_str = f"{year}_{month:02d}_{day:02d}"


    # spacecraft = ["stereo_a", "solo", "psp", "mex"]
    spacecraft = []
    if not args.spacecraft:
        # Manually inputing Spacecraft:
        windsc = True
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
        else:
            windsc=False

        if "stereo_a" in spacecraft:
            steasc=True
        else:
            steasc=False

        if "stereo_b" in spacecraft:
            stebsc=True
        else:
            stebsc=False

        if "solo" in spacecraft:
            solosc=True
        else:
            solosc=False

        if "psp" in spacecraft:
            pspsc=True
        else:
            pspsc=False

        if "mex" in spacecraft:
            mexsc=True
        else:
            mexsc=False


        print(f"Spacecraft selected: {spacecraft}")

    if not args.freqs:
        freqlimmin = 0.5
        freqlimmax = 3
    else:
        f_min = np.min(args.freqs)
        f_max = np.max(args.freqs)
        if f_min.is_integer():
            freqlimmin = int(f_min)
        else:
            freqlimmin = f_min
        if f_max.is_integer():
            freqlimmax = int(f_max)
        else:
            freqlimmax = f_max


    if not args.profile:
        profile = "LE"     # BB - Backbone or LE leading edge or TE trailing edge
    else:
        profile = args.profile

    if not args.note:
        note = 'BURST_A'   # manual default
    else:
        note = args.note   # leave empty "" for fitted curve.

    # REPLACE these with your data file
    typeiiidir = f'./Data/TypeIII/{year}_{month:02}_{day:02}/'
    typeiiifile = typeiiidir + f"typeIII_{year}{month:02}{day:02}_{observation_start}"

    sc_str = ""
    if "wind" in spacecraft:
        sc_str += "_WIND"
    if "stereo_a" in spacecraft:
        sc_str += "_STEA"
    if "stereo_b" in spacecraft:
        sc_str += "_STEB"
    if "solo" in spacecraft:
        sc_str += "_SOLO"
    if "psp" in spacecraft:
        sc_str += "_PSP"
    if "mex" in spacecraft:
        sc_str += "_MEX"


    typeiiifile += f"{sc_str}_Freqs_{freqlimmin}_{freqlimmax}_{profile}"
    if note == "":
        typeiiifile += ".pkl"
    else:
        typeiiifile += f"_{note}.pkl"


    #DATA
    #sys.path.insert(1, '../2020_07_11_SOLO_PSP_STEREOA_WIND')
    # np.random.seed(1)
    ncores_cpu = multiprocessing.cpu_count()


    solarsystem = solarmap.get_HEE_coord(date=[year, month, day], objects=spacecraft, orbitlength=-1)
    st_buff = solarsystem.locate()
    if "wind" in spacecraft:
        wind_coord = st_buff['wind'][:2].reshape(2)
    if "stereo_a" in spacecraft:
        stereo_a_coord = st_buff['STEREO-A'][:2].reshape(2)
    if "stereo_b" in spacecraft:
        stereo_b_coord = st_buff['STEREO-B'][:2].reshape(2)
    if "solo" in spacecraft:
        solo_coord = st_buff['solo'][:2].reshape(2)
    if "psp" in spacecraft:
        psp_coord = st_buff['psp'][:2].reshape(2)
    if "mex" in spacecraft:
        mex_coord = st_buff['Mars Express (spacecraft)'][:2].reshape(2)

    # stations_rsun = np.array([solo_coord, psp_coord, stereo_coord, wind_coord]) # in order wind, stereoa, psp, solo
    # stations_rsun = np.array([wind_coord, stereo_coord, psp_coord, solo_coord])
    stations_rsun = []
    for sc in spacecraft:
        if sc =="wind":
            stations_rsun.append(wind_coord)
        elif sc == "stereo_a":
            stations_rsun.append(stereo_a_coord)
        elif sc == "stereo_b":
            stations_rsun.append(stereo_b_coord)
        elif sc =="solo":
            stations_rsun.append(solo_coord)
        elif sc =="psp":
            stations_rsun.append(psp_coord)
        elif sc =="mex":
            stations_rsun.append(mex_coord)

    stations_rsun = np.array(stations_rsun)

    N_STATIONS = len(stations_rsun)
    stations = stations_rsun*R_sun.value


    typeIII = btrack.loadtypeiii(typeiiifile, spacecraft)

    print(f"\nfile for triangulation:\n{typeiiifile}\n")
    typeIII_times = {}
    i = 0
    t0_norm = dt.datetime(year,month,day, 0,0,0)
    for each in typeIII['freqs']:
        if "wind" in spacecraft:
            windnorm = typeIII['wind'][i] - t0_norm
        if "stereo_a" in spacecraft:
            steAnorm = typeIII['stereo_a'][i] - t0_norm
        if "stereo_b" in spacecraft:
            steBnorm = typeIII['stereo_b'][i] - t0_norm
        if "solo" in spacecraft:
            solonorm = typeIII['solo'][i] - t0_norm
        if "psp" in spacecraft:
            pspnorm = typeIII['psp'][i] - t0_norm
        if "mex" in spacecraft:
            mexnorm = typeIII['mex'][i] - t0_norm

        typeIII_times[f"{each}"] = []
        for sc in spacecraft:
            if sc == "wind":
                typeIII_times[f"{each}"].append(windnorm.seconds)
            elif sc == "stereo_a":
                typeIII_times[f"{each}"].append(steAnorm.seconds)
            elif sc == "stereo_b":
                typeIII_times[f"{each}"].append(steBnorm.seconds)
            elif sc == "solo":
                typeIII_times[f"{each}"].append(solonorm.seconds)
            elif sc == "psp":
                typeIII_times[f"{each}"].append(pspnorm.seconds)
            elif sc == "mex":
                typeIII_times[f"{each}"].append(mexnorm.seconds)
        i = i + 1

        # CHECK if DATA and SPACECRAFT COORDINATES are input in same order
        # This bug has been fixed but just in case ill leave this here.
        # ordercheck = []
        # for tt in typeIII_times[f"{each}"]:
        #     if "wind" in spacecraft:
        #         if tt == windnorm.seconds:
        #             ordercheck.append("wind")
        #     if "stereo_a" in spacecraft:
        #         if tt == steAnorm.seconds:
        #             ordercheck.append("stereo_a")
        #     if "stereo_b" in spacecraft:
        #         if tt == steBnorm.seconds:
        #             ordercheck.append("stereo_b")
        #     if "solo" in spacecraft:
        #         if tt == solonorm.seconds:
        #             ordercheck.append("solo")
        #     if "psp" in spacecraft:
        #         if tt == pspnorm.seconds:
        #             ordercheck.append("psp")
        #     if "mex" in spacecraft:
        #         if tt == mexnorm.seconds:
        #             ordercheck.append("mex")
        # if spacecraft != ordercheck:
        #     raise ValueError("ERROR: SPACECRAFT NOT IN CORRECT ORDER. ")

    """  ---------------------------------------------------------------------   """
    """  --------------------------- ON/OFF ----------------------------------   """
    """  ---------------------------------------------------------------------   """
    runtracker = True

    """  ---------------------------------------------------------------------   """
    """  ------------------------- Tracking ----------------------------------   """
    """  ---------------------------------------------------------------------   """
    results = []
    count = 0
    start_time = dt.datetime.now()
    dtloop = []

    # TEST = list(["3.0"])#"1.0020202020202018",
    for i_freq in list(typeIII_times):
        # for i_freq in TEST:
        print("")
        currentloop = f"""Freq : {i_freq} MHz \n loop : {count}/{len(typeIII_times)} """
        print(currentloop)

        if dtloop != []:
            print(f"Elapsed time       : {dt.datetime.now()-start_time}")
            print(f"Estimated time left: {dtloop*(len(typeIII_times)-count)} ")
        count += 1

        tloop0 = dt.datetime.now()
        check = False
        while check is False:
            try:
                # connect
                mu, sd, t1_pred, trace, summary, t_emission_fromtrace, v_analysis = bp.triangulate(stations, typeIII_times[i_freq], t_cadence=60, v_sd=0.0001*c.value, cores=4, progressbar=True, report=0, plot=0,traceplot=True, savetraceplot=True, traceplotdir=f'{sc_str}_{profile}_{note}', traceplotfn=f'{i_freq}.jpg')
                check = True
            except Exception as e:
                print(f"ERROR: {e}, try again")


        tloop1 = dt.datetime.now()
        dtloop = tloop1-tloop0
        xy = mu[0] / R_sun.value, mu[1] / R_sun.value

        # TEST that location is correct
        xx, yy = xy[0], xy[1]  # TEST times
        x_true = np.array([xx * R_sun.value, yy * R_sun.value])  # "true" (observed) source position (m)
        v_true = summary["mean"]["v"]#c.value  #  speed of light (m/s)
        d_true = np.linalg.norm(stations - x_true, axis=1)
        t1_true = d_true / v_true  # true time of flight values
        t_obs = t1_true


        print(f"observed position: {xy}")
        print(f"Observation      : {t_obs}")
        print(f"Data             : {typeIII_times[i_freq]}")
        print(f"Difference t     : {np.subtract(t_obs, typeIII_times[i_freq])} ")



        t0 = np.array(typeIII_times[i_freq]) - t1_pred
        t_emission = [t0_norm + dt.timedelta(seconds=(t0[0])), t0_norm + dt.timedelta(seconds=(t0[1])), t0_norm + dt.timedelta(seconds=(t0[2]))]
        t_emission_averages = t0_norm + dt.timedelta(seconds=(t0.mean()))
        t_emission_maxmindiff = t0.max() - t0.min()


        results.append([float(i_freq), mu,sd, t1_pred, t_emission, t_emission_averages, t_emission_maxmindiff, trace, summary])


    results = np.array(results)


    #
    # i = 0
    # t_emission = []
    # t_emission_averages = []
    # t_emission_maxmindiff = []
    # for i_freq in list(typeIII_times):
    #     t1_pred = timestracked[i]
    #     i+=1
    #     t0 = np.array(typeIII_times[i_freq])-t1_pred
    #     t_emission.append([t0_norm + dt.timedelta(seconds=( t0[0])),t0_norm + dt.timedelta(seconds=( t0[1])),t0_norm + dt.timedelta(seconds=( t0[2]))])
    #     t_emission_averages.append(t0_norm + dt.timedelta(seconds=( t0.mean())))
    #     t_emission_maxmindiff.append(t0.max()-t0.min())


    if note == "":
        btrack.savetrackedtypeiii(results, date_str=date_str, N_STATIONS=N_STATIONS,profile=profile, dir=f"Data/Multilaterated/{date_str}/", title=f"TRACKING_{date_str}_results_{N_STATIONS}stations{sc_str}_{profile}_Freqs_{freqlimmin}_{freqlimmax}.pkl")
    else:
        btrack.savetrackedtypeiii(results, date_str=date_str, N_STATIONS=N_STATIONS,profile=profile, dir=f"Data/Multilaterated/{date_str}/", title=f"TRACKING_{date_str}_results_{N_STATIONS}stations{sc_str}_{profile}_Freqs_{freqlimmin}_{freqlimmax}_{note}.pkl")

    end_time = datetime.datetime.now()


    # string = f"""
    # Process finished:
    # -------------------------------
    # Start time: {start_time}
    # End time  : {end_time}
    # Total time: {end_time-start_time}
    #
    #
    # stations_rsun = {stations_rsun}
    # t_obs      =    {typeIII_times[f"{each}"]}
    #
    # order = {spacecraft}
    # check = {ordercheck}
    #
    #
    #     """
    # print(string)
    #



    # stations_rsun = array([[ 216.3864883 ,   -0.82640064], [  76.26846667, -192.1493994 ],[ 143.14674755,   62.28368691],[ -38.28536325,  125.55611214]])
    # t_1MHz = [658, 621, 568, 480]
    # t_3MHz  = [627, 540, 559, 480]
