
# Local imports
import datetime
import datetime as dt
# Parallel processing imports
import multiprocessing

import numpy as np
# Third Party imports
# BAYESIAN IMPORTS
import pymc3 as pm
# General
import solarmap
import sys
sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA')

from bella.multilaterate.bayes_positioner import triangulate
from bella.multilaterate.bayesian_tracker import loadtypeiii, savetrackedtypeiii

# Standard Library imports
from astropy.constants import R_sun

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    string =f'Running on PyMC3 v{pm.__version__}'
    print(string)

    #DATA
    #sys.path.insert(1, '../2020_07_11_SOLO_PSP_STEREOA_WIND')
    # np.random.seed(1)
    ncores_cpu = multiprocessing.cpu_count()

    # Date of the observation
    day = 7
    month = 6
    year = 2012
    date_str = f"{year}_{month:02d}_{day:02d}"
    spacecraft = ["earth", "stereo_a","stereo_b"]  # error in jpl horizons for wind before 2019. use 0.99% distance of earth until fixed
    freqlimmin = 0.15
    freqlimmax = 2
    profile = "LE"     # BB - Backbone or LE leading edge
    note = "SCATTER"   # leave empty "" for fitted curve.

    solarsystem = solarmap.get_sc_coord(date=[year, month, day], objects=spacecraft, orbitlength=-1)
    st_buff = solarsystem.locate()
    wind_coord = st_buff['earth'][:2].reshape(2)
    wind_coord[0] = 0.99 * wind_coord[0]  # wind coordinates bug fix
    spacecraft[0] = 'wind'    # fixing the wind ephemerides error. Earth position was used to obtain the wind coords
    stereo_a_coord = st_buff['stereo_a'][:2].reshape(2)
    stereo_b_coord = st_buff['stereo_b'][:2].reshape(2)

    # stations_rsun = np.array([solo_coord, psp_coord, stereo_coord, wind_coord]) # in order wind, stereoa, psp, solo
    # stations_rsun = np.array([wind_coord, stereo_coord, psp_coord, solo_coord])
    stations_rsun = []
    for sc in spacecraft:
        if sc =="wind":
            stations_rsun.append(wind_coord)
        elif sc == "stereo_a":
            stations_rsun.append(stereo_a_coord)
        elif sc =="stereo_b":
            stations_rsun.append(stereo_b_coord)

    stations_rsun = np.array(stations_rsun)


    # stations_rsun = np.array([solo_coord,wind_coord, psp_coord, stereo_coord])

    N_STATIONS = len(stations_rsun)
    stations = stations_rsun*R_sun.value


    typeiiidir = f'../Type_III_Fitter/Data/TypeIII/{year}_{month:02}_{day:02}/'

    if note == "":
        typeiiifile = typeiiidir + f'typeIII_{year}{month:02}{day:02}_1920_WIND_STEREO_A_STEREO_B_Freqs_{freqlimmin}_{freqlimmax}_{profile}_HR.pkl'
    else:
        typeiiifile = typeiiidir + f'typeIII_{year}{month:02}{day:02}_1920_WIND_STEREO_A_STEREO_B_Freqs_{freqlimmin}_{freqlimmax}_{profile}_HR_{note}.pkl'  # USE FOR SCATTER TRIANGULATION
    # typeiiifile = typeiiidir+f'typeIII_{year}{month:02}{day:02}_1920_WIND_STEREO_A_STEREO_B_Freqs_{freqlimmin}_{freqlimmax}_{profile}_HR.pkl'


    typeIII = loadtypeiii(typeiiifile, spacecraft)
    print(f"\nfile for triangulation:\n{typeiiifile}\n")
    typeIII_times = {}
    i = 0
    t0_norm = dt.datetime(year,month,day, 0,0,0)
    for each in typeIII['freqs']:
        windnorm = typeIII['wind'][i] - t0_norm
        steAnorm = typeIII['stereo_a'][i] - t0_norm
        steBnorm = typeIII['stereo_b'][i] - t0_norm


        # typeIII_times[f"{each}"] = [windnorm.seconds,stenorm.seconds ,pspnorm.seconds ,solonorm.seconds]
        typeIII_times[f"{each}"] = []
        for sc in spacecraft:
            if sc == "wind":
                typeIII_times[f"{each}"].append(windnorm.seconds)
            elif sc == "stereo_a":
                typeIII_times[f"{each}"].append(steAnorm.seconds)
            elif sc == "stereo_b":
                typeIII_times[f"{each}"].append(steBnorm.seconds)
        i = i + 1



    """  ---------------------------------------------------------------------   """
    """  --------------------------- ON/OFF ----------------------------------   """
    """  ---------------------------------------------------------------------   """
    runtracker = True
    run_failed_again = False  # THIS GIVES ERROR. DO NOT TURN ON UNTIL FIXED.


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
                mu, sd, t1_pred, trace, summary, t_emission_fromtrace, v_analysis = triangulate(stations, typeIII_times[i_freq], t_cadence=60, cores=4, progressbar=True, report=0, plot=0,traceplot=True, savetraceplot=True, traceplotdir=f'{date_str}', traceplotfn=f'{i_freq}.jpg')
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
        savetrackedtypeiii(results, date_str=date_str, N_STATIONS=N_STATIONS,profile=profile, dir="Data/", title=f"TRACKING_{date_str}_results_{N_STATIONS}stations_{profile}_Freqs_{freqlimmin}_{freqlimmax}_HR.pkl")
    else:
        savetrackedtypeiii(results, date_str=date_str, N_STATIONS=N_STATIONS,profile=profile, dir="Data/", title=f"TRACKING_{date_str}_results_{N_STATIONS}stations_{profile}_Freqs_{freqlimmin}_{freqlimmax}_HR_{note}.pkl")

    end_time = datetime.datetime.now()

    ordercheck = []
    for tt in typeIII_times[f"{each}"]:
        if tt == windnorm.seconds:
            ordercheck.append("w")
        if tt == steAnorm.seconds:
            ordercheck.append("a")
        if tt == steBnorm.seconds:
            ordercheck.append("b")

    string = f"""
    Process finished:
    -------------------------------
    Start time: {start_time}
    End time  : {end_time}
    Total time: {end_time-start_time}


    stations_rsun = {stations_rsun}
    t_obs      =    {typeIII_times[f"{each}"]}

    order = {spacecraft}
    check = {ordercheck}


        """
    print(string)

    # stations_rsun = array([[ 216.3864883 ,   -0.82640064], [  76.26846667, -192.1493994 ],[ 143.14674755,   62.28368691],[ -38.28536325,  125.55611214]])
    # t_1MHz = [658, 621, 568, 480]
    # t_3MHz  = [627, 540, 559, 480]
