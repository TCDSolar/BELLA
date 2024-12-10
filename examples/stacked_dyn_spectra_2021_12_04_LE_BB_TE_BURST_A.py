# Author: L Alberto Canizares canizares (at) cp.dias.ie
# some_file.py
import sys
sys.path.insert(1, '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/TCDSolarBELLA')


from datetime import datetime

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import astropy.units as u
from astropy.visualization import ImageNormalize


from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter

from astropy.constants import R_sun, au


import importlib

from radiospectra.spectrogram import Spectrogram  # in the process of updating old spectrogram

from sunpy.net import attrs as a

from bella.type_III_fitter import dynspec, openmarsis
from bella.type_III_fitter import typeIIIfitter as t3f
from bella.type_III_fitter.psp_quicklook import rfs_spec
from bella.type_III_fitter.solo_quicklook_L3_data import open_rpw_l3
# from spacepy import pycdf
# from rpw_mono.thr.hfr.reader import read_hfr_autoch
# from maser.data import Data

import pickle
import argparse

importlib.reload(t3f)
importlib.reload(dynspec)

r_sun = R_sun.value
AU=au.value

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.family': "Times New Roman"})

def reverse_colourmap(cmap, name='my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k, reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r



if __name__=="__main__":
    plt.close('all')
    parser = argparse.ArgumentParser(description='Process spacecraft names.')
    parser.add_argument('--spacecraft', nargs='+', help='Spacecraft name(s)')

    args = parser.parse_args()

    #plt.close('all')

    # ---------------------------------------------------------------- #
    " Settings "
    # ---------------------------------------------------------------- #
    YYYY = 2021
    MM = 12
    dd = 4
    HH_0 = 13
    mm_0 = 0
    HH_1 = 13
    mm_1 = 20
    # HH_1 = 14
    # mm_1 = 52
    #
    background_subtraction = True
    plot_residuals = False
    savedata = True
    savefigs = True
    showfigs = True

    sat_hist = False    # Saturated histograms

    stokesV = True
    leadingedge = True
    backbone = False
    trailingedge = False


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

    # Cadence compensation or shifts
    wind_cad_comp = 0
    stea_cad_comp = 0
    steb_cad_comp = 0
    solo_cad_comp = 0
    psp_cad_comp = 0
    mex_cad_comp = 0

    local_data_dir = "/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/2021_12_04" # Path to data directory

    # Wind Spacecraft LOCAL DATA ON or OFF
    wind_local = True
    if wind_local is True:
        w_loc_file_RAD1 = f"{local_data_dir}/wind_data/waves/wi_wa_rad1_l2_60s_20211204_v01.dat"    # PATH to wind data
        w_loc_file_RAD2 = f"{local_data_dir}/wind_data/waves/wi_wa_rad2_l2_60s_20211204_v01.dat"    # PATH to wind data

    sc_str=""
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

    profile = ""
    if leadingedge:
        profile += "_LE"
    if backbone:
        profile += "_BB"
    if trailingedge:
        profile += "_TE"
    profile = profile[1:]  # remove underscore from beginning

    note_str = "BURST_A"    # This particular timeframe shows two type IIIs. Burst A is the earlier one. Burst B is the later one.
    figdir = t3f.mkdirectory(f"Figures/{YYYY}_{MM:02}_{dd:02}/{profile}")

    my_cmap = mpl.cm.jet  # Dynamic spectra colormap

    # ----------------------------------- #
    " TIMING LIMIT SETTINGS "
    # ----------------------------------- #

    mintime = datetime(YYYY, MM, dd, HH_0,mm_0)
    maxtime =  datetime(YYYY, MM, dd, HH_1,mm_1)
    timelims = [mintime,maxtime]

    # TIME LIMITS SPECIFIC TO EACH SPACECRAFT
    if windsc:
        mintime_windsc = datetime(YYYY, MM, dd, HH_0, mm_0)
        maxtime_windsc = datetime(YYYY, MM, dd, HH_1, mm_1)
        timelims_windsc = [mintime_windsc, maxtime_windsc]
    if steasc:
        mintime_steasc = datetime(YYYY, MM, dd, 13, 0)
        maxtime_steasc = datetime(YYYY, MM, dd, 15, 30)
        timelims_steasc = [mintime_steasc, maxtime_steasc]
    if stebsc:
        mintime_stebsc = datetime(YYYY, MM, dd, HH_0, mm_0)
        maxtime_stebsc = datetime(YYYY, MM, dd,  HH_1, mm_1)
        timelims_stebsc = [mintime_stebsc, maxtime_stebsc]
    if solosc:
        mintime_solosc = datetime(YYYY, MM, dd, 13, 0)
        maxtime_solosc = datetime(YYYY, MM, dd, 15, 30)
        timelims_solosc= [mintime_solosc, maxtime_solosc]
    if pspsc:
        mintime_pspsc = datetime(YYYY, MM, dd, 12, 50)
        maxtime_pspsc = datetime(YYYY, MM, dd, 13, 50)
        timelims_pspsc = [mintime_pspsc, maxtime_pspsc]
    if mexsc:
        mintime_mexsc = datetime(YYYY, MM, dd, 13, 9) # 2021-12-04 12:30:22.745000
        maxtime_mexsc = datetime(YYYY, MM, dd, 13, 14) # 2021-12-04 13:14:10.035000
        timelims_mexsc = [mintime_mexsc, maxtime_mexsc]

        mintime_mexsc_disp = datetime(YYYY, MM, dd, 13, 0)  # 2021-12-04 12:30:22.745000
        maxtime_mexsc_disp = datetime(YYYY, MM, dd, 13, 30)  # 2021-12-04 13:14:10.035000
        timelims_mexsc_disp = [mintime_mexsc_disp, maxtime_mexsc_disp]

    # ----------------------------------------- #
    " Obtain DYNAMIC SPECTRA "
    # ----------------------------------------- #
    if windsc:
        if wind_local is True:
            print("Loading Wind Spectrogram LOCALLY")
            print("----------")
            # Waves RAD 1 ( low freqs)
            waves_spec_lfr = dynspec.local_waves_spec_l2_60s(w_loc_file_RAD1, datatype='RAD1', kind='SMEAN', bg_subtraction=background_subtraction, lighttravelshift=0)
            # Waves RAD 2 (High freqs)
            waves_spec_hfr = dynspec.local_waves_spec_l2_60s(w_loc_file_RAD2, datatype='RAD2', kind='SMEAN', bg_subtraction=background_subtraction, lighttravelshift=0)
        else:
            print("Loading Wind Spectrogram with PYSPEDAS")
            print("----------")
            # Waves RAD 1 ( low freqs)
            waves_spec_lfr = dynspec.waves_spec(mintime, maxtime, datatype='RAD1', bg_subtraction=background_subtraction)
            # Waves RAD 2 (High freqs)
            waves_spec_hfr = dynspec.waves_spec(mintime, maxtime, datatype='RAD2', bg_subtraction=background_subtraction)

        # FILL BLANK GAP
        freqs_fill = [waves_spec_lfr.frequencies[-1].value, waves_spec_hfr.frequencies[0].value]* u.MHz
        meta = {
            'observatory': "WIND_fill",
            'instrument': "WAVES_fill",
            'detector': "RAD2",
            'freqs': freqs_fill,
            'times': waves_spec_hfr.times,
            'wavelength': a.Wavelength(freqs_fill[0], freqs_fill[-1]),
            'start_time': waves_spec_hfr.times[0],
            'end_time': waves_spec_hfr.times[-1]
        }
        data_fill = np.array([waves_spec_hfr.data[0],waves_spec_hfr.data[0]])
        waves_spec_fill = Spectrogram(data_fill, meta)
        print("Wind Spectrogram loaded")
        print("----------\n")
    if steasc:
        print("Loading STEREO A Spectrogram")
        print("----------")
        # SWAVES A HFR
        swaves_a_spec_hfr = dynspec.swaves_highres_spec(mintime, maxtime, probe='a', datatype='hfr', bg_subtraction=True)
        # SWAVES A LFR
        swaves_a_spec_lfr = dynspec.swaves_highres_spec(mintime, maxtime, probe='a', datatype='lfr', bg_subtraction=True)

        if stokesV:
            # STOKES V
            swaves_a_POL_hfr = dynspec.swaves_stokesV(mintime, maxtime, probe='a', datatype='hfr', bg_subtraction=False)
            # SWAVES A LFR
            swaves_a_POL_lfr = dynspec.swaves_stokesV(mintime, maxtime, probe='a', datatype='lfr', bg_subtraction=False)


        print("STEREO A Spectrogram loaded")
        print("----------\n")
    if stebsc:
        print("Loading STEREO B Spectrogram")
        print("----------")
        # SWAVES B HFR
        swaves_b_spec_hfr = dynspec.swaves_highres_spec(mintime, maxtime, probe='b', datatype='hfr', bg_subtraction=True)
        # SWAVES A LFR
        swaves_b_spec_lfr = dynspec.swaves_highres_spec(mintime, maxtime, probe='b', datatype='lfr', bg_subtraction=True)
        print("STEREO B Spectrogram loaded")
        print("----------\n")
    if solosc:
        print("Loading SolO Spectrogram")
        print("----------")
        # RPW SOLO
        cdf_file_path = f'{local_data_dir}/solo_data/RPW/l3/solo_L3_rpw-hfr-flux_20211204_V01.cdf'
        rpw_spec_hfr = open_rpw_l3(cdf_file_path, bg_subtraction=True, lighttravelshift=solo_cad_comp)

        cdf_file_path = f"{local_data_dir}/solo_data/RPW/l3/solo_L3_rpw-tnr-flux_20211204_V01.cdf"
        rpw_spec_tnr = open_rpw_l3(cdf_file_path, bg_subtraction=True, lighttravelshift=solo_cad_comp)
        # rpw_mm_tnr = ImageNormalize(rpw_spec_tnr.data, interval=PercentileInterval(97.5))
        print("SolO Spectrogram loaded")
        print("----------\n")
    if pspsc:
        print("Loading PSP Spectrogram")
        print("----------")
        # RFS FIELDS PSP
        rfs_spec_lfr = rfs_spec(mintime, maxtime, datatype='rfs_lfr', bg_subtraction=True, lighttravelshift=0)
        rfs_spec_hfr = rfs_spec(mintime, maxtime, datatype='rfs_hfr', bg_subtraction=True, lighttravelshift=0)
        print("PSP Spectrogram loaded")
        print("----------\n")
    if mexsc:
        print("Loading MEX MARSIS Spectrogram")
        print("----------")
        # MEX MARSIS
        mex_filepath = f"{local_data_dir}/mex_data/marsis/22648/FRM_AIS_RDR_22648_ASCII_.DAT"  #
        mex_spec = openmarsis.marsis_spectra(mex_filepath)
        # background = np.mean(mex_spec.data, axis=0)
        # mex_spec.data = mex_spec.data - background

        print("MEX MARSIS Spectrogram loaded")
        print("----------\n")



    # ----------------------------------------- #
    " HISTOGRAM "
    # ----------------------------------------- #
    if windsc:
        if sat_hist:
            hist_l = np.percentile(waves_spec_lfr.data[~np.isnan(waves_spec_lfr.data)], [80,90])
            hist_h = np.percentile(waves_spec_hfr.data[~np.isnan(waves_spec_hfr.data)], [80,90])
            waves_mm_l = ImageNormalize(vmin=hist_l[0], vmax=hist_l[1])
            waves_mm_h = ImageNormalize(vmin=hist_h[0], vmax=hist_h[1])
        else:
            waves_mm_l = np.percentile(waves_spec_lfr.data[~np.isnan(waves_spec_lfr.data)], [10,99.99])
            waves_mm_h = np.percentile(waves_spec_hfr.data[~np.isnan(waves_spec_hfr.data)], [10,99.99])
    if steasc:
        if sat_hist:
            hist_l = np.percentile(swaves_a_spec_lfr.data[~np.isnan(swaves_a_spec_lfr.data)], [90,99])
            hist_h = np.percentile(swaves_a_spec_hfr.data[~np.isnan(swaves_a_spec_hfr.data)], [90,99])
            swaves_a_mm_l = ImageNormalize(vmin=hist_l[0], vmax=hist_l[1])
            swaves_a_mm_h = ImageNormalize(vmin=hist_h[0], vmax=hist_h[1])
        else:
            swaves_a_mm_l = np.percentile(swaves_a_spec_lfr.data[~np.isnan(swaves_a_spec_lfr.data)], [20, 99.2])
            swaves_a_mm_h = np.percentile(swaves_a_spec_hfr.data[~np.isnan(swaves_a_spec_hfr.data)], [20, 99.94])

        if stokesV:
            swaves_a_POL_lfr_data = swaves_a_POL_lfr.data[~np.isnan(swaves_a_POL_lfr.data)]
            if len(swaves_a_POL_lfr_data)==0:
                print(" WARNING : NO STOKES V DATA FOR SWAVES A LFR")
            else:
                swaves_a_pol_hist_l = np.percentile(swaves_a_POL_lfr_data, [30, 91])

            swaves_a_POL_hfr_data = swaves_a_POL_hfr.data[~np.isnan(swaves_a_POL_hfr.data)]
            if len(swaves_a_POL_hfr_data)==0:
                print(" WARNING : NO STOKES V DATA FOR SWAVES A LFR")
            else:
                swaves_a_pol_hist_h = np.percentile(swaves_a_POL_hfr.data[~np.isnan(swaves_a_POL_hfr.data)], [50, 95])
    if stebsc:
        if sat_hist:
            hist_l = np.percentile(swaves_b_spec_lfr.data[~np.isnan(swaves_b_spec_lfr.data)], [80,90])
            hist_h = np.percentile(swaves_b_spec_hfr.data[~np.isnan(swaves_b_spec_hfr.data)], [80,90])
            swaves_b_mm_l = ImageNormalize(vmin=hist_l[0], vmax=hist_l[1])
            swaves_b_mm_h = ImageNormalize(vmin=hist_h[0], vmax=hist_h[1])

        else:
            swaves_b_mm_l = np.percentile(swaves_b_spec_lfr.data[~np.isnan(swaves_b_spec_lfr.data)], [20,99.8])
            swaves_b_mm_h = np.percentile(swaves_b_spec_hfr.data[~np.isnan(swaves_b_spec_hfr.data)], [20,99.8])
    if solosc:
        if sat_hist:
            hist_l = np.percentile(rpw_spec_tnr.data[~np.isnan(rpw_spec_tnr.data)], [80,90])
            hist_h = np.percentile(rpw_spec_hfr.data[~np.isnan(rpw_spec_hfr.data)], [80,90])
            rpw_mm_l = ImageNormalize(vmin=hist_l[0], vmax=hist_l[1])
            rpw_mm_h = ImageNormalize(vmin=hist_h[0], vmax=hist_h[1])
        else:
            rpw_mm_l = np.percentile(rpw_spec_tnr.data[~np.isnan(rpw_spec_tnr.data)], [20,99.8])
            rpw_mm_h = np.percentile(rpw_spec_hfr.data[~np.isnan(rpw_spec_hfr.data)], [20,99.8])
    if pspsc:
        if sat_hist:
            rfs_mm_l = np.percentile(rfs_spec_lfr.data[~np.isnan(rfs_spec_lfr.data)], [70,99.2])
            rfs_mm_h = np.percentile(rfs_spec_hfr.data[~np.isnan(rfs_spec_hfr.data)], [70,99.2])
        else:
            rfs_mm_l = np.percentile(rfs_spec_lfr.data[~np.isnan(rfs_spec_lfr.data)], [20,99.6])
            rfs_mm_h = np.percentile(rfs_spec_hfr.data[~np.isnan(rfs_spec_hfr.data)], [20,99.6])
    if mexsc:
        filtered_data = mex_spec.data[~np.isnan(mex_spec.data) & (mex_spec.data > 0)]
        if sat_hist:
            mex_mm_h = np.percentile(filtered_data, [1,50])
        else:
            mex_mm_h = np.percentile(filtered_data, [10,98])


    # ----------------------------------------- #
    " FREQUENCY SETTINGS "
    # ----------------------------------------- #
    ""
    #  FREQUENCY RANGES
    if leadingedge is True:
        freqlimmin_LE = 0.5 # 0.18
        freqlimmax_LE = 10
        freqlims_LE = [freqlimmin_LE, freqlimmax_LE]
    if backbone is True:
        freqlimmin_BB = 0.14  # 0.18
        freqlimmax_BB = 15
        freqlims_BB = [freqlimmin_BB, freqlimmax_BB]
    if trailingedge is True:
        freqlimmin_TE = 0.4  # 0.18
        freqlimmax_TE = 3
        freqlims_TE = [freqlimmin_TE, freqlimmax_TE]

    freqlimmin = 0.18#0.18
    freqlimmax = 3
    freqlims = [freqlimmin, freqlimmax]

    if windsc:
        freqlimmin_wind = 0.18  # 0.18
        freqlimmax_wind = 3
        freqlims_wind = [freqlimmin_wind, freqlimmax_wind]

        freqlimmin_wind_l = freqlimmin_wind
        freqlimmax_wind_l = 1
        freqlims_wind_l = [freqlimmin_wind_l, freqlimmax_wind_l]

        freqlimmin_wind_h = 1
        freqlimmax_wind_h = freqlimmax_wind
        freqlims_wind_h = [freqlimmin_wind_h, freqlimmax_wind_h]
    if steasc:
        freqlimmin_stea = 0.5  # 0.18
        freqlimmax_stea = 10
        freqlims_stea = [freqlimmin_stea, freqlimmax_stea]

        freqlimmin_stea_l = freqlimmin_stea
        freqlimmax_stea_l = 1
        freqlims_stea_l = [freqlimmin_stea_l, freqlimmax_stea_l]

        freqlimmin_stea_h = 1  # 0.18
        freqlimmax_stea_h = freqlimmax_stea
        freqlims_stea_h = [freqlimmin_stea_h, freqlimmax_stea_h]
    if stebsc:
        freqlimmin_steb = 0.5  # 0.18
        freqlimmax_steb = 10
        freqlims_steb = [freqlimmin_steb, freqlimmax_steb]

        freqlimmin_steb_l = freqlimmin_steb
        freqlimmax_steb_l = 1
        freqlims_steb_l = [freqlimmin_steb_l, freqlimmax_steb_l]

        freqlimmin_steb_h = 1  # 0.18
        freqlimmax_steb_h = freqlimmax_steb
        freqlims_steb_h = [freqlimmin_steb_h, freqlimmax_steb_h]
    if solosc:
        freqlimmin_solo = 0.5  # 0.18
        freqlimmax_solo = 5
        freqlims_solo = [freqlimmin_solo, freqlimmax_solo]

        freqlimmin_solo_l = freqlimmin_solo  # 0.18
        freqlimmax_solo_l = 1
        freqlims_solo_l = [freqlimmin_solo_l, freqlimmax_solo_l]

        freqlimmin_solo_h = 1
        freqlimmax_solo_h = freqlimmax_solo
        freqlims_solo_h = [freqlimmin_solo_h, freqlimmax_solo_h]
    if pspsc:
        freqlimmin_psp = 0.1
        freqlimmax_psp = 18
        freqlims_psp = [freqlimmin_psp, freqlimmax_psp]

        freqlimmin_psp_l = freqlimmin_psp
        freqlimmax_psp_l = 1
        freqlims_psp_l = [freqlimmin_psp_l, freqlimmax_psp_l]

        freqlimmin_psp_h = 1
        freqlimmax_psp_h = freqlimmax_psp
        freqlims_psp_h = [freqlimmin_psp_h, freqlimmax_psp_h]
    if mexsc:
        freqlimmin_mex = 0.05
        freqlimmax_mex = 10
        freqlims_mex = [freqlimmin_mex, freqlimmax_mex]

        freqlimmin_mex_l = freqlimmin_mex
        freqlimmax_mex_l = 1
        freqlims_mex_l = [freqlimmin_mex_l, freqlimmax_mex_l]

        freqlimmin_mex_h = 1
        freqlimmax_mex_h = freqlimmax_mex
        freqlims_mex_h = [freqlimmin_mex_h, freqlimmax_mex_h]

    # Input for fit
    freqfitmin = 0.04
    freqfitmax = 50

    #  Frequencies to be extracted for triangulation
    freq4trimin = 0.5
    freq4trimax = 3

    # DISPLAY ONLY FREQUENCY LIMITS
    freqlims_disp = [0.1, 15]




    # ---------------------------------------------------------------- #
    " FITTING Lightcurves "
    # ---------------------------------------------------------------- #
    ""
    # MANUAL DATA EXTRACTION
    # MANUAL points only because of Type III pair
    print("Manual lightcurve data selection. \n ---------- \n")
    freqs_manual = np.logspace(np.log10(freq4trimin), np.log10(freq4trimax), num=10)
    # -------------------------   LEADING EDGE -----------------------------------------  #
    if leadingedge is True:
        if windsc:
            # WAVES low
            # waves_risetimes_l_LE =[]
            # waves_riseval_l_LE = []
            # waves_testfreq_l_LE = []
            #
            # # WAVES high
            # waves_risetimes_h_LE = [datetime(2021, 12, 4, 13, 10, 5),
            #                       datetime(2021, 12, 4, 13, 9, 8),
            #                       datetime(2021, 12, 4, 13, 8, 9),
            #                       datetime(2021, 12, 4, 13, 7, 23),
            #                       datetime(2021, 12, 4, 13, 7, 6),
            #                       datetime(2021, 12, 4, 13, 6, 44),
            #                       datetime(2021, 12, 4, 13, 6, 36),
            #                       datetime(2021, 12, 4, 13, 6, 30),
            #                       datetime(2021, 12, 4, 13, 6, 22),
            #                       datetime(2021, 12, 4, 13, 6, 6)]
            # waves_riseval_h_LE = []
            # waves_testfreq_h_LE = np.logspace(np.log10(0.4), np.log10(3.6), num=10)
            #
            # waves_risetimes_h_LE.insert(10, datetime(2021, 12, 4, 13, 6, 12))
            # waves_testfreq_h_LE = np.insert(waves_testfreq_h_LE, 10, 10)
            #
            # # WAVES low
            waves_risetimes_l_LE =[]
            waves_riseval_l_LE = []
            waves_testfreq_l_LE = []

            # waves_risetimes_l_LE.insert(0, datetime(2021, 12, 4, 13, 17, 31))
            # waves_testfreq_l_LE = np.insert(waves_testfreq_l_LE, 0, 0.14)

            # WAVES high
            # waves_risetimes_h_LE = [datetime(2021, 12, 4, 13, 7, 30),
            #                       datetime(2021, 12, 4, 13, 6, 31),
            #                       datetime(2021, 12, 4, 13, 6, 9),
            #                       datetime(2021, 12, 4, 13, 5, 23),
            #                       datetime(2021, 12, 4, 13, 5, 6),
            #                       datetime(2021, 12, 4, 13, 4, 59),
            #                       datetime(2021, 12, 4, 13, 4, 51),
            #                       datetime(2021, 12, 4, 13, 4, 46),
            #                       datetime(2021, 12, 4, 13, 4, 36),
            #                       datetime(2021, 12, 4, 13, 4, 29)]
            waves_risetimes_h_LE = [datetime(2021, 12, 4, 13, 8, 48),
                                  datetime(2021, 12, 4, 13, 7, 41),
                                  datetime(2021, 12, 4, 13, 6, 39),
                                  datetime(2021, 12, 4, 13, 5, 56),
                                  datetime(2021, 12, 4, 13, 5, 34),
                                  datetime(2021, 12, 4, 13, 5, 10),
                                  datetime(2021, 12, 4, 13, 5, 9),
                                  datetime(2021, 12, 4, 13, 4, 57),
                                  datetime(2021, 12, 4, 13, 4, 51),
                                  datetime(2021, 12, 4, 13, 4, 40)]

            waves_riseval_h_LE = []
            waves_testfreq_h_LE = np.logspace(np.log10(0.4), np.log10(3.6), num=10)

            waves_risetimes_h_LE.insert(len(waves_risetimes_h_LE), datetime(2021, 12, 4, 13, 4, 33))
            waves_testfreq_h_LE = np.insert(waves_testfreq_h_LE, len(waves_testfreq_h_LE), 10)

            # waves_risetimes_h_LE.insert(10, datetime(2021, 12, 4, 13, 4, 20))
            # waves_testfreq_h_LE = np.insert(waves_testfreq_h_LE, 10, 10)
        if steasc:
            # SWAVES A Low freqs
            swaves_a_risetimes_l_LE = []
            swaves_a_riseval_l_LE = []
            swaves_a_testfreq_l_LE = []
            # SWAVES A High freqs
            swaves_a_risetimes_h_LE = [ datetime(2021, 12, 4, 13, 14, 37),
                                         datetime(2021, 12, 4, 13, 10, 27),
                                         datetime(2021, 12, 4, 13, 8, 3),
                                         datetime(2021, 12, 4, 13, 6, 54),
                                         datetime(2021, 12, 4, 13, 5, 47),
                                         datetime(2021, 12, 4, 13, 5, 23),
                                         datetime(2021, 12, 4, 13, 5, 46),
                                         datetime(2021, 12, 4, 13, 5, 15),
                                         datetime(2021, 12, 4, 13, 5, 15),
                                         datetime(2021, 12, 4, 13, 4, 54)]

            swaves_a_riseval_h_LE = []
            swaves_a_testfreq_h_LE = np.logspace(np.log10(0.23), np.log10(10), num=10)
        if stebsc:
            # SWAVES A Low freqs
            swaves_b_risetimes_l_LE = []
            swaves_b_riseval_l_LE = []
            swaves_b_testfreq_l_LE = []
            # SWAVES B High freqs
            swaves_b_risetimes_h_LE = []
            swaves_b_riseval_h_LE = []
            swaves_b_testfreq_h_LE = []
        if solosc:
            # RPW TNR
            rpw_risetimes_l_LE = []
            rpw_riseval_l_LE = []
            rpw_testfreq_l_LE = []

            # RPW HFR
            rpw_risetimes_h_LE = [datetime(2021, 12, 4, 13, 9, 20),
                                  datetime(2021, 12, 4, 13, 8, 9),
                                  datetime(2021, 12, 4, 13, 7, 12),
                                  datetime(2021, 12, 4, 13, 6, 30),
                                  datetime(2021, 12, 4, 13, 6, 8),
                                  datetime(2021, 12, 4, 13, 5, 44),
                                  datetime(2021, 12, 4, 13, 5, 36),
                                  datetime(2021, 12, 4, 13, 5, 30),
                                  datetime(2021, 12, 4, 13, 5, 22),
                                  datetime(2021, 12, 4, 13, 5, 16)]
            rpw_riseval_h_LE = []
            rpw_testfreq_h_LE = np.logspace(np.log10(0.4), np.log10(3.6), num=10)



            rpw_risetimes_h_LE.insert(10, datetime(2021, 12, 4, 13, 5, 6))
            rpw_testfreq_h_LE = np.insert(rpw_testfreq_h_LE, 10, 10)

        if pspsc:
            # RFS LFR
            rfs_risetimes_l_LE = []
            rfs_riseval_l_LE = []
            rfs_testfreq_l_LE = []
            # RFS HFR
            rfs_risetimes_h_LE = [datetime(2021, 12, 4, 13, 7, 6),
                                 datetime(2021, 12, 4, 13, 4, 51),
                                 datetime(2021, 12, 4, 13, 3, 5),
                                 datetime(2021, 12, 4, 13, 2, 19),
                                 datetime(2021, 12, 4, 13, 2, 7),
                                 datetime(2021, 12, 4, 13, 1, 42),
                                 datetime(2021, 12, 4, 13, 1, 14),
                                 datetime(2021, 12, 4, 13, 1, 8),
                                 datetime(2021, 12, 4, 13, 0, 27),
                                 datetime(2021, 12, 4, 13, 0, 16)]
            rfs_riseval_h_LE = []
            rfs_testfreq_h_LE = np.logspace(np.log10(0.53), np.log10(10), num=10)

            # Correct fit at low frequency
            # rfs_risetimes_h_LE.insert(0, datetime(2021, 12, 4, 13, 32, 58))
            # rfs_testfreq_h_LE = np.insert(rfs_testfreq_h_LE, 0, 0.11)
        if mexsc:
            # MEX MARSIS
            mex_risetimes_l_LE = []
            mex_riseval_l_LE = []
            mex_testfreq_l_LE = []

            mex_risetimes_h_LE = [datetime(2021, 12, 4, 13, 13, 47),
                                     datetime(2021, 12, 4, 13, 12, 25),
                                     datetime(2021, 12, 4, 13, 11, 36),
                                     datetime(2021, 12, 4, 13, 10, 58),
                                     datetime(2021, 12, 4, 13, 10, 38),
                                     datetime(2021, 12, 4, 13, 10, 25),
                                     datetime(2021, 12, 4, 13, 10, 10),
                                     datetime(2021, 12, 4, 13, 10, 3),
                                     datetime(2021, 12, 4, 13, 10, 0),
                                     datetime(2021, 12, 4, 13, 9, 57)]
            mex_riseval_h_LE = []
            mex_testfreq_h_LE =  np.logspace(np.log10(0.5), np.log10(4.5), num=10)
    # -------------------------   Backbone -----------------------------------------  #
    if backbone is True:
        if windsc:
            # WAVES low
            waves_risetimes_l_BB =[]
            waves_riseval_l_BB = []
            waves_testfreq_l_BB = []

            # WAVES high
            waves_risetimes_h_BB = []
            waves_riseval_h_BB= []
            waves_testfreq_h_BB = []
        if steasc:
            # SWAVES A LOW freqs
            swaves_a_risetimes_l_BB = []
            swaves_a_riseval_l_BB = []
            swaves_a_testfreq_l_BB = []
            # SWAVES A High freqs
            swaves_a_risetimes_h_BB = []
            swaves_a_riseval_h_BB = []
            swaves_a_testfreq_h_BB = []
        if stebsc:
            # SWAVES B LOW freqs
            swaves_b_risetimes_l_BB = []
            swaves_b_riseval_l_BB = []
            swaves_b_testfreq_l_BB = []
            # SWAVES B High freqs
            swaves_b_risetimes_h_BB = []
            swaves_b_riseval_h_BB = []
            swaves_b_testfreq_h_BB = []
        if solosc:
            # RPW TNR
            rpw_risetimes_l_BB = []
            rpw_riseval_l_BB = []
            rpw_testfreq_l_BB = []

            # RPW HFR
            rpw_risetimes_h_BB = []
            rpw_riseval_h_BB = []
            rpw_testfreq_h_BB = []
        if pspsc:
            # RFS LFR
            rfs_risetimes_l_BB = []
            rfs_riseval_l_BB = []
            rfs_testfreq_l_BB = []
            # RFS HFR
            rfs_risetimes_h_BB = []
            rfs_riseval_h_BB = []
            rfs_testfreq_h_BB = []
        if mexsc:
            # MEX MARSIS
            mex_risetimes_l_BB = []
            mex_riseval_l_BB = []
            mex_testfreq_l_BB = []

            mex_risetimes_h_BB = []
            mex_riseval_h_BB = []
            mex_testfreq_h_BB = []
    # -------------------------   Trailing EDGE -----------------------------------------  #
    if trailingedge is True:
        if windsc:
            # WAVES low
            waves_risetimes_l_TE =[]
            waves_riseval_l_TE = []
            waves_testfreq_l_TE = []

            # WAVES high
            waves_risetimes_h_TE = []
            waves_riseval_h_TE= []
            waves_testfreq_h_TE = []
        if steasc:
            # SWAVES A LOW freqs
            swaves_a_risetimes_l_TE = []
            swaves_a_riseval_l_TE = []
            swaves_a_testfreq_l_TE = []
            # SWAVES A High freqs
            swaves_a_risetimes_h_TE = []
            swaves_a_riseval_h_TE = []
            swaves_a_testfreq_h_TE = []
        if stebsc:
            # SWAVES B LOW freqs
            swaves_b_risetimes_l_TE = []
            swaves_b_riseval_l_TE = []
            swaves_b_testfreq_l_TE = []
            # SWAVES B High freqs
            swaves_b_risetimes_h_TE = []
            swaves_b_riseval_h_TE = []
            swaves_b_testfreq_h_TE = []
        if solosc:
            # RPW TNR
            rpw_risetimes_l_TE = []
            rpw_riseval_l_TE = []
            rpw_testfreq_l_TE = []

            # RPW HFR
            rpw_risetimes_h_TE = []
            rpw_riseval_h_TE = []
            rpw_testfreq_h_TE = []
        if pspsc:
            # RFS LFR
            rfs_risetimes_l_TE = []
            rfs_riseval_l_TE = []
            rfs_testfreq_l_TE = []
            # RFS HFR
            rfs_risetimes_h_TE = []
            rfs_riseval_h_TE = []
            rfs_testfreq_h_TE = []
        if mexsc:
            # MEX MARSIS
            mex_risetimes_l_TE = []
            mex_riseval_l_TE = []
            mex_testfreq_l_TE = []

            mex_risetimes_h_TE = []
            mex_riseval_h_TE = []
            mex_testfreq_h_TE = []


    # ---------------------------------------------------------------- #
    " Type III FITTING "
    # ---------------------------------------------------------------- #
    fitfreqs = np.logspace(np.log10(freqfitmin), np.log10(freqfitmax), num=50)
    freqs4tri = np.logspace(np.log10(freq4trimin), np.log10(freq4trimax), num=50)
    if leadingedge is True:
        print("Leading edge fitting. \n ---------- \n")


        if windsc:
            #  Wind
            wavesrisetimes_LE =  list(waves_risetimes_l_LE) + list(waves_risetimes_h_LE)
            wavesriseval_LE   =  list(waves_riseval_l_LE)   + list(waves_riseval_h_LE)
            wavestestfreq_LE  =  list(waves_testfreq_l_LE)  + list(waves_testfreq_h_LE)

            times4tri_waves_LE_dt, fitfreqs_waves_LE, fittimes_corrected_waves_LE = t3f.typeIIIfitting(wavesrisetimes_LE,
                                                                                                   wavestestfreq_LE,
                                                                                                   fitfreqs, freqs4tri,
                                                                                                   plot_residuals=False)
        if steasc:
            swaves_a_risetimes_LE = list(swaves_a_risetimes_h_LE) + list(swaves_a_risetimes_l_LE)
            swaves_a_riseval_LE = list(swaves_a_riseval_h_LE)  + list(swaves_a_riseval_l_LE)
            swaves_a_testfreq_LE = list(swaves_a_testfreq_h_LE) + list(swaves_a_testfreq_l_LE)

            times4tri_swaves_a_LE_dt, fitfreqs_swaves_a_LE, fittimes_corrected_swaves_a_LE =  t3f.typeIIIfitting(swaves_a_risetimes_LE,
                                                                                                   swaves_a_testfreq_LE,
                                                                                                   fitfreqs, freqs4tri,
                                                                                                   plot_residuals=False)
        if stebsc:
            swaves_b_risetimes_LE = list(swaves_b_risetimes_h_LE) + list(swaves_b_risetimes_l_LE)
            swaves_b_riseval_LE = list(swaves_b_riseval_h_LE)  + list(swaves_b_riseval_l_LE)
            swaves_b_testfreq_LE = list(swaves_b_testfreq_h_LE) + list(swaves_b_testfreq_l_LE)

            times4tri_swaves_b_LE_dt, fitfreqs_swaves_b_LE, fittimes_corrected_swaves_b_LE =  t3f.typeIIIfitting(swaves_b_risetimes_LE,
                                                                                                   swaves_b_testfreq_LE,
                                                                                                   fitfreqs, freqs4tri,
                                                                                                   plot_residuals=False)
        if solosc:
            rpw_risetimes_LE = list(rpw_risetimes_h_LE) + list(rpw_risetimes_l_LE)
            rpw_riseval_LE = list(rpw_riseval_h_LE) + list(rpw_riseval_l_LE)
            rpw_testfreq_LE = list(rpw_testfreq_h_LE) +list(rpw_testfreq_l_LE)

            times4tri_rpw_LE_dt, fitfreqs_rpw_LE, fittimes_corrected_rpw_LE =  t3f.typeIIIfitting(rpw_risetimes_LE,
                                                                                             rpw_testfreq_LE,
                                                                                             fitfreqs, freqs4tri,
                                                                                             plot_residuals=False)
        if pspsc:
            rfsrisetimes_LE = list(rfs_risetimes_l_LE) + list(rfs_risetimes_h_LE)
            rfsriseval_LE = list(rfs_riseval_l_LE) + list(rfs_riseval_h_LE)
            rfstestfreq_LE = list(rfs_testfreq_l_LE) + list(rfs_testfreq_h_LE)

            times4tri_rfs_LE_dt, fitfreqs_rfs_LE, fittimes_corrected_rfs_LE = t3f.typeIIIfitting(rfsrisetimes_LE,
                                                                                             rfstestfreq_LE,
                                                                                             fitfreqs, freqs4tri,
                                                                                             plot_residuals=False)
        if mexsc:
            # MEX MARSIS
            mex_risetimes_LE = list(mex_risetimes_l_LE) + list(mex_risetimes_h_LE)
            mex_riseval_LE = list(mex_riseval_l_LE) + list(mex_riseval_h_LE)
            mex_testfreq_LE = list(mex_testfreq_l_LE) + list(mex_testfreq_h_LE)

            times4tri_mex_LE_dt, fitfreqs_mex_LE, fittimes_corrected_mex_LE = t3f.typeIIIfitting(mex_risetimes_LE,
                                                                                                   mex_testfreq_LE,
                                                                                                   fitfreqs, freqs4tri,
                                                                                                   plot_residuals=False)

        print("Leading edge fitting. DONE. \n ---------- \n")

    # BACKBONE
    if backbone is True:
        print("Leading edge fitting. \n ---------- \n")

        if windsc:
            #  Wind
            wavesrisetimes_BB = list(waves_risetimes_l_BB) + list(waves_risetimes_h_BB)
            wavesriseval_BB = list(waves_riseval_l_BB) + list(waves_riseval_h_BB)
            wavestestfreq_BB = list(waves_testfreq_l_BB) + list(waves_testfreq_h_BB)

            times4tri_waves_BB_dt, fitfreqs_waves_BB, fittimes_corrected_waves_BB = t3f.typeIIIfitting(
                wavesrisetimes_BB,
                wavestestfreq_BB,
                fitfreqs, freqs4tri,
                plot_residuals=False)
        if steasc:
            swaves_a_risetimes_BB = list(swaves_a_risetimes_h_BB) + list(swaves_a_risetimes_l_BB)
            swaves_a_riseval_BB = list(swaves_a_riseval_h_BB) + list(swaves_a_riseval_l_BB)
            swaves_a_testfreq_BB = list(swaves_a_testfreq_h_BB) + list(swaves_a_testfreq_l_BB)

            times4tri_swaves_a_BB_dt, fitfreqs_swaves_a_BB, fittimes_corrected_swaves_a_BB = t3f.typeIIIfitting(
                swaves_a_risetimes_BB,
                swaves_a_testfreq_BB,
                fitfreqs, freqs4tri,
                plot_residuals=False)
        if stebsc:
            swaves_b_risetimes_BB = list(swaves_b_risetimes_h_BB) + list(swaves_b_risetimes_l_BB)
            swaves_b_riseval_BB = list(swaves_b_riseval_h_BB) + list(swaves_b_riseval_l_BB)
            swaves_b_testfreq_BB = list(swaves_b_testfreq_h_BB) + list(swaves_b_testfreq_l_BB)

            times4tri_swaves_b_BB_dt, fitfreqs_swaves_b_BB, fittimes_corrected_swaves_b_BB = t3f.typeIIIfitting(
                swaves_b_risetimes_BB,
                swaves_b_testfreq_BB,
                fitfreqs, freqs4tri,
                plot_residuals=False)
        if solosc:
            rpw_risetimes_BB = list(rpw_risetimes_h_BB) + list(rpw_risetimes_l_BB)
            rpw_riseval_BB = list(rpw_riseval_h_BB) + list(rpw_riseval_l_BB)
            rpw_testfreq_BB = list(rpw_testfreq_h_BB) + list(rpw_testfreq_l_BB)

            times4tri_rpw_BB_dt, fitfreqs_rpw_BB, fittimes_corrected_rpw_BB = t3f.typeIIIfitting(rpw_risetimes_BB,
                                                                                                 rpw_testfreq_BB,
                                                                                                 fitfreqs, freqs4tri,
                                                                                                 plot_residuals=False)
        if pspsc:
            rfsrisetimes_BB = list(rfs_risetimes_l_BB) + list(rfs_risetimes_h_BB)
            rfsriseval_BB = list(rfs_riseval_l_BB) + list(rfs_riseval_h_BB)
            rfstestfreq_BB = list(rfs_testfreq_l_BB) + list(rfs_testfreq_h_BB)

            times4tri_rfs_BB_dt, fitfreqs_rfs_BB, fittimes_corrected_rfs_BB = t3f.typeIIIfitting(rfsrisetimes_BB,
                                                                                                 rfstestfreq_BB,
                                                                                                 fitfreqs, freqs4tri,
                                                                                                 plot_residuals=False)
        if mexsc:
            # MEX MARSIS
            mex_risetimes_BB = list(mex_risetimes_l_BB) + list(mex_risetimes_h_BB)
            mex_riseval_BB = list(mex_riseval_l_BB) + list(mex_riseval_h_BB)
            mex_testfreq_BB = list(mex_testfreq_l_BB) + list(mex_testfreq_h_BB)

            times4tri_mex_BB_dt, fitfreqs_mex_BB, fittimes_corrected_mex_BB = t3f.typeIIIfitting(mex_risetimes_BB,
                                                                                                 mex_testfreq_BB,
                                                                                                 fitfreqs, freqs4tri,
                                                                                                 plot_residuals=False)

        print("Leading edge fitting. DONE. \n ---------- \n")

    # TRAILING EDGE
    if trailingedge is True:
        print("Leading edge fitting. \n ---------- \n")

        if windsc:
            #  Wind
            wavesrisetimes_TE = list(waves_risetimes_l_TE) + list(waves_risetimes_h_TE)
            wavesriseval_TE = list(waves_riseval_l_TE) + list(waves_riseval_h_TE)
            wavestestfreq_TE = list(waves_testfreq_l_TE) + list(waves_testfreq_h_TE)

            times4tri_waves_TE_dt, fitfreqs_waves_TE, fittimes_corrected_waves_TE = t3f.typeIIIfitting(
                wavesrisetimes_TE,
                wavestestfreq_TE,
                fitfreqs, freqs4tri,
                plot_residuals=False)
        if steasc:
            swaves_a_risetimes_TE = list(swaves_a_risetimes_h_TE) + list(swaves_a_risetimes_l_TE)
            swaves_a_riseval_TE = list(swaves_a_riseval_h_TE) + list(swaves_a_riseval_l_TE)
            swaves_a_testfreq_TE = list(swaves_a_testfreq_h_TE) + list(swaves_a_testfreq_l_TE)

            times4tri_swaves_a_TE_dt, fitfreqs_swaves_a_TE, fittimes_corrected_swaves_a_TE = t3f.typeIIIfitting(
                swaves_a_risetimes_TE,
                swaves_a_testfreq_TE,
                fitfreqs, freqs4tri,
                plot_residuals=False)
        if stebsc:
            swaves_b_risetimes_TE = list(swaves_b_risetimes_h_TE) + list(swaves_b_risetimes_l_TE)
            swaves_b_riseval_TE = list(swaves_b_riseval_h_TE) + list(swaves_b_riseval_l_TE)
            swaves_b_testfreq_TE = list(swaves_b_testfreq_h_TE) + list(swaves_b_testfreq_l_TE)

            times4tri_swaves_b_TE_dt, fitfreqs_swaves_b_TE, fittimes_corrected_swaves_b_TE = t3f.typeIIIfitting(
                swaves_b_risetimes_TE,
                swaves_b_testfreq_TE,
                fitfreqs, freqs4tri,
                plot_residuals=False)
        if solosc:
            rpw_risetimes_TE = list(rpw_risetimes_h_TE) + list(rpw_risetimes_l_TE)
            rpw_riseval_TE = list(rpw_riseval_h_TE) + list(rpw_riseval_l_TE)
            rpw_testfreq_TE = list(rpw_testfreq_h_TE) + list(rpw_testfreq_l_TE)

            times4tri_rpw_TE_dt, fitfreqs_rpw_TE, fittimes_corrected_rpw_TE = t3f.typeIIIfitting(rpw_risetimes_TE,
                                                                                                 rpw_testfreq_TE,
                                                                                                 fitfreqs, freqs4tri,
                                                                                                 plot_residuals=False)
        if pspsc:
            rfsrisetimes_TE = list(rfs_risetimes_l_TE) + list(rfs_risetimes_h_TE)
            rfsriseval_TE = list(rfs_riseval_l_TE) + list(rfs_riseval_h_TE)
            rfstestfreq_TE = list(rfs_testfreq_l_TE) + list(rfs_testfreq_h_TE)

            times4tri_rfs_TE_dt, fitfreqs_rfs_TE, fittimes_corrected_rfs_TE = t3f.typeIIIfitting(rfsrisetimes_TE,
                                                                                                 rfstestfreq_TE,
                                                                                                 fitfreqs, freqs4tri,
                                                                                                 plot_residuals=False)
        if mexsc:
            # MEX MARSIS
            mex_risetimes_TE = list(mex_risetimes_l_TE) + list(mex_risetimes_h_TE)
            mex_riseval_TE = list(mex_riseval_l_TE) + list(mex_riseval_h_TE)
            mex_testfreq_TE = list(mex_testfreq_l_TE) + list(mex_testfreq_h_TE)

            times4tri_mex_TE_dt, fitfreqs_mex_TE, fittimes_corrected_mex_TE = t3f.typeIIIfitting(mex_risetimes_TE,
                                                                                                 mex_testfreq_TE,
                                                                                                 fitfreqs, freqs4tri,
                                                                                                 plot_residuals=False)

        print("Leading edge fitting. DONE. \n ---------- \n")

    # ---------------------------------------------------------------- #
    " Plotting TYPE IIIs "
    # ---------------------------------------------------------------- #
    if windsc:
        #  ---------------------------------------------------------
        #     WIND
        # ----------------------------------------------------------
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 9))
        if isinstance(waves_mm_l, ImageNormalize):
            waves_spec_lfr.plot(axes=axes, norm=waves_mm_l, cmap=my_cmap)
        else:
            waves_spec_lfr.plot(axes=axes, norm=LogNorm(vmin=waves_mm_l[0], vmax=waves_mm_l[1]), cmap=my_cmap)
        if isinstance(waves_mm_h, ImageNormalize):
            waves_spec_hfr.plot(axes=axes, norm=waves_mm_h, cmap=my_cmap)
        else:
            waves_spec_hfr.plot(axes=axes, norm=LogNorm(vmin=waves_mm_h[0], vmax=waves_mm_h[1]), cmap=my_cmap)

        # LEADING EDGE
        if leadingedge is True:
            # axes.plot(waves_risetimes_l_LE, waves_testfreq_l_LE, 'k*')
            # axes.plot(waves_risetimes_h_LE, waves_testfreq_h_LE, 'k*')
            axes.plot(wavesrisetimes_LE, wavestestfreq_LE, 'k*')
            axes.plot(fittimes_corrected_waves_LE, fitfreqs_waves_LE, "k--")
            axes.plot(fittimes_corrected_waves_LE, fitfreqs_waves_LE, "y--")

            # axes.plot(rpw_risetimes_LE, rpw_testfreq_LE, 'k*')
            # axes.plot(fittimes_corrected_rpw_LE, fitfreqs_rpw_LE, "k--")
            # axes.plot(fittimes_corrected_rpw_LE, fitfreqs_rpw_LE, "y--")
        # BACKBONE
        if backbone is True:
            # axes.plot(waves_risetimes_l_BB, waves_testfreq_l_BB, 'k*')
            # axes.plot(waves_risetimes_h_BB, waves_testfreq_h_BB, 'k*')
            axes.plot(wavesrisetimes_BB, wavestestfreq_BB, 'k*')
            axes.plot(fittimes_corrected_waves_BB, fitfreqs_waves_BB, "k--")
            axes.plot(fittimes_corrected_waves_BB, fitfreqs_waves_BB, "y--")

        # TRAILING EDGE
        if trailingedge is True:
            # axes.plot(waves_risetimes_l_TE, waves_testfreq_l_TE, 'k*')
            # axes.plot(waves_risetimes_h_TE, waves_testfreq_h_TE, 'k*')
            axes.plot(wavesrisetimes_TE, wavestestfreq_TE, 'k*')
            axes.plot(fittimes_corrected_waves_TE, fitfreqs_waves_TE, "k--")
            axes.plot(fittimes_corrected_waves_TE, fitfreqs_waves_TE, "y--")

        axes.set_ylim(reversed(axes.get_ylim()))
        axes.set_yscale('log')
        axes.set_xlim(mintime, maxtime)
        # axes.set_ylim([freqlimmax, freqlimmin])
        plt.subplots_adjust(hspace=0.31)
        if showfigs:
            plt.show(block=False)
        else:
            plt.close(fig)
    if steasc:
        # ---------------------------------------------------------------------
        #            STEREO A
        # ---------------------------------------------------------------------
        #
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 9))

        if isinstance(swaves_a_mm_l, ImageNormalize):
            swaves_a_spec_lfr.plot(axes=axes, norm=swaves_a_mm_l, cmap=my_cmap)
        else:
            swaves_a_spec_lfr.plot(axes=axes, norm=LogNorm(vmin=swaves_a_mm_l[0], vmax=swaves_a_mm_l[1]), cmap=my_cmap)

        if isinstance(swaves_a_mm_h, ImageNormalize):
            swaves_a_spec_hfr.plot(axes=axes, norm=swaves_a_mm_h, cmap=my_cmap)
        else:
            swaves_a_spec_hfr.plot(axes=axes, norm=LogNorm(vmin=swaves_a_mm_h[0], vmax=swaves_a_mm_h[1]), cmap=my_cmap)

        # LEADING EDGE
        if leadingedge is True:
            axes.plot(swaves_a_risetimes_h_LE, swaves_a_testfreq_h_LE,  'ro', markeredgecolor="w")
            axes.plot(fittimes_corrected_swaves_a_LE, fitfreqs_swaves_a_LE, "k--")

        # BACKBONE
        if backbone is True:
            axes.plot(swaves_a_risetimes_h_BB, swaves_a_testfreq_h_BB, 'r*')
            axes.plot(fittimes_corrected_swaves_a_BB, fitfreqs_swaves_a_BB, "k--")

        # TRAILING EDGE
        if trailingedge is True:
            axes.plot(swaves_a_risetimes_h_TE, swaves_a_testfreq_h_TE, 'r*')
            axes.plot(fittimes_corrected_swaves_a_TE, fitfreqs_swaves_a_TE, "k--")

        axes.set_yscale('log')
        axes.set_xlim(mintime_steasc, maxtime_steasc)
        axes.set_ylim(reversed(axes.get_ylim()))
        # axes.set_ylim([freqlimmax, freqlimmin])
        plt.subplots_adjust(hspace=0.31)
        if showfigs:
            plt.show(block=False)
        else:
            plt.close(fig)
        if stokesV:
            # STOKES V
            fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 9))
            swaves_a_POL_hfr.plot(axes=axes, vmin=swaves_a_pol_hist_h[0], vmax=swaves_a_pol_hist_h[1], cmap=my_cmap)
            # swaves_a_POL_lfr.plot(axes=axes, norm=LogNorm(vmin=swaves_a_pol_hist_l[0], vmax=swaves_a_pol_hist_l[1]), cmap=my_cmap)

            # # LEADING EDGE
            # if leadingedge  is  True:
            #     axes.plot(swaves_a_risetimes_h_LE, swaves_a_testfreq_h_LE, 'ro', markeredgecolor="w")
            #     axes.plot(fittimes_corrected_swaves_a_LE, fitfreqs_swaves_a_LE, "k--")
            #
            # # BACKBONE
            # if backbone  is  True:
            #     axes.plot(swaves_a_risetimes_h_BB, swaves_a_testfreq_h_BB, 'r*')
            #     axes.plot(fittimes_corrected_swaves_a_BB, fitfreqs_swaves_a_BB, "k--")
            #
            # # TRAILING EDGE
            # if trailingedge  is  True:
            #     axes.plot(swaves_a_risetimes_h_TE, swaves_a_testfreq_h_TE, 'r*')
            #     axes.plot(fittimes_corrected_swaves_a_TE, fitfreqs_swaves_a_TE, "k--")

            axes.set_yscale('log')
            axes.set_xlim(mintime_steasc, maxtime_steasc)
            axes.set_ylim(reversed(axes.get_ylim()))
            # axes.set_ylim([freqlimmax, freqlimmin])
            plt.subplots_adjust(hspace=0.31)
            if showfigs:
                plt.show(block=False)
            else:
                plt.close(fig)
        # freqs_pol = [2, 6]
        # tlim_pol = [datetime(2021,12, 4, 13,12), datetime(2021,12, 4, 13,14)]
        # fpol_idx = np.where(swaves_a_POL_hfr.frequencies.value>freqs_pol[0]) and np.where(swaves_a_POL_hfr.frequencies.value<freqs_pol[1])
        # tpol_idx = np.where(swaves_a_POL_hfr.times.value>tlim_pol[0]) and np.where(swaves_a_POL_hfr.times.value<tlim_pol[1])
        #
        # plt.figure()
    if stebsc:
        # ---------------------------------------------------------------------
        #            STEREO B
        # ---------------------------------------------------------------------
        #
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 9))

        if isinstance(swaves_b_mm_l, ImageNormalize):
            swaves_b_spec_lfr.plot(axes=axes, norm=swaves_b_mm_l, cmap=my_cmap)
        else:
            swaves_b_spec_lfr.plot(axes=axes, norm=LogNorm(vmin=swaves_b_mm_l[0], vmax=swaves_b_mm_l[1]), cmap=my_cmap)

        if isinstance(swaves_b_mm_h, ImageNormalize):
            swaves_b_spec_hfr.plot(axes=axes, norm=swaves_b_mm_h, cmap=my_cmap)
        else:
            swaves_b_spec_hfr.plot(axes=axes, norm=LogNorm(vmin=swaves_b_mm_h[0], vmax=swaves_b_mm_h[1]), cmap=my_cmap)
        # LEADING EDGE
        if leadingedge is True:
            axes.plot(swaves_b_risetimes_h_LE, swaves_b_testfreq_h_LE,  'ro', markeredgecolor="w")
            axes.plot(fittimes_corrected_swaves_b_LE, fitfreqs_swaves_b_LE, "k--")

        # BACKBONE
        if backbone is True:
            axes.plot(swaves_b_risetimes_h_BB, swaves_b_testfreq_h_BB, 'r*')
            axes.plot(fittimes_corrected_swaves_b_BB, fitfreqs_swaves_b_BB, "k--")

        # TRAILING EDGE
        if trailingedge is True:
            axes.plot(swaves_b_risetimes_h_TE, swaves_b_testfreq_h_TE, 'r*')
            axes.plot(fittimes_corrected_swaves_b_TE, fitfreqs_swaves_b_TE, "k--")

        axes.set_yscale('log')
        axes.set_xlim(mintime_steasc, maxtime_steasc)
        axes.set_ylim(reversed(axes.get_ylim()))
        # axes.set_ylim([freqlimmax, freqlimmin])
        plt.subplots_adjust(hspace=0.31)
        if showfigs:
            plt.show(block=False)
        else:
            plt.close(fig)
    if solosc:
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 9))
        if isinstance(rpw_mm_l, ImageNormalize):
            rpw_spec_tnr.plot(axes=axes, norm=rpw_mm_l, cmap=my_cmap)
        else:
            rpw_spec_tnr.plot(axes=axes, norm=LogNorm(vmin=rpw_mm_l[0], vmax=rpw_mm_l[1]), cmap=my_cmap)

        if isinstance(rpw_mm_h, ImageNormalize):
            rpw_spec_hfr.plot(axes=axes, norm=rpw_mm_h, cmap=my_cmap)
        else:
            rpw_spec_hfr.plot(axes=axes, norm=LogNorm(vmin=rpw_mm_h[0], vmax=rpw_mm_h[1]), cmap=my_cmap)

        # LEADING EDGE
        if leadingedge is True:
            axes.plot(rpw_risetimes_h_LE, rpw_testfreq_h_LE,  'ro', markeredgecolor="w")
            axes.plot(fittimes_corrected_rpw_LE, fitfreqs_rpw_LE, "k--")

        # BACKBONE
        if backbone is True:
            axes.plot(rpw_risetimes_h_BB, rpw_testfreq_h_BB, 'r*')
            axes.plot(fittimes_corrected_rpw_BB, fitfreqs_rpw_BB, "k--")

        # TRAILING EDGE
        if trailingedge is True:
            axes.plot(rpw_risetimes_l_TE, rpw_testfreq_l_TE, 'r*')
            axes.plot(rpw_risetimes_h_TE, rpw_testfreq_h_TE, 'r*')
            axes.plot(fittimes_corrected_rpw_TE, fitfreqs_rpw_TE, "k--")

        axes.set_yscale('log')
        axes.set_xlim(mintime_solosc, maxtime_solosc)
        axes.set_ylim(reversed(axes.get_ylim()))
        # axes.set_ylim([freqlimmax, freqlimmin])
        plt.subplots_adjust(hspace=0.31)
        if showfigs:
            plt.show(block=False)
        else:
            plt.close(fig)
    if pspsc:
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 9))
        if isinstance(rfs_mm_l, ImageNormalize):
            rfs_spec_lfr.plot(axes=axes, norm=rfs_mm_l, cmap=my_cmap)
        else:
            rfs_spec_lfr.plot(axes=axes, norm=LogNorm(vmin=rfs_mm_l[0], vmax=rfs_mm_l[1]), cmap=my_cmap)

        if isinstance(rfs_mm_h, ImageNormalize):
            rfs_spec_hfr.plot(axes=axes, norm=rfs_mm_h, cmap=my_cmap)
        else:
            rfs_spec_hfr.plot(axes=axes, norm=LogNorm(vmin=rfs_mm_h[0], vmax=rfs_mm_h[1]), cmap=my_cmap)

        # LEADING EDGE
        if leadingedge is True:
            # axes.plot(rfs_risetimes_l_LE, rfs_testfreq_l_LE, 'k*')
            # axes.plot(rfs_risetimes_h_LE, rfs_testfreq_h_LE, 'k*')
            axes.plot(rfsrisetimes_LE, rfstestfreq_LE,  'ro', markeredgecolor="w")
            axes.plot(fittimes_corrected_rfs_LE, fitfreqs_rfs_LE, "k--")

        # BACKBONE
        if backbone is True:
            # axes.plot(rfs_risetimes_l_BB, rfs_testfreq_l_BB, 'k*')
            # axes.plot(rfs_risetimes_h_BB, rfs_testfreq_h_BB, 'k*')
            axes.plot(rfsrisetimes_BB, rfstestfreq_BB, 'k*')
            axes.plot(fittimes_corrected_rfs_BB, fitfreqs_rfs_BB, "k--")

        # TRAILING EDGE
        if trailingedge is True:
            # axes.plot(rfs_risetimes_l_TE, rfs_testfreq_l_TE, 'k*')
            # axes.plot(rfs_risetimes_h_TE, rfs_testfreq_h_TE, 'k*')
            axes.plot(rfsrisetimes_TE, rfstestfreq_TE, 'k*')
            axes.plot(fittimes_corrected_rfs_TE, fitfreqs_rfs_TE, "k--")

        axes.set_ylim(reversed(axes.get_ylim()))
        axes.set_yscale('log')
        axes.set_xlim(mintime_pspsc, maxtime_pspsc)
        axes.set_ylim([freqlimmax_psp, freqlimmin_psp])
        plt.subplots_adjust(hspace=0.31)
        if showfigs:
            plt.show(block=False)
        else:
            plt.close(fig)
    if mexsc:
        #  MEX MARSIS
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 9))
        if isinstance(mex_mm_h, ImageNormalize):
            mex_spec.plot(axes=axes, norm=mex_mm_h, cmap=my_cmap)
        else:
            mex_spec.plot(axes=axes, norm=LogNorm(vmin=mex_mm_h[0], vmax=mex_mm_h[1]), cmap=my_cmap)

        # LEADING EDGE
        if leadingedge is True:
            axes.plot(mex_risetimes_LE, mex_testfreq_LE, 'ro', markeredgecolor="w")
            axes.plot(fittimes_corrected_mex_LE, fitfreqs_mex_LE, "k--")

        # BACKBONE
        if backbone is True:
            axes.plot(mex_risetimes_BB, mex_testfreq_BB, 'ro', markeredgecolor="w")
            axes.plot(fittimes_corrected_mex_BB, fitfreqs_mex_BB, "k--")

        # TRAILING EDGE
        if trailingedge is True:
            axes.plot(mex_risetimes_TE, mex_testfreq_TE, 'ro', markeredgecolor="w")
            axes.plot(fittimes_corrected_mex_TE, fitfreqs_mex_TE, "k--")

        axes.set_yscale('log')
        axes.set_ylim(reversed(axes.get_ylim()))
        axes.set_xlim(mintime_mexsc, maxtime_mexsc)
        axes.set_ylim([freqlimmax_mex, freqlimmin_mex])
        plt.subplots_adjust(hspace=0.31)

        # axes.set_ylim([freq4trimax,freq4trimin])
        if showfigs:
            plt.show(block=False)
        else:
            plt.close(fig)
    # ---------------------------------------------------------------- #
    " JOINT PLOT "
    # ---------------------------------------------------------------- #
    ""
    # ---------------------------------------------------------------- #
    " JOINT PLOT Horizontal "
    # ---------------------------------------------------------------- #
    n_axes = 0
    # Define a dictionary to map keys to axes
    ax_mapping = {}
    # Define a list to store the titles of the spectrograms
    titles = []
    # Define a list to store the data and normalization parameters for each spectrogram
    spectrograms = []
    # Define a list to store the conditions and associated data for each spectrogram
    conditions = []

    # in order of distance to the sources
    if pspsc:
        conditions.append(('psp', "PSP/FIELDS", [rfs_spec_lfr, rfs_spec_hfr], [rfs_mm_l, rfs_mm_h]))
    if steasc:
        conditions.append(('stea', "STEREO A/WAVES", [swaves_a_spec_lfr, swaves_a_spec_hfr], [swaves_a_mm_l, swaves_a_mm_h]))
    if stebsc:
        conditions.append(('steb', "STEREO B, WAVES, HFR+LFR", [swaves_b_spec_lfr, swaves_b_spec_hfr], [swaves_b_mm_l, swaves_b_mm_h]))
    if windsc:
        conditions.append(('wind', "Wind/WAVES", [waves_spec_lfr, waves_spec_hfr], [waves_mm_l, waves_mm_h]))
    if solosc:
        conditions.append(('solo', "SolO/RPW", [rpw_spec_tnr, rpw_spec_hfr], [rpw_mm_l, rpw_mm_h]))
    if mexsc:
        conditions.append(('mex', "MEX/MARSIS", [mex_spec], [mex_mm_h]))

    models = {}
    if leadingedge:
        models['LE'] = {}
        if windsc:
            models['LE']['wind'] = (fittimes_corrected_waves_LE, fitfreqs_waves_LE, times4tri_waves_LE_dt, freqs4tri)
        if steasc:
            models['LE']['stea'] = (
            fittimes_corrected_swaves_a_LE, fitfreqs_swaves_a_LE, times4tri_swaves_a_LE_dt, freqs4tri)
        if stebsc:
            models['LE']['steb'] = (
            fittimes_corrected_swaves_b_LE, fitfreqs_swaves_b_LE, times4tri_swaves_b_LE_dt, freqs4tri)
        if solosc:
            models['LE']['solo'] = (fittimes_corrected_rpw_LE, fitfreqs_rpw_LE, times4tri_rpw_LE_dt, freqs4tri)
        if pspsc:
            models['LE']['psp'] = (fittimes_corrected_rfs_LE, fitfreqs_rfs_LE, times4tri_rfs_LE_dt, freqs4tri)
        if mexsc:
            models['LE']['mex'] = (fittimes_corrected_mex_LE, fitfreqs_mex_LE, times4tri_mex_LE_dt, freqs4tri)
    if backbone:
        models['BB'] = {}
        if windsc:
            models['BB']['wind'] = (fittimes_corrected_waves_BB, fitfreqs_waves_BB, times4tri_waves_BB_dt, freqs4tri)
        if steasc:
            models['BB']['stea'] = (
            fittimes_corrected_swaves_a_BB, fitfreqs_swaves_a_BB, times4tri_swaves_a_BB_dt, freqs4tri)
        if stebsc:
            models['BB']['steb'] = (
            fittimes_corrected_swaves_b_BB, fitfreqs_swaves_b_BB, times4tri_swaves_b_BB_dt, freqs4tri)
        if solosc:
            models['BB']['solo'] = (fittimes_corrected_rpw_BB, fitfreqs_rpw_BB, times4tri_rpw_BB_dt, freqs4tri)
        if pspsc:
            models['BB']['psp'] = (fittimes_corrected_rfs_BB, fitfreqs_rfs_BB, times4tri_rfs_BB_dt, freqs4tri)
        if mexsc:
            models['BB']['mex'] = (fittimes_corrected_mex_BB, fitfreqs_mex_BB, times4tri_mex_BB_dt, freqs4tri)
    if trailingedge:
        models['TE'] = {}
        if windsc:
            models['TE']['wind'] = (fittimes_corrected_waves_TE, fitfreqs_waves_TE, times4tri_waves_TE_dt, freqs4tri)
        if steasc:
            models['TE']['stea'] = (
            fittimes_corrected_swaves_a_TE, fitfreqs_swaves_a_TE, times4tri_swaves_a_TE_dt, freqs4tri)
        if stebsc:
            models['TE']['steb'] = (
            fittimes_corrected_swaves_b_TE, fitfreqs_swaves_b_TE, times4tri_swaves_b_TE_dt, freqs4tri)
        if solosc:
            models['TE']['solo'] = (fittimes_corrected_rpw_TE, fitfreqs_rpw_TE, times4tri_rpw_TE_dt, freqs4tri)
        if pspsc:
            models['TE']['psp'] = (fittimes_corrected_rfs_TE, fitfreqs_rfs_TE, times4tri_rfs_TE_dt, freqs4tri)
        if mexsc:
            models['TE']['mex'] = (fittimes_corrected_mex_TE, fitfreqs_mex_TE, times4tri_mex_TE_dt, freqs4tri)


    # Create subplots based on the dynamically determined number of axes
    fig, axes = plt.subplots(1, len(conditions), sharex=True, sharey=True, figsize=(17, 8))

    # Set face color for each subplot
    cmap_lowest = my_cmap(0)
    for ax in axes:
        ax.set_facecolor(cmap_lowest)

    # Plot each spectrogram on the allocated axes
    for i, (key, title_spec, data, mm_values) in enumerate(conditions):
        ax = axes[i] if len(conditions) > 1 else axes  # Adjust for when there's only one condition
        ax_mapping[key] = i  # Assigning an index to the axes for easy access
        n_axes += 1
        titles.append(title_spec)

        # Plot spectrogram data
        for j, spec_data in enumerate(data):
            mm_value = mm_values[j] if len(mm_values) > j else mm_values[0]  # Adjusted condition for mm_values
            if isinstance(mm_value, ImageNormalize):
                spec_data.plot(axes=ax, norm=mm_value, cmap=my_cmap)
            else:
                spec_data.plot(axes=ax, norm=LogNorm(vmin=mm_value[0], vmax=mm_value[1]), cmap=my_cmap)

        ax.set_title(title_spec)

        # Plot model lines for the current condition if applicable
        for model_type in models:
            if key in models[model_type]:
                fittimes, fitfreqs, times4tri, freqs4tri = models[model_type][key]
                ax.plot(fittimes, fitfreqs, "r-", label=model_type + " fit")  # Model fit line
                ax.plot(times4tri, freqs4tri, "w-", lw=2, label=model_type + " data")  # Model data points/triangles

        # # It might be useful to add a legend if you're plotting model lines
        # if key in models.get('LE', {}) or key in models.get('BB', {}) or key in models.get('TE', {}):
        #     ax.legend(loc='upper right')

    for each in axes:
        each.set_ylim(reversed(each.get_ylim()))
        each.set_yscale('log')


    axes[0].set_ylim([freqlims_disp[1],freqlims_disp[0]])
    # axes[1].set_ylim([freqlimmax, 0.2])
    # axes[2].set_ylim([freqlimmax, 0.2])
    date_format = DateFormatter('%H:%M')
    axes[0].xaxis.set_major_formatter(date_format)

    axes[0].set_xlim(datetime(YYYY, MM, dd, HH_0,mm_0), datetime(YYYY, MM, dd, HH_1,mm_1))
    plt.subplots_adjust(left=0.05, bottom=0.096, right=0.984, top=0.93, wspace=0.1, hspace=0.31)
    plt.tick_params(axis='y', which='minor')
    # axes[0].yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    plt.tick_params(axis='y', which='major')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[0].set_ylabel("Frequency (MHz)")
    # plt.tight_layout()
    if savefigs:
        if note_str!="":
            note = f"_{note_str}"
        figfname = f"{figdir}/{YYYY}_{MM:02}_{dd:02}{sc_str}_Horizontal{note}.png"
        plt.savefig(figfname, dpi='figure')
        print(f"\nFigure saved: {figfname}")

    if showfigs:
        plt.show(block=False)
    else:
        plt.close(fig)
    # ---------------------------------------------------------------- #
    "END      JOINT PLOT"
    # ---------------------------------------------------------------- #


    ""
    # ---------------------------------------------------------------- #
    "Save the data"
    # ---------------------------------------------------------------- #
    if savedata is True:
        typeIIIdir = t3f.mkdirectory(f"Data/TypeIII/{YYYY}_{MM:02}_{dd:02}/")
        if leadingedge is True:
            typeIII_LE = {'Freqs': freqs4tri}  # Start with 'Freqs' key-value pair
            # Conditional assignments based on the presence of spacecraft data
            if windsc:
                typeIII_LE['WindTime'] = times4tri_waves_LE_dt
            if steasc:
                typeIII_LE['StereoATime'] = times4tri_swaves_a_LE_dt
            if stebsc:
                typeIII_LE['StereoBTime'] = times4tri_swaves_b_LE_dt
            if solosc:
                typeIII_LE['SoloTime'] = times4tri_rpw_LE_dt
            if pspsc:
                typeIII_LE['PSPTime'] = times4tri_rfs_LE_dt
            if mexsc:
                typeIII_LE['MEXTime'] = times4tri_mex_LE_dt
            if note_str=="":
                savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}{sc_str}_Freqs_{freq4trimin}_{freq4trimax}_LE.pkl'
            else:
                savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}{sc_str}_Freqs_{freq4trimin}_{freq4trimax}_LE_{note_str}.pkl'
            with open(savedfilepath, 'wb') as outp:
                pickle.dump(typeIII_LE, outp, pickle.HIGHEST_PROTOCOL)
            print(f"\nSaved results: {savedfilepath}")
        if backbone is True:
            typeIII_BB = {'Freqs': freqs4tri}  # Start with 'Freqs' key-value pair
            # Conditional assignments based on the presence of spacecraft data
            if windsc:
                typeIII_BB['WindTime'] = times4tri_waves_BB_dt
            if steasc:
                typeIII_BB['StereoATime'] = times4tri_swaves_a_BB_dt
            if stebsc:
                typeIII_BB['StereoBTime'] = times4tri_swaves_b_BB_dt
            if solosc:
                typeIII_BB['SoloTime'] = times4tri_rpw_BB_dt
            if pspsc:
                typeIII_BB['PSPTime'] = times4tri_rfs_BB_dt
            if mexsc:
                typeIII_BB['MEXTime'] = times4tri_mex_BB_dt

            if note_str=="":
                savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}{sc_str}_Freqs_{freq4trimin}_{freq4trimax}_BB.pkl'
            else:
                savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}{sc_str}_Freqs_{freq4trimin}_{freq4trimax}_BB_{note_str}.pkl'
            with open(savedfilepath, 'wb') as outp:
                pickle.dump(typeIII_BB, outp, pickle.HIGHEST_PROTOCOL)
            print(f"\nSaved results: {savedfilepath}")
        if trailingedge is True:
            typeIII_TE = {'Freqs': freqs4tri}  # Start with 'Freqs' key-value pair
            # Conditional assignments based on the presence of spacecraft data
            if windsc:
                typeIII_TE['WindTime'] = times4tri_waves_TE_dt
            if steasc:
                typeIII_TE['StereoATime'] = times4tri_swaves_a_TE_dt
            if stebsc:
                typeIII_TE['StereoBTime'] = times4tri_swaves_b_TE_dt
            if solosc:
                typeIII_TE['SoloTime'] = times4tri_rpw_TE_dt
            if pspsc:
                typeIII_TE['PSPTime'] = times4tri_rfs_TE_dt
            if mexsc:
                typeIII_TE['MEXTime'] = times4tri_mex_TE_dt
            if note_str=="":
                savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}{sc_str}_Freqs_{freq4trimin}_{freq4trimax}_TE.pkl'
            else:
                savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}{sc_str}_Freqs_{freq4trimin}_{freq4trimax}_TE_{note_str}.pkl'
            with open(savedfilepath, 'wb') as outp:
                pickle.dump(typeIII_TE, outp, pickle.HIGHEST_PROTOCOL)
            print(f"\nSaved results: {savedfilepath}")

    else:
        print("Warning. savedata is False, data was not saved.")



    """ --------------------------------------------------------------- """
    # VERTICAL PLOT
    """ --------------------------------------------------------------- """

    # Create subplots based on the dynamically determined number of axes
    fig, axes = plt.subplots(len(conditions),1, sharex=True, sharey=True, figsize=(13, 13))

    # Set face color for each subplot
    cmap_lowest = my_cmap(0)
    for ax in axes:
        ax.set_facecolor(cmap_lowest)

    # Plot each spectrogram on the allocated axes
    for i, (key, title_spec, data, mm_values) in enumerate(conditions):
        ax = axes[i] if len(conditions) > 1 else axes  # Adjust for when there's only one condition
        ax_mapping[key] = i  # Assigning an index to the axes for easy access
        n_axes += 1
        titles.append(title_spec)

        # Plot spectrogram data
        for j, spec_data in enumerate(data):
            mm_value = mm_values[j] if len(mm_values) > j else mm_values[0]  # Adjusted condition for mm_values
            if isinstance(mm_value, ImageNormalize):
                spec_data.plot(axes=ax, norm=mm_value, cmap=my_cmap)
            else:
                spec_data.plot(axes=ax, norm=LogNorm(vmin=mm_value[0], vmax=mm_value[1]), cmap=my_cmap)

        ax.set_title(title_spec)

        # Plot model lines for the current condition if applicable
        for model_type in models:
            if key in models[model_type]:
                fittimes, fitfreqs, times4tri, freqs4tri = models[model_type][key]
                ax.plot(fittimes, fitfreqs, "r-", label=model_type + " fit")  # Model fit line
                ax.plot(times4tri, freqs4tri, "w-", lw=2, label=model_type + " data")  # Model data points/triangles

        # # It might be useful to add a legend if you're plotting model lines
        # if key in models.get('LE', {}) or key in models.get('BB', {}) or key in models.get('TE', {}):
        #     ax.legend(loc='upper right')

    for each in axes:
        each.set_ylim(reversed(each.get_ylim()))
        each.set_yscale('log')


    axes[0].set_ylim([freqlims_disp[1],freqlims_disp[0]])
    # axes[1].set_ylim([freqlimmax, 0.2])
    # axes[2].set_ylim([freqlimmax, 0.2])


    axes[0].set_xlim(datetime(YYYY, MM, dd, HH_0,mm_0), datetime(YYYY, MM, dd, HH_1,mm_1))
    plt.subplots_adjust(left=0.041, bottom=0.096, right=0.984, top=0.93, wspace=0.132, hspace=0.31)
    plt.tick_params(axis='y', which='minor')
    axes[0].yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    plt.tick_params(axis='y', which='major')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[0].set_ylabel("Frequency (MHz)")
    plt.tight_layout()
    if savefigs:
        if note_str!="":
            note = f"_{note_str}"
        figfname = f"{figdir}/{YYYY}_{MM:02}_{dd:02}{sc_str}_Vertical{note}.png"
        plt.savefig(figfname, dpi='figure')
        print(f"\nFigure saved: {figfname}")

    if showfigs:
        plt.show(block=False)
    else:
        plt.close(fig)


    report_cadences = f"""
    Report CADENCES
    Wind:
        lfr:{dynspec.check_spectro_cadence(waves_spec_lfr).total_seconds()}s
        hfr:{dynspec.check_spectro_cadence(waves_spec_lfr).total_seconds()}s
    STEREO:

    """

    for each in conditions:
        print(f"{each[1]}:")
        for spec in each[2]:
            print(f"{dynspec.check_spectro_cadence(spec).total_seconds()}s")
