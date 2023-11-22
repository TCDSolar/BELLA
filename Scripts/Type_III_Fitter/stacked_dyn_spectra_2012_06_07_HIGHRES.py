# Author: L Alberto Canizares canizares (at) cp.dias.ie
from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.interpolate import CubicSpline
from numpy import interp
import pyspedas
from pytplot import get_data

from astropy.time import Time
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size': 22})
from matplotlib.ticker import FormatStrFormatter,LogFormatter

from datetime import datetime
from datetime import timedelta

from astropy.constants import c, m_e, R_sun, e, eps0, au
r_sun = R_sun.value
AU=au.value

from sunpy.net import Fido, attrs as a

from radiospectra import net #let Fido know about the radio clients
from radiospectra.spectrogram import Spectrogram # in the process of updating old spectrogram

import cdflib
# from spacepy import pycdf

# from rpw_mono.thr.hfr.reader import read_hfr_autoch

from typeIIIfitter import *
from maser.data import Data
import solarmap
import pickle
import os

def f_to_angs(f_mhz,c=299792458):
    angstrom = (c / (f_mhz * 10 ** 6)) * 10 ** 10
    return angstrom

def backSub(data, percentile=1):
    """ Background subtraction:
        This function has been modified from Eoin Carley's backsub funcion
        https://github.com/eoincarley/ilofar_scripts/blob/master/Python/bst/plot_spectro.py

        data:        numpy 2D matrix of floating values for dynamic spectra
        percentile:  integer value def = 1. bottom X percentile of time slices



        METHOD ----
        * This function takes the bottom x percentile of time slices
        * Averages those time slices.
        * Subtracts the average value from the whole dataset


        """
    # Get time slices with standard devs in the bottom nth percentile.
    # Get average spectra from these time slices.
    # Devide through by this average spec.
    # Expects (row, column)

    print("Start of Background Subtraction of data")
    dat = data.T
    # dat = np.log10(dat)
    # dat[np.where(np.isinf(dat) == True)] = 0.0
    dat_std = np.std(dat, axis=0)
    dat_std = dat_std[np.nonzero(dat_std)]
    min_std_indices = np.where(dat_std < np.percentile(dat_std, percentile))[0]
    min_std_spec = dat[:, min_std_indices]
    min_std_spec = np.mean(min_std_spec, axis=1)
    dat = np.transpose(np.divide(np.transpose(dat), min_std_spec))

    data = dat.T

    # Alternative: Normalizing frequency channel responses using median of values.
    # for sb in np.arange(data.shape[0]):
    #       data[sb, :] = data[sb, :]/np.mean(data[sb, :])

    print("Background Subtraction of data done")
    return data

def solo_rpw_hfr(filepath):
    rpw_l2_hfr = cdflib.CDF(filepath)
    # l2_cdf_file = pycdf.CDF(filepath)

    # times = l2_cdf_file['Epoch']
    # times = times[:]

    times = rpw_l2_hfr.varget('EPOCH')


    freqs = rpw_l2_hfr.varget('FREQUENCY')

    # Indicates the THR sensor configuration (V1=1, V2=2, V3=3, V1-V2=4, V2-V3=5,
    # V3-V1=6, B_MF=7, HF_V1-V2=9, HF_V2-V3=10, HF_V3-V1=11)
    sensor = rpw_l2_hfr.varget('SENSOR_CONFIG')
    freq_uniq = np.unique(rpw_l2_hfr.varget('FREQUENCY'))  # frequency channels list
    sample_time = rpw_l2_hfr.varget('SAMPLE_TIME')

    agc1 = rpw_l2_hfr.varget('AGC1')
    agc2 = rpw_l2_hfr.varget('AGC2')

    flux_density1 = rpw_l2_hfr.varget('FLUX_DENSITY1')
    flux_density2 = rpw_l2_hfr.varget('FLUX_DENSITY2')

    rpw_l2_hfr.close()
    # l2_cdf_file.close()

    # For CH1 extract times, freqs and data points
    slices1 = []
    times1 = []
    freq1 = []
    for cfreq in freq_uniq:
        search = np.argwhere((freqs == cfreq) & (sensor[:, 0] == 9) & (agc1 != 0))
        if search.size > 0:
            slices1.append(agc1[search])
            times1.append(times[search])
            freq1.append(cfreq)

    # For CH1 extract times, freqs and data points
    slices2 = []
    times2 = []
    freq2 = []
    for cfreq in freq_uniq:
        search = np.argwhere((freqs == cfreq) & (sensor[:, 1] == 9) & (agc2 != 0))
        if search.size > 0:
            slices2.append(agc2[search])
            times2.append(times[search])
            freq2.append(cfreq)

    # Kinda arb but pick a time near middle of freq sweep
    tt1 = np.hstack(times1)[:, -1]#160]
    tt2 = np.hstack(times2)[:, -1]#50]

    spec1 = np.hstack(slices1)
    spec2 = np.hstack(slices2)

    return tt1, freq1, spec1, tt2, freq2, spec2

def check_cadence(times, plot=True, method="mode", title=""):
    dtime = []
    times = np.array(times)
    time_idx = []
    for i in range(1, len(times)):
        ti = times[i].datetime
        ti_1 = times[i-1].datetime
        diff = ti - ti_1
        dtime.append(diff.total_seconds())
        time_idx.append(times[i].datetime)


    if plot==True:
        from matplotlib import pyplot as plt, dates as mdates

        plt.figure()
        ax = plt.gca()
        plt.plot_date(time_idx, dtime, ".")
        plt.xlabel(f"Time {time_idx[0].year}/{time_idx[0].month:02}/{time_idx[0].day:02} ")
        plt.ylabel("cadence")
        plt.title(title)
        formatter = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(formatter)

        plt.show(block=False)

    if method=="average":
        return np.mean(dtime)
    elif method=="max":
        return np.amax(dtime)
    elif method=="min":
        return np.amin(dtime)
    elif method=="mode":
        return stats.mode(dtime)[0][0]
    else:
        print("method can only be mode, average, max or min")

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def swaves_highres_spec(start, endt, probe='a', datatype='hfr', bg_subtraction=False):
    """
    Downloads and processes high-resolution SWAVES spectrogram data from the STEREO mission.

    Parameters
    ----------
    start : datetime
        The start time of the data range to be downloaded.
    endt : datetime
        The end time of the data range to be downloaded.
    probe : str, optional
        The STEREO spacecraft probe to use. Default is 'a'. Alternative is 'b'.
    datatype : str, optional
        The type of SWAVES data to use. Default is 'hfr'. Alternative is 'lfr'
    bg_subtraction : bool, optional
        Whether or not to perform background subtraction on the spectrogram. Default is False.

    Returns
    -------
    swaves_spec : `~sunpy.timeseries.Spectrogram`
        A `sunpy.timeseries.Spectrogram` object containing the SWAVES spectrogram data. please see radiospectra https://github.com/sunpy/radiospectra

    Notes
    -----
    This function uses the `pyspedas` and sunpy's 'radiospectra' libraries to download and process the SWAVES data. The resulting
    spectrogram object includes metadata such as the observatory, instrument, detector, frequencies, and time range.
    """
    stereo_variables = pyspedas.stereo.waves([start.strftime("%m/%d/%Y %H:%M:%S"), endt.strftime("%m/%d/%Y %H:%M:%S")], probe = probe, datatype = datatype)
    print(f"Stereo {probe.upper()} {datatype.upper()} downloaded: stereo_variables:")
    print(stereo_variables)
    swaves_data = get_data('PSD_FLUX')


    swaves_freqs_MHz = swaves_data.v/ 1e6 * u.MHz

    swaves_timestamps = swaves_data.times
    swaves_times = Time(Time(np.array([datetime.utcfromtimestamp(t_) for t_ in swaves_timestamps])).isot)


    swaves_psdarray = swaves_data.y

    meta = {
        'observatory': f"STEREO {probe.upper()}",
        'instrument': "SWAVES",
        'detector': datatype.upper(),
        'freqs': swaves_freqs_MHz,
        'times': swaves_times,
        'wavelength': a.Wavelength(swaves_freqs_MHz[0], swaves_freqs_MHz[-1]),
        'start_time': swaves_times[0],
        'end_time': swaves_times[-1]
    }
    swaves_spec = Spectrogram(swaves_psdarray.T, meta)

    if bg_subtraction:
        swaves_spec.data = backSub(swaves_spec.data.T).T

    return swaves_spec

def waves_spec(start, endt,datatype="RAD1", bg_subtraction=False):
    """
    Downloads and processes high-resolution WAVES spectrogram data from the WIND mission.

    Parameters
    ----------
    start : datetime
        The start time of the data range to be downloaded.
    endt : datetime
        The end time of the data range to be downloaded.

    datatype : str, optional
        The type of WAVES data to use. Default is 'RAD1'. Alternative is 'RAD2'

    bg_subtraction : bool, optional
        Whether or not to perform background subtraction on the spectrogram. Default is False.

    Returns
    -------
    wwaves_spec : `~sunpy.timeseries.Spectrogram`
        A `sunpy.timeseries.Spectrogram` object containing the WAVES spectrogram data. please see radiospectra https://github.com/sunpy/radiospectra

    Notes
    -----
    This function uses the `pyspedas` and sunpy's 'radiospectra' libraries to download and process the WAVES data. The resulting
    spectrogram object includes metadata such as the observatory, instrument, detector, frequencies, and time range.
    """
    wind_variables = pyspedas.wind.waves([start.strftime("%m/%d/%Y %H:%M:%S"), endt.strftime("%m/%d/%Y %H:%M:%S")])
    print(f"Wind WAVES downloaded: wind_variables:")
    print(wind_variables)
    waves_data = get_data(f'E_VOLTAGE_{datatype}')



    waves_freqs_MHz = waves_data.v/ 1e3 * u.MHz

    waves_timestamps = waves_data.times
    waves_times = Time(Time(np.array([datetime.utcfromtimestamp(t_) for t_ in waves_timestamps])).isot)


    waves_psdarray = waves_data.y

    meta = {
        'observatory': f"WIND",
        'instrument': "WAVES",
        'detector': datatype,
        'freqs': waves_freqs_MHz,
        'times': waves_times,
        'wavelength': a.Wavelength(waves_freqs_MHz[0], waves_freqs_MHz[-1]),
        'start_time': waves_times[0],
        'end_time': waves_times[-1]
    }
    waves_spec = Spectrogram(waves_psdarray.T, meta)

    if bg_subtraction:
        waves_spec.data = backSub(waves_spec.data.T).T

    return waves_spec

def reciprocal_3rdorder(x, a0, a1, a2, a3):
    return a0 + a1 / x + a2 / x ** 2+ a3 / x ** 3

def reciprocal_2ndorder(x, a0, a1, a2):
    return a0 + a1 / x + a2 / x ** 2

def loadpickle(filenamef):
    import pickle
    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)

    inp.close()
    return results

if __name__=="__main__":
    # Fido
    YYYY = 2012
    MM = 6
    dd = 7
    HH_0 = 19
    mm_0 = 20
    HH_1 = 20
    mm_1 = 00
    #
    background_subtraction = True
    leadingedge = True
    backbone = False
    plot_residuals = False
    figdir = mkdirectory(f"Figures/{YYYY}_{MM:02}_{dd:02}")

    mintime = datetime(YYYY, MM, dd, HH_0,mm_0)
    maxtime =  datetime(YYYY, MM, dd, HH_1,mm_1)
    timelims = [mintime,maxtime]





    # Waves RAD 1 ( low freqs)
    waves_spec_lfr = waves_spec(mintime, maxtime, datatype='RAD1', bg_subtraction=True)
    # Waves RAD 2 (High freqs)
    waves_spec_hfr = waves_spec(mintime, maxtime, datatype='RAD2', bg_subtraction=True)


    # SWAVES A HFR
    swaves_a_spec_hfr = swaves_highres_spec(mintime, maxtime, probe='a', datatype='hfr', bg_subtraction=True)

    # SWAVES B HFR
    swaves_b_spec_hfr = swaves_highres_spec(mintime, maxtime, probe='b', datatype='hfr', bg_subtraction=True)




    # Histogram levels
    waves_mm_l = np.percentile(waves_spec_lfr.data, [20,100])
    waves_mm_h = np.percentile(waves_spec_hfr.data, [20,100])

    # swaves_a_mm_l = np.percentile(swaves_a_spec_lfr.data, [70,99.99])
    swaves_a_mm_h = np.percentile(swaves_a_spec_hfr.data, [1,99.999])

    # swaves_b_mm_l = np.percentile(swaves_b_spec_lfr.data, [70,99.99])
    swaves_b_mm_h = np.percentile(swaves_b_spec_hfr.data, [1,99.999])


    histograms_check = np.concatenate((waves_mm_l,waves_mm_h ,swaves_a_mm_h,swaves_b_mm_h))
    if any(check <= 0 for check in histograms_check):
        print('WARNING: One or more histogram levels is 0. Increase this value to avoid error in plots. ')


    #  Dynamic spectra frequency range.
    # LEADING EDGE
    freqlimmin_LE = 0.02 #0.18
    freqlimmax_LE = 3
    freqlims_LE = [freqlimmin_LE, freqlimmax_LE]

    freqlimmin_BB = 0.1#0.18
    freqlimmax_BB = 3
    freqlims_BB = [freqlimmin_BB, freqlimmax_BB]


    # Input for fit
    freqfitmin = 0.02# 0.15
    freqfitmax = 13

    #  Frequencies to be extracted for Multilateration
    freq4trimin = 0.15
    freq4trimax = 2



    if leadingedge ==True:
        # -------------------------   LEADING EDGE -----------------------------------------  #
        # WAVES low
        waves_risetimes_l_LE, waves_riseval_l_LE, waves_testfreq_l_LE = auto_rise_times(waves_spec_lfr, freqlims=[0.1, 3], timelims=timelims, sigma=1, sdauto=5, h3guess=0, h4guess=0,method="sigma", saveplots=True)
        # WAVES high
        waves_risetimes_h_LE, waves_riseval_h_LE, waves_testfreq_h_LE = auto_rise_times(waves_spec_hfr, freqlims=[0.15, 3], timelims=timelims, sigma=1, sdauto=2, h3guess=0.1, h4guess=0,method="sigma", saveplots=True)

        swaves_a_risetimes_h_LE, swaves_a_riseval_h_LE, swaves_a_testfreq_h_LE = auto_rise_times(swaves_a_spec_hfr, freqlims=freqlims_LE, timelims=timelims, sigma=1.5, sdauto=2, h3guess=0.1, h4guess=0,method="sigma", saveplots=True)
        swaves_b_risetimes_h_LE, swaves_b_riseval_h_LE, swaves_b_testfreq_h_LE = auto_rise_times(swaves_b_spec_hfr, freqlims=freqlims_LE, timelims=timelims, sigma=1.5, sdauto=2, h3guess=0.1, h4guess=0,method="sigma", saveplots=True)


        # Fine tuning.
        # # MANUAL EDITS BASED ON VISUAL INSPECTION OF THE DYNSPECTRA.
        # Fix Outliers and to prevent unphysical morphologies.

        waves_risetimes_l_LE[0] = datetime(2012, 6, 7, 19, 38)
        waves_risetimes_l_LE[1] = datetime(2012, 6, 7, 19, 37)
        waves_risetimes_l_LE[2] = datetime(2012, 6, 7, 19, 36, 50)
        waves_risetimes_l_LE[3] = datetime(2012, 6, 7, 19, 36, 40)
        waves_risetimes_l_LE[4] = datetime(2012, 6, 7, 19, 36, 30)
        waves_risetimes_l_LE[5] = datetime(2012, 6, 7, 19, 36, 20)
        waves_risetimes_l_LE[6] = datetime(2012, 6, 7, 19, 35, 50)
        waves_risetimes_l_LE[7] = datetime(2012, 6, 7, 19, 35, 30)
        waves_risetimes_l_LE[8] = datetime(2012, 6, 7, 19, 35, 50)
        waves_risetimes_l_LE[9] = datetime(2012, 6, 7, 19, 35, 50)
        waves_risetimes_l_LE[10] = datetime(2012, 6, 7, 19, 35,0)
        waves_risetimes_l_LE[11] = datetime(2012, 6, 7, 19, 35,0)
        waves_risetimes_l_LE[12] = datetime(2012, 6, 7, 19, 35,0)

        swaves_a_risetimes_h_LE[0] = datetime(2012, 6, 7, 19, 50, 0)
        swaves_a_risetimes_h_LE[1] = datetime(2012, 6, 7, 19, 39, 42)

        swaves_b_risetimes_h_LE[0] = datetime(2012, 6, 7, 19, 46, 50)
        swaves_b_risetimes_h_LE[1] = datetime(2012, 6, 7, 19, 40, 38)



    if backbone ==True:
        # -------------------------   BACKBONE -----------------------------------------  #
        # WAVES low
        waves_risetimes_l_BB, waves_riseval_l_BB, waves_testfreq_l_BB = auto_rise_times(waves_spec_lfr, freqlims=freqlims_BB, timelims=timelims, sdauto=5, h3guess=0, h4guess=0,method="peak", saveplots=True)
        # WAVES high
        waves_risetimes_h_BB, waves_riseval_h_BB, waves_testfreq_h_BB = auto_rise_times(waves_spec_hfr, freqlims=freqlims_BB, timelims=timelims, sdauto=2, h3guess=0.1, h4guess=0,method="peak", saveplots=True)

        swaves_a_risetimes_h_BB, swaves_a_riseval_h_BB, swaves_a_testfreq_h_BB = auto_rise_times(swaves_a_spec_hfr, freqlims=freqlims_BB, timelims=timelims, sdauto=2, h3guess=0.1, h4guess=0,method="peak", saveplots=True)
        swaves_b_risetimes_h_BB, swaves_b_riseval_h_BB, swaves_b_testfreq_h_BB = auto_rise_times(swaves_b_spec_hfr, freqlims=freqlims_BB, timelims=timelims, sdauto=2, h3guess=0.1, h4guess=0,method="peak", saveplots=True)


    # -------------------------------------------------------------------------------------------------------------------- #
    #                                                           FITTING TYPE IIIs
    # -------------------------------------------------------------------------------------------------------------------- #




    fitfreqs = np.logspace(np.log10(freqfitmin), np.log10(freqfitmax), num=200)
    freqs4tri = np.logspace(np.log10(freq4trimin), np.log10(freq4trimax), num=200)
    # fitfreqs2 = np.linspace(freqlimmin,freqlimmax, num=300 )


    # running this shows why fitfreqs needs to be in logspace.
    #  if fitfreqs is in logspace then theres an even distribution of points between low freqs and highfreqs
    # xx = np.logspace(np.log10(freqfitmin), np.log10(freqfitmax), num=300)
    # xx2 = np.linspace(freqfitmin,freqfitmax, num=300 )
    #
    # plt.figure()
    # plt.plot(xx, "r.")
    # plt.plot(xx2, "b.")
    # plt.yscale('log')
    # plt.show(block=False)



    # ---------------------------------------------------------------- #
    # LEADING EDGE
    # ---------------------------------------------------------------- #
    if leadingedge == True:
        # waves_risetimes_l_LE_manual = datetime(2012, 6, 7, 19, 40, 0)
        # waves_riseval_l_LE_manual = np.average(waves_riseval_l_LE)  # just add arbitrary value to keep all matrices the same size
        # waves_testfreq_l_LE_manual =

        #  Wind
        wavesrisetimes_LE =  list(waves_risetimes_l_LE) + list(waves_risetimes_h_LE)
        wavesriseval_LE   =  list(waves_riseval_l_LE)   + list(waves_riseval_h_LE)
        wavestestfreq_LE  =  list(waves_testfreq_l_LE)  + list(waves_testfreq_h_LE)

        xdata_waves_LE, xref_waves_LE = epoch2time(wavesrisetimes_LE)
        ydata_waves_LE = wavestestfreq_LE



        popt_waves_LE, pcov_waves_LE = curve_fit(reciprocal_2ndorder, ydata_waves_LE, xdata_waves_LE)

        fittimes_waves_LE = reciprocal_2ndorder(fitfreqs, *popt_waves_LE)
        times4tri_waves_LE = reciprocal_2ndorder(freqs4tri,  *popt_waves_LE)



        notnan = np.where(~np.isnan(fittimes_waves_LE))
        fitfreqs_waves_LE = fitfreqs[notnan]
        fittimes_waves_notnan_LE = fittimes_waves_LE[notnan]

        times4tri_waves_LE_dt = time2epoch(times4tri_waves_LE, xref_waves_LE)
        fittimes_corrected_waves_LE = time2epoch(fittimes_waves_notnan_LE, xref_waves_LE)

        # residuals waves Leading edge
        fittimes_waves_for_residuals_LE = reciprocal_2ndorder(np.array(wavestestfreq_LE), *popt_waves_LE)
        residuals_waves_LE = np.subtract(xdata_waves_LE, fittimes_waves_for_residuals_LE)

        if plot_residuals == True:
            plt.figure()
            plt.plot(residuals_waves_LE, "r.")
            plt.title("residuals WAVES LEADING EDGE")
            plt.xlabel("index")
            plt.ylabel("difference")
            plt.show(block=False)



        # Stereo A
        swaves_a_risetimes_LE = list(swaves_a_risetimes_h_LE)# list([datetime(2012, 6, 7, 19, 58, 8)]) +
        swaves_a_riseval_LE  =  list(swaves_a_riseval_h_LE)#list([np.average(swaves_a_riseval_h_LE)]) +
        swaves_a_testfreq_LE  = list(swaves_a_testfreq_h_LE)#list([0.15]) +

        xdata_swaves_a_LE, xref_swaves_a_LE = epoch2time(swaves_a_risetimes_LE)
        ydata_swaves_a_LE = swaves_a_testfreq_LE
        popt_swaves_a_LE, pcov_swaves_a_LE = curve_fit(reciprocal_2ndorder, ydata_swaves_a_LE, xdata_swaves_a_LE)


        fittimes_swaves_a_LE = reciprocal_2ndorder(fitfreqs, *popt_swaves_a_LE)
        times4tri_swaves_a_LE_dt = reciprocal_2ndorder(freqs4tri, *popt_swaves_a_LE)

        notnan = np.where(~np.isnan(fittimes_swaves_a_LE))
        fitfreqs_swaves_a_LE = fitfreqs[notnan]
        fittimes_swaves_a_notnan_LE = fittimes_swaves_a_LE[notnan]



        fittimes_corrected_swaves_a_LE = time2epoch(fittimes_swaves_a_notnan_LE, xref_swaves_a_LE)
        times4tri_swaves_a_LE_dt = time2epoch(times4tri_swaves_a_LE_dt, xref_swaves_a_LE)

        # residuals swaves a leading edge
        fittimes_swavesa_for_residuals_LE = reciprocal_2ndorder(np.array(swaves_a_testfreq_LE), *popt_swaves_a_LE)
        residuals_swaves_a_LE = np.subtract(xdata_swaves_a_LE, fittimes_swavesa_for_residuals_LE)

        if plot_residuals == True:
            plt.figure()
            plt.plot(residuals_swaves_a_LE, "r.")
            plt.title("residuals SWAVES A LEADING EDGE")
            plt.xlabel("index")
            plt.ylabel("difference")
            plt.show(block=False)

        # Stereo B
        swaves_b_risetimes_LE =  list(swaves_b_risetimes_h_LE)  # list( [datetime(2012, 6, 7, 19, 59, 29)]) +
        swaves_b_riseval_LE   =  list(swaves_b_riseval_h_LE) #list([np.average(swaves_a_riseval_h_LE)]) +
        swaves_b_testfreq_LE  = list(swaves_b_testfreq_h_LE) #list([0.15]) +

        xdata_swaves_b_LE, xref_swaves_b_LE = epoch2time(swaves_b_risetimes_LE)
        ydata_swaves_b_LE = swaves_b_testfreq_LE
        popt_swaves_b_LE, pcov_swaves_b_LE = curve_fit(reciprocal_2ndorder, ydata_swaves_b_LE, xdata_swaves_b_LE)


        fittimes_swaves_b_LE = reciprocal_2ndorder(fitfreqs, *popt_swaves_b_LE)
        times4tri_swaves_b_LE_dt = reciprocal_2ndorder(freqs4tri, *popt_swaves_b_LE)

        notnan = np.where(~np.isnan(fittimes_swaves_b_LE))
        fitfreqs_swaves_b_LE = fitfreqs[notnan]
        fittimes_swaves_b_notnan_LE = fittimes_swaves_b_LE[notnan]
        fittimes_corrected_swaves_b_LE = time2epoch(fittimes_swaves_b_notnan_LE, xref_swaves_b_LE)
        times4tri_swaves_b_LE_dt = time2epoch(times4tri_swaves_b_LE_dt, xref_swaves_b_LE)

        # residuals swaves b leading edge
        fittimes_swavesb_for_residuals_LE = reciprocal_2ndorder(np.array(swaves_b_testfreq_LE), *popt_swaves_b_LE)
        residuals_swaves_b_LE = np.subtract(xdata_swaves_b_LE, fittimes_swavesb_for_residuals_LE)
        if plot_residuals == True:
            plt.figure()
            plt.plot(residuals_swaves_b_LE, "r.")
            plt.title("residuals SWAVES B LEADING EDGE")
            plt.xlabel("index")
            plt.ylabel("difference")
            plt.show(block=False)





    # ---------------------------------------------------------------- #
    # BACKBONE
    # ---------------------------------------------------------------- #
    if backbone == True:

        #  Wind
        wavesrisetimes_BB =  list(waves_risetimes_l_BB) + list(waves_risetimes_h_BB)
        wavesriseval_BB   =  list(waves_riseval_l_BB)   + list(waves_riseval_h_BB)
        wavestestfreq_BB  =  list(waves_testfreq_l_BB)  + list(waves_testfreq_h_BB)

        xdata_waves_BB, xref_waves_BB = epoch2time(wavesrisetimes_BB)
        ydata_waves_BB = wavestestfreq_BB



        popt_waves_BB, pcov_waves_BB = curve_fit(reciprocal_2ndorder, ydata_waves_BB, xdata_waves_BB)

        fittimes_waves_BB = reciprocal_2ndorder(fitfreqs, *popt_waves_BB)
        times4tri_waves_BB = reciprocal_2ndorder(freqs4tri,  *popt_waves_BB)



        notnan = np.where(~np.isnan(fittimes_waves_BB))
        fitfreqs_waves_BB = fitfreqs[notnan]
        fittimes_waves_notnan_BB = fittimes_waves_BB[notnan]

        times4tri_waves_BB_dt = time2epoch(times4tri_waves_BB, xref_waves_BB)
        fittimes_corrected_waves_BB = time2epoch(fittimes_waves_notnan_BB, xref_waves_BB)

        # residuals waves Leading edge
        fittimes_waves_for_residuals_BB = reciprocal_2ndorder(np.array(wavestestfreq_BB), *popt_waves_BB)
        residuals_waves_BB = np.subtract(xdata_waves_BB, fittimes_waves_for_residuals_BB)

        if plot_residuals == True:
            plt.figure()
            plt.plot(residuals_waves_BB, "r.")
            plt.title("residuals WAVES BACKBONE")
            plt.xlabel("index")
            plt.ylabel("difference")
            plt.show(block=False)



        # Stereo A
        swaves_a_risetimes_BB = list(swaves_a_risetimes_h_BB)# list([datetime(2012, 6, 7, 19, 58, 8)]) +
        swaves_a_riseval_BB  =  list(swaves_a_riseval_h_BB)#list([np.average(swaves_a_riseval_h_BB)]) +
        swaves_a_testfreq_BB  = list(swaves_a_testfreq_h_BB)#list([0.15]) +

        xdata_swaves_a_BB, xref_swaves_a_BB = epoch2time(swaves_a_risetimes_BB)
        ydata_swaves_a_BB = swaves_a_testfreq_BB
        popt_swaves_a_BB, pcov_swaves_a_BB = curve_fit(reciprocal_2ndorder, ydata_swaves_a_BB, xdata_swaves_a_BB)


        fittimes_swaves_a_BB = reciprocal_2ndorder(fitfreqs, *popt_swaves_a_BB)
        times4tri_swaves_a_BB_dt = reciprocal_2ndorder(freqs4tri, *popt_swaves_a_BB)

        notnan = np.where(~np.isnan(fittimes_swaves_a_BB))
        fitfreqs_swaves_a_BB = fitfreqs[notnan]
        fittimes_swaves_a_notnan_BB = fittimes_swaves_a_BB[notnan]



        fittimes_corrected_swaves_a_BB = time2epoch(fittimes_swaves_a_notnan_BB, xref_swaves_a_BB)
        times4tri_swaves_a_BB_dt = time2epoch(times4tri_swaves_a_BB_dt, xref_swaves_a_BB)

        # residuals swaves a leading edge
        fittimes_swavesa_for_residuals_BB = reciprocal_2ndorder(np.array(swaves_a_testfreq_BB), *popt_swaves_a_BB)
        residuals_swaves_a_BB = np.subtract(xdata_swaves_a_BB, fittimes_swavesa_for_residuals_BB)

        if plot_residuals == True:
            plt.figure()
            plt.plot(residuals_swaves_a_BB, "r.")
            plt.title("residuals SWAVES A BACKBONE")
            plt.xlabel("index")
            plt.ylabel("difference")
            plt.show(block=False)

        # Stereo B
        swaves_b_risetimes_BB =  list(swaves_b_risetimes_h_BB)  # list( [datetime(2012, 6, 7, 19, 59, 29)]) +
        swaves_b_riseval_BB   =  list(swaves_b_riseval_h_BB) #list([np.average(swaves_a_riseval_h_BB)]) +
        swaves_b_testfreq_BB  = list(swaves_b_testfreq_h_BB) #list([0.15]) +

        xdata_swaves_b_BB, xref_swaves_b_BB = epoch2time(swaves_b_risetimes_BB)
        ydata_swaves_b_BB = swaves_b_testfreq_BB
        popt_swaves_b_BB, pcov_swaves_b_BB = curve_fit(reciprocal_2ndorder, ydata_swaves_b_BB, xdata_swaves_b_BB)


        fittimes_swaves_b_BB = reciprocal_2ndorder(fitfreqs, *popt_swaves_b_BB)
        times4tri_swaves_b_BB_dt = reciprocal_2ndorder(freqs4tri, *popt_swaves_b_BB)

        notnan = np.where(~np.isnan(fittimes_swaves_b_BB))
        fitfreqs_swaves_b_BB = fitfreqs[notnan]
        fittimes_swaves_b_notnan_BB = fittimes_swaves_b_BB[notnan]
        fittimes_corrected_swaves_b_BB = time2epoch(fittimes_swaves_b_notnan_BB, xref_swaves_b_BB)
        times4tri_swaves_b_BB_dt = time2epoch(times4tri_swaves_b_BB_dt, xref_swaves_b_BB)

        # residuals swaves b leading edge
        fittimes_swavesb_for_residuals_BB = reciprocal_2ndorder(np.array(swaves_b_testfreq_BB), *popt_swaves_b_BB)
        residuals_swaves_b_BB = np.subtract(xdata_swaves_b_BB, fittimes_swavesb_for_residuals_BB)
        if plot_residuals == True:
            plt.figure()
            plt.plot(residuals_swaves_b_BB, "r.")
            plt.title("residuals SWAVES B BACKBONE")
            plt.xlabel("index")
            plt.ylabel("difference")
            plt.show(block=False)





    # ---------------------------------------------------------------- #
    # Plotting TYPE IIIs
    # ---------------------------------------------------------------- #

    # WIND
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(9, 13))
    waves_spec_lfr.plot(axes=axes,norm=LogNorm(vmin=waves_mm_l[0], vmax=waves_mm_l[1]), cmap="jet")
    waves_spec_hfr.plot(axes=axes,norm=LogNorm(vmin=waves_mm_h[0], vmax=waves_mm_h[1]), cmap="jet")


    # LEADING EDGE
    if leadingedge ==True:
        # axes.plot(waves_risetimes_l_LE, waves_testfreq_l_LE, 'k*')
        # axes.plot(waves_risetimes_h_LE, waves_testfreq_h_LE, 'k*')
        axes.plot(wavesrisetimes_LE, wavestestfreq_LE, 'k*')
        axes.plot(fittimes_corrected_waves_LE,fitfreqs_waves_LE, "k--")
        axes.plot(fittimes_corrected_waves_LE,fitfreqs_waves_LE, "y--")
    # BACKBONE
    if backbone ==True:
        # axes.plot(waves_risetimes_l_BB, waves_testfreq_l_BB, 'k*')
        # axes.plot(waves_risetimes_h_BB, waves_testfreq_h_BB, 'k*')
        axes.plot(wavesrisetimes_BB, wavestestfreq_BB, 'k*')
        axes.plot(fittimes_corrected_waves_BB,fitfreqs_waves_BB, "k--")
        axes.plot(fittimes_corrected_waves_BB,fitfreqs_waves_BB, "y--")




    axes.set_ylim(reversed(axes.get_ylim()))
    axes.set_yscale('log')
    axes.set_xlim(datetime(YYYY, MM, dd, 19,28), datetime(YYYY, MM, dd, HH_1,mm_1))
    axes.set_ylim([freqfitmax, freqfitmin])
    plt.subplots_adjust(hspace=0.31)
    figfname = f"{figdir}/{YYYY}_{MM:02}_{dd:02}_WIND.png"
    plt.savefig(figfname, dpi='figure')

    plt.show(block=False)


    ###
    # ---------------------------------------------------------------------
    # FITTING TESTS
    # ---------------------------------------------------------------------
    if backbone == True:
        testfreqs = wavestestfreq_BB
        testtimes, testref = epoch2time(wavesrisetimes_BB)

    if leadingedge == True:
        testfreqs = wavestestfreq_LE
        testtimes,testref = epoch2time(wavesrisetimes_LE)

    popt_test, pcov_test = curve_fit(exponential_func3, testfreqs, testtimes)
    times_fit = exponential_func3(np.array(testfreqs), *popt_test)


    residuals = times_fit - testtimes
    if plot_residuals == True:
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))

        # Plot the data on each subplot
        axs[0].plot(testtimes,testfreqs, 'k*')
        axs[0].set_title('fit')
        axs[0].set_xlabel('time [s]')
        axs[0].set_ylabel('frequency MHz')
        axs[0].plot(times_fit, testfreqs,'r--')
        # axs[0].set_yscale("log")
        axs[0].set_yscale("linear")
        axs[0].semilogy(base=np.e)




        axs[1].plot(residuals, 'r.')
        axs[1].set_title('residuals')

        # Display the plot
        plt.show(block=False)

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------

    ###
    #  STEREO A
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(9, 13))
    swaves_a_spec_hfr.plot(axes=axes,norm=LogNorm(vmin=swaves_a_mm_h[0], vmax=swaves_a_mm_h[1]), cmap="jet")
    # swaves_a_spec_lfr.plot(axes=axes)

    # LEADING EDGE
    if leadingedge ==True:
        axes.plot(swaves_a_risetimes_h_LE, swaves_a_testfreq_h_LE, 'r*')
        axes.plot(fittimes_corrected_swaves_a_LE,fitfreqs_swaves_a_LE, "k--")
    # BACKBONE
    if backbone ==True:
        axes.plot(swaves_a_risetimes_h_BB, swaves_a_testfreq_h_BB, 'r*')
        axes.plot(fittimes_corrected_swaves_a_BB,fitfreqs_swaves_a_BB, "k--")


    axes.set_ylim(reversed(axes.get_ylim()))
    axes.set_yscale('log')
    axes.set_xlim(datetime(YYYY, MM, dd, HH_0,mm_0), datetime(YYYY, MM, dd, HH_1,mm_1))
    axes.set_ylim([freqfitmax, freqfitmin])
    plt.subplots_adjust(hspace=0.31)
    figfname = f"{figdir}/{YYYY}_{MM:02}_{dd:02}_STEA.png"
    plt.savefig(figfname, dpi='figure')
    plt.show(block=False)


    #  STEREO B
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(9, 13))
    swaves_b_spec_hfr.plot(axes=axes,norm=LogNorm(vmin=swaves_b_mm_h[0], vmax=swaves_b_mm_h[1]), cmap="jet")
    # swaves_b_spec_lfr.plot(axes=axes)

    # LEADING EDGE
    if leadingedge ==True:
        axes.plot(swaves_b_risetimes_h_LE, swaves_b_testfreq_h_LE, 'ro',markeredgecolor="w")
        axes.plot(fittimes_corrected_swaves_b_LE,fitfreqs_swaves_b_LE, "k--")

    if backbone ==True:
        axes.plot(swaves_b_risetimes_h_BB, swaves_b_testfreq_h_BB, 'ro',markeredgecolor="w")
        axes.plot(fittimes_corrected_swaves_b_BB,fitfreqs_swaves_b_BB, "k--")




    axes.set_ylim(reversed(axes.get_ylim()))
    axes.set_yscale('log')
    axes.set_xlim(datetime(YYYY, MM, dd, HH_0,25), datetime(YYYY, MM, dd, HH_1,mm_1))
    axes.set_ylim([freqfitmax, freqfitmin])

    plt.subplots_adjust(hspace=0.31)
    # axes.set_ylim([freq4trimax,freq4trimin])
    figfname = f"{figdir}/{YYYY}_{MM:02}_{dd:02}_STEB.png"
    plt.savefig(figfname, dpi='figure')
    plt.show(block=False)



    # ---------------------------------------------------------------- #
    # JOINT PLOT
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #
    # JOINT PLOT Horizontal
    # ---------------------------------------------------------------- #

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 9))

    waves_spec_lfr.plot(axes=axes[0], norm=LogNorm(vmin=waves_mm_l[0], vmax=waves_mm_l[1]), cmap="jet")
    waves_spec_hfr.plot(axes=axes[0], norm=LogNorm(vmin=waves_mm_h[0], vmax=waves_mm_h[1]), cmap="jet")
    swaves_a_spec_hfr.plot(axes=axes[1], norm=LogNorm(vmin=swaves_a_mm_h[0], vmax=swaves_a_mm_h[1]), cmap="jet")
    # swaves_spec[1].plot(axes=axes[1], norm=LogNorm(vmin=swaves_mm_h[0], vmax=swaves_mm_h[1]))
    swaves_b_spec_hfr.plot(axes=axes[2], norm=LogNorm(vmin=swaves_b_mm_h[0], vmax=swaves_b_mm_h[1]), cmap="jet")

    axes[0].set_title("WIND, WAVES, RAD1+RAD2")
    axes[1].set_title("STEREO A, SWAVES, HFR")
    axes[2].set_title("STEREO B, SWAVES, HFR")

    axes[0].set_ylabel("Frequency (MHz)")

    # # by default y-axis low to hight flip so moving away fro sun with time
    axes[0].set_ylim(reversed(axes[0].get_ylim()))
    axes[1].set_ylim(reversed(axes[1].get_ylim()))
    axes[2].set_ylim(reversed(axes[2].get_ylim()))


    # log y-axis
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    if leadingedge == True:
        times4tri_waves_LE_dt_plus30 = list(np.array(times4tri_waves_LE_dt)+timedelta(seconds=30))
        times4tri_waves_LE_dt_minus30 = list(np.array(times4tri_waves_LE_dt)-timedelta(seconds=30))

    if backbone == True:
        times4tri_waves_BB_dt_plus30 = list(np.array(times4tri_waves_BB_dt)+timedelta(seconds=30))
        times4tri_waves_BB_dt_minus30 = list(np.array(times4tri_waves_BB_dt)-timedelta(seconds=30))



    # Models Leading Edge
    if leadingedge ==True:
        axes[0].plot(wavesrisetimes_LE, wavestestfreq_LE, c = "white", markeredgecolor="red", marker="o", ls="")
        axes[1].plot(swaves_a_risetimes_h_LE, swaves_a_testfreq_h_LE, c = "white", markeredgecolor="red", marker="o", ls="")
        axes[2].plot(swaves_b_risetimes_h_LE, swaves_b_testfreq_h_LE,c = "white", markeredgecolor="red", marker="o", ls="")


        axes[0].plot(times4tri_waves_LE_dt_minus30, freqs4tri , "k--")
        axes[0].plot(np.array(times4tri_waves_LE_dt), freqs4tri , "k-")
        axes[0].plot(times4tri_waves_LE_dt_plus30, freqs4tri , "k--")

        axes[1].plot(np.array(times4tri_swaves_a_LE_dt)-timedelta(seconds=30), freqs4tri, "k--")
        axes[1].plot(np.array(times4tri_swaves_a_LE_dt), freqs4tri, "k-")
        axes[1].plot(np.array(times4tri_swaves_a_LE_dt)+timedelta(seconds=30), freqs4tri, "k--")

        axes[2].plot(np.array(times4tri_swaves_b_LE_dt)-timedelta(seconds=30), freqs4tri, "k--")
        axes[2].plot(np.array(times4tri_swaves_b_LE_dt), freqs4tri, "k-")
        axes[2].plot(np.array(times4tri_swaves_b_LE_dt)+timedelta(seconds=30), freqs4tri, "k--")

    # Models BACKBONE
    if backbone ==True:
        axes[0].plot(wavesrisetimes_BB, wavestestfreq_BB, c = "white", markeredgecolor="red", marker="o", ls="")
        axes[1].plot(swaves_a_risetimes_h_BB, swaves_a_testfreq_h_BB, c = "white", markeredgecolor="red", marker="o", ls="")
        axes[2].plot(swaves_b_risetimes_h_BB, swaves_b_testfreq_h_BB,c = "white", markeredgecolor="red", marker="o", ls="")


        axes[0].plot(times4tri_waves_BB_dt_minus30, freqs4tri , "k--")
        axes[0].plot(np.array(times4tri_waves_BB_dt), freqs4tri , "k-")
        axes[0].plot(times4tri_waves_BB_dt_plus30, freqs4tri , "k--")

        axes[1].plot(np.array(times4tri_swaves_a_BB_dt)-timedelta(seconds=30), freqs4tri, "k--")
        axes[1].plot(np.array(times4tri_swaves_a_BB_dt), freqs4tri, "k-")
        axes[1].plot(np.array(times4tri_swaves_a_BB_dt)+timedelta(seconds=30), freqs4tri, "k--")

        axes[2].plot(np.array(times4tri_swaves_b_BB_dt)-timedelta(seconds=30), freqs4tri, "k--")
        axes[2].plot(np.array(times4tri_swaves_b_BB_dt), freqs4tri, "k-")
        axes[2].plot(np.array(times4tri_swaves_b_BB_dt)+timedelta(seconds=30), freqs4tri, "k--")




    axes[0].set_ylim([freq4trimax,freq4trimin])
    # axes[1].set_ylim([freqlimmax, 0.2])
    # axes[2].set_ylim([freqlimmax, 0.2])


    axes[1].set_xlim(datetime(YYYY, MM, dd, HH_0,mm_0), datetime(YYYY, MM, dd, HH_1,mm_1))
    plt.subplots_adjust(left=0.041, bottom=0.096, right=0.984, top=0.93, wspace=0.132, hspace=0.31)

    plt.tick_params(axis='y', which='minor')
    axes[0].yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    plt.tick_params(axis='y', which='major')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    figfname = f"{figdir}/{YYYY}_{MM:02}_{dd:02}_Horizontal.png"
    plt.savefig(figfname, dpi='figure')
    plt.show(block=False)





    if leadingedge == True:
        wavesrisetimes_LE_stamp = datetime2timestamp(wavesrisetimes_LE)
        spl = CubicSpline(wavestestfreq_LE, wavesrisetimes_LE_stamp)

        wavesrisetimes_LE_SWAVESfrequencies = timestamp2datetime(spl(swaves_a_testfreq_LE))
        wavesrisetimes_LE_SWAVESfrequencies = timestamp2datetime(interp(swaves_a_testfreq_LE,wavestestfreq_LE, wavesrisetimes_LE_stamp ))

        plt.figure()
        plt.plot(times4tri_waves_LE_dt, freqs4tri, 'r--', label="WAVES")
        plt.plot(times4tri_swaves_a_LE_dt, freqs4tri, 'b--', label="SWAVES_A")
        plt.plot(times4tri_swaves_b_LE_dt, freqs4tri, 'g--', label="SWAVES_B")

        plt.plot(wavesrisetimes_LE, wavestestfreq_LE, 'r*')
        plt.plot(wavesrisetimes_LE_SWAVESfrequencies, swaves_b_testfreq_LE, 'k.')  # WAVES datapoints interpolated to match the SWAVES frequency channels
        plt.plot(swaves_a_risetimes_LE, swaves_a_testfreq_LE, 'b*')
        plt.plot(swaves_b_risetimes_LE, swaves_b_testfreq_LE, 'g*')

        plt.xlabel('epoch')
        plt.ylabel('freq')
        plt.yscale('log')
        plt.legend(loc=1)
        plt.title("Type IIIs datapoints combined")
        plt.show(block=False)

    if backbone == True:
        wavesrisetimes_BB_stamp = datetime2timestamp(wavesrisetimes_BB)
        spl = CubicSpline(wavestestfreq_BB, wavesrisetimes_BB_stamp)

        wavesrisetimes_BB_SWAVESfrequencies = timestamp2datetime(spl(swaves_a_testfreq_BB))
        wavesrisetimes_BB_SWAVESfrequencies = timestamp2datetime(
            interp(swaves_a_testfreq_BB, wavestestfreq_BB, wavesrisetimes_BB_stamp))

        plt.figure()
        plt.plot(times4tri_waves_BB_dt, freqs4tri, 'r--', label="WAVES")
        plt.plot(times4tri_swaves_a_BB_dt, freqs4tri, 'b--', label="SWAVES_A")
        plt.plot(times4tri_swaves_b_BB_dt, freqs4tri, 'g--', label="SWAVES_B")

        plt.plot(wavesrisetimes_BB, wavestestfreq_BB, 'r*')
        plt.plot(wavesrisetimes_BB_SWAVESfrequencies, swaves_b_testfreq_BB, 'g.')
        plt.plot(swaves_a_risetimes_BB, swaves_a_testfreq_BB, 'b*')
        plt.plot(swaves_b_risetimes_BB, swaves_b_testfreq_BB, 'g*')

        plt.xlabel('epoch')
        plt.ylabel('freq')
        plt.yscale('log')
        plt.legend(loc=1)
        plt.show(block=False)

    typeIIIdir = mkdirectory(f"Data/TypeIII/{YYYY}_{MM:02}_{dd:02}/")
    if leadingedge == True:
        if (np.array_equal(fitfreqs, fitfreqs_waves_LE) and np.array_equal(fitfreqs,fitfreqs_swaves_a_LE) and np.array_equal(fitfreqs, fitfreqs_swaves_b_LE)):
            typeIII_LE = ({
                'Freqs': freqs4tri,
                'WindTime': times4tri_waves_LE_dt,#times4tri_waves_LE_dt_minus30, #
                'StereoATime': times4tri_swaves_a_LE_dt,
                'StereoBTime': times4tri_swaves_b_LE_dt,
            })
            savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}_WIND_STEREO_A_STEREO_B_Freqs_{freq4trimin}_{freq4trimax}_LE_HR.pkl'
            with open(savedfilepath, 'wb') as outp:
                pickle.dump(typeIII_LE, outp, pickle.HIGHEST_PROTOCOL)
            print(f"Saved results: {savedfilepath}")

            typeIII_LE = ({
                'Freqs': swaves_a_testfreq_LE,
                'WindTime': wavesrisetimes_LE_SWAVESfrequencies,#times4tri_waves_LE_dt_minus30, #
                'StereoATime': swaves_a_risetimes_LE,
                'StereoBTime': swaves_b_risetimes_LE,
            })
            savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}_WIND_STEREO_A_STEREO_B_Freqs_{freq4trimin}_{freq4trimax}_LE_HR_SCATTER.pkl'
            with open(savedfilepath, 'wb') as outp:
                pickle.dump(typeIII_LE, outp, pickle.HIGHEST_PROTOCOL)
            print(f"Saved results: {savedfilepath}")

        else:
            print(f"Missing frequencies in one of the spacecraft. Poor fit of the radio burst.")
    if backbone == True:
        if (np.array_equal(fitfreqs, fitfreqs_waves_BB) and np.array_equal(fitfreqs,fitfreqs_swaves_a_BB) and np.array_equal(fitfreqs, fitfreqs_swaves_b_BB)):
            typeIII_BB = ({
                'Freqs': freqs4tri,
                'WindTime': times4tri_waves_BB_dt,#times4tri_waves_BB_dt_minus30, #
                'StereoATime': times4tri_swaves_a_BB_dt,
                'StereoBTime': times4tri_swaves_b_BB_dt,
            })
            savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}_WIND_STEREO_A_STEREO_B_Freqs_{freq4trimin}_{freq4trimax}_BB_HR.pkl'
            with open(savedfilepath, 'wb') as outp:
                pickle.dump(typeIII_BB, outp, pickle.HIGHEST_PROTOCOL)
            print(f"Saved results: {savedfilepath}")

            typeIII_BB = ({
                'Freqs': swaves_a_testfreq_BB,
                'WindTime': wavesrisetimes_BB_SWAVESfrequencies,#times4tri_waves_BB_dt_minus30, #
                'StereoATime': swaves_a_risetimes_BB,
                'StereoBTime': swaves_b_risetimes_BB,
            })
            savedfilepath = f'{typeIIIdir}typeIII_{YYYY}{MM:02}{dd:02}_{HH_0:02}{mm_0:02}_WIND_STEREO_A_STEREO_B_Freqs_{freq4trimin}_{freq4trimax}_BB_HR_SCATTER.pkl'
            with open(savedfilepath, 'wb') as outp:
                pickle.dump(typeIII_BB, outp, pickle.HIGHEST_PROTOCOL)
            print(f"Saved results: {savedfilepath}")

        else:
            print(f"Missing frequencies in one of the spacecraft. Poor fit of the radio burst.")
