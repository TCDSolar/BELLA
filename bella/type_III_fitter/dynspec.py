import argparse
from datetime import datetime, timedelta

import cdflib
import matplotlib as mpl
import numpy as np
import pyspedas
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pytplot import get_data
from scipy import stats
from scipy.interpolate import interp2d

import astropy.units as u
from astropy.constants import R_sun, au
from astropy.time import Time

from radiospectra.spectrogram import Spectrogram  # in the process of updating old spectrogram

from sunpy.net import attrs as a

r_sun = R_sun.value
AU=au.value

##########################################################################
#            Functions to LOAD SC dyn spectra
##########################################################################
def waves_spec(start, endt,datatype="RAD1", bg_subtraction=False, lighttravelshift=0):
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
    print("Wind WAVES downloaded: wind_variables:")
    print(wind_variables)
    waves_data = get_data(f'E_VOLTAGE_{datatype}')

    if datatype=='RAD2':
        frequencies = np.insert(waves_data.v, 0, 1000, axis=0) # fill gap at 1MHz
        waves_freqs_MHz = frequencies/1e3 * u.MHz
    else:
        waves_freqs_MHz = waves_data.v/ 1e3 * u.MHz

    waves_timestamps = waves_data.times
    waves_times = Time(Time(np.array([(datetime.utcfromtimestamp(t_)+timedelta(seconds=lighttravelshift)) for t_ in waves_timestamps])).isot)

    if datatype=='RAD2':
        waves_psdarray = waves_data.y
        waves_psdarray = np.insert(waves_psdarray, 0, waves_psdarray[:,0], axis=1)  # fill gap at 1MHz
    else:
        waves_psdarray = waves_data.y

    meta = {
        'observatory': "WIND",
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

def local_waves_spec_l2_60s(file, datatype='RAD1', kind='SMEAN', bg_subtraction=False, lighttravelshift=0):
    from bella.type_III_fitter.read_wi_wa_l2_60s_data import \
        read_l2_60s  # This file can be found at https://cdpp-archive.cnes.fr/
    header, data = read_l2_60s(file)

    frequencies = data['FKHZ']
    waves_epoch_dt = data['TIME']
    waves_times = Time(Time(np.array([(t_+timedelta(seconds=lighttravelshift)) for t_ in waves_epoch_dt])).isot)

    if datatype=='RAD2':
        frequencies = list([1000.0]) + list(frequencies)
        waves_freqs_MHz = np.array(frequencies)/1e3*u.MHz
    else:
        waves_freqs_MHz = np.array(frequencies)/1e3*u.MHz

    if datatype=='RAD2':
        waves_psdarray = data[kind]
        waves_psdarray = np.insert(waves_psdarray, 0, waves_psdarray[:,0], axis=1)
    else:
        waves_psdarray = data[kind]

    meta = {
        'observatory': "WIND",
        'instrument': "WAVES",
        'detector': datatype,
        'freqs': waves_freqs_MHz[:-1],
        'times': waves_times[:-1],
        'wavelength': a.Wavelength(waves_freqs_MHz[0], waves_freqs_MHz[-1]),
        'start_time': waves_times[0],
        'end_time': waves_times[-1]
    }
    waves_spec = Spectrogram(waves_psdarray.T, meta)

    if bg_subtraction:
        waves_spec.data = backSub(waves_spec.data.T).T

    return waves_spec


def swaves_highres_spec(start, endt, probe='a', datatype='hfr', bg_subtraction=False, lighttravelshift=0):
    """
    Default values for copy and paste
    start=mintime
    endt = maxtime
    probe='a'
    datatype='hfr'
    bg_subtraction=False
    lighttravelshift=0




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

    if stereo_variables == []:
        swaves_freqs_MHz = np.linspace(0.1, 16, 319) * u.MHz

        cadence = timedelta(seconds=38)
        swaves_epoch = generate_datetime_range(start, endt, cadence)
        swaves_timestamps = [dt.timestamp() for dt in swaves_epoch]
        swaves_times = Time(Time(np.array([datetime.utcfromtimestamp(t_) for t_ in swaves_timestamps])).isot)

        swaves_psdarray = np.zeros((len(swaves_times),len(swaves_freqs_MHz)))

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

    else:
        swaves_data = get_data('PSD_FLUX')

        swaves_freqs_MHz = swaves_data.v/ 1e6 * u.MHz

        swaves_timestamps = swaves_data.times
        swaves_times = Time(Time(np.array([(datetime.utcfromtimestamp(t_)+timedelta(seconds=lighttravelshift)) for t_ in swaves_timestamps])).isot)
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
def swaves_stokesV(start, endt, probe='a', datatype='hfr', bg_subtraction=False, lighttravelshift=0):
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

    if stereo_variables == []:
        swaves_freqs_MHz = np.linspace(0.1, 16, 319) * u.MHz

        cadence = timedelta(seconds=38)
        swaves_epoch = generate_datetime_range(start, endt, cadence)
        swaves_timestamps = [dt.timestamp() for dt in swaves_epoch]
        swaves_times = Time(Time(np.array([datetime.utcfromtimestamp(t_) for t_ in swaves_timestamps])).isot)

        swaves_psdarray = np.zeros((len(swaves_times),len(swaves_freqs_MHz)))

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

    else:
        swaves_data = get_data('STOKES_V')
        #swaves_data_stokes_V = get_data('STOKES_V')
        # swaves_data_stokes_I = get_data('STOKES_I')

        # swaves_data_VdivI = np.divide(swaves_data_stokes_V.y, swaves_data_stokes_I.y)

        # swaves_data = swaves_data_stokes_V
        swaves_freqs_MHz = swaves_data.v/ 1e6 * u.MHz

        swaves_timestamps = swaves_data.times
        swaves_times = Time(Time(np.array([(datetime.utcfromtimestamp(t_)+timedelta(seconds=lighttravelshift)) for t_ in swaves_timestamps])).isot)



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
    rpw_l2_hfr.varget('SAMPLE_TIME')

    agc1 = rpw_l2_hfr.varget('AGC1')
    agc2 = rpw_l2_hfr.varget('AGC2')

    rpw_l2_hfr.varget('FLUX_DENSITY1')
    rpw_l2_hfr.varget('FLUX_DENSITY2')

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


##########################################################################
#            Other functions
##########################################################################
def backSub(data, percentile=1):
    """ Background subtraction:
        This function has been modified from Eoin Carley's backsub function
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
    # Divide through by this average spec.
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


    if plot is True:
        plt.figure()
        ax = plt.gca()
        plt.plot_date(time_idx, dtime, ".")
        plt.xlabel(f"Time {time_idx[0].year}/{time_idx[0].month:02}/{time_idx[0].day:02} ")
        plt.ylabel("cadence")
        plt.title(title)
        formatter = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(formatter)

        plt.show(block=False)

    if (method=="average") or (method=="mean"):
        return np.mean(dtime)
    elif method=="max":
        return np.amax(dtime)
    elif method=="min":
        return np.amin(dtime)
    elif method=="mode":
        return stats.mode(dtime)[0][0]
    else:
        print("method can only be mode, average, max or min")

def check_spectro_cadence(spectrogram, type="mean"):
    # Take the difference between items
    cad_np = np.diff(spectrogram.times.datetime)
    if type == "mean":
        return np.mean(cad_np)
    elif type == "mode":
        mode_result = stats.mode(cad_np)
        return mode_result.mode[0]
    elif type == "min":
        return np.min(cad_np)
    elif type == "max":
        return np.max(cad_np)
    else:
        print("Only accepted types: 'mean', 'mode', 'min', 'max'")


def f_to_angs(f_mhz,c=299792458):
    angstrom = (c / (f_mhz * 10 ** 6)) * 10 ** 10
    return angstrom

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def generate_datetime_range(start_time, end_time, cadence):
    datetime_range = []
    current_time = start_time

    while current_time <= end_time:
        datetime_range.append(current_time)
        current_time += cadence

    return datetime_range

def resample_spectra(data, time, frequency, new_cadence, new_freq_resolution):
    """
    Resample a dynamic spectra data to a specified cadence and frequency resolution.

    Parameters:
    data (2D array): The original dynamic spectra data.
    time (1D array): The time/epoch array corresponding to the data.
    frequency (1D array): The frequency array corresponding to the data.
    new_cadence (float): The desired cadence in seconds.
    new_freq_resolution (float): The desired frequency resolution.

    Returns:
    2D array: The resampled dynamic spectra data.
    """
    # Create new time and frequency grids
    new_time = np.arange(time[0], time[-1], new_cadence)
    new_frequency = np.arange(frequency[0], frequency[-1], new_freq_resolution)

    # Interpolate
    interpolator = interp2d(time, frequency, data, kind='linear')
    new_data = interpolator(new_time, new_frequency)

    return new_data

if __name__=="__main__":
    # Fido
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Convert a date string to a datetime object")

    # Define a command line argument for the date
    parser.add_argument('-d', '--date', type=str, default=None, help="Input date in the format 'YYYYMMDD'")

    # Parse the command line arguments
    args = parser.parse_args()
    if args.date is None:
        print("No date argument provided. Default Manual date:")
        YYYY = 2011
        MM = 6
        dd = 24

    else:
        try:
            # Attempt to parse the input date string and convert it to a datetime object
            date_object = datetime.strptime(args.date, "%Y%m%d")
            print("Datetime:", date_object)
            YYYY = date_object.year
            MM = date_object.month
            dd = date_object.day
        except ValueError:
            print("Invalid date format. Please use 'YYYYMMDD' format.")

    HH_0 = 0
    mm_0 = 0
    HH_1 = 23
    mm_1 = 59
    #
    background_subtraction = True

    mintime = datetime(YYYY, MM, dd, HH_0,mm_0)
    maxtime =  datetime(YYYY, MM, dd, HH_1,mm_1)
    timelims = [mintime,maxtime]


    # Waves RAD 1 ( low freqs)
    waves_spec_lfr = waves_spec(mintime, maxtime, datatype='RAD1', bg_subtraction=background_subtraction)
    # Waves RAD 2 (High freqs)
    waves_spec_hfr = waves_spec(mintime, maxtime, datatype='RAD2', bg_subtraction=background_subtraction)

    # SWAVES A HFR
    swaves_a_spec_hfr = swaves_highres_spec(mintime, maxtime, probe='a', datatype='hfr', bg_subtraction=True)
    # SWAVES A LFR
    swaves_a_spec_lfr = swaves_highres_spec(mintime, maxtime, probe='a', datatype='lfr', bg_subtraction=True)

    # SWAVES B HFR
    swaves_b_spec_hfr = swaves_highres_spec(mintime, maxtime, probe='b', datatype='hfr', bg_subtraction=True)

    # SWAVES B HFR
    swaves_b_spec_lfr = swaves_highres_spec(mintime, maxtime, probe='b', datatype='lfr', bg_subtraction=True)


    # FILL BLANK GAP
    freqs_fill = [waves_spec_lfr.frequencies[-1].value,waves_spec_hfr.frequencies[0].value]* u.MHz
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

    waves_mm_l = np.percentile(waves_spec_lfr.data, [20,99])
    waves_mm_h = np.percentile(waves_spec_hfr.data, [20,99])

    swaves_a_mm_l = np.percentile(swaves_a_spec_lfr.data, [40,99.99])
    swaves_a_mm_h = np.percentile(swaves_a_spec_hfr.data, [10,99.99])

    swaves_b_mm_l = np.percentile(swaves_b_spec_lfr.data, [40,99.99])
    swaves_b_mm_h = np.percentile(swaves_b_spec_hfr.data, [10,99.99])



    # ---------------------------------------------------------------- #
    # JOINT PLOT
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #
    # JOINT PLOT Horizontal
    # ---------------------------------------------------------------- #

    my_cmap = mpl.cm.jet
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(20, 9))

    swaves_a_spec_lfr.plot(axes=axes[0], norm=LogNorm(vmin=swaves_a_mm_l[0], vmax=swaves_a_mm_l[1]), cmap=my_cmap)
    swaves_a_spec_hfr.plot(axes=axes[0], norm=LogNorm(vmin=swaves_a_mm_h[0], vmax=swaves_a_mm_h[1]), cmap=my_cmap)

    swaves_b_spec_lfr.plot(axes=axes[1], norm=LogNorm(vmin=swaves_b_mm_l[0], vmax=swaves_b_mm_l[1]), cmap=my_cmap)
    swaves_b_spec_hfr.plot(axes=axes[1], norm=LogNorm(vmin=swaves_b_mm_h[0], vmax=swaves_b_mm_h[1]), cmap=my_cmap)

    waves_spec_lfr.plot(axes=axes[2],norm=LogNorm(vmin=waves_mm_l[0], vmax=waves_mm_l[1]), cmap="jet")
    waves_spec_hfr.plot(axes=axes[2],norm=LogNorm(vmin=waves_mm_h[0], vmax=waves_mm_h[1]), cmap="jet")



    axes[0].set_title("STEREO A, SWAVES, HFR + LFR")
    axes[1].set_title("STEREO B, SWAVES, HFR + LFR")
    axes[2].set_title("Wind, Waves, Rad1 + Rad2")

    axes[0].set_ylabel("Frequency (MHz)")
    axes[1].set_ylabel("Frequency (MHz)")
    axes[2].set_ylabel("Frequency (MHz)")

    # # # by default y-axis low to height flip so moving away from sun with time
    axes[0].set_ylim(reversed(axes[0].get_ylim()))
    axes[1].set_ylim(reversed(axes[1].get_ylim()))
    axes[2].set_ylim(reversed(axes[2].get_ylim()))

    #
    # log y-axis
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    axes[0].set_xlim(datetime(YYYY, MM, dd, HH_0, mm_0), datetime(YYYY, MM, dd, HH_1, mm_1))
    plt.subplots_adjust(left=0.041, bottom=0.096, right=0.984, top=0.93, wspace=0.132, hspace=0.31)

    # plt.tick_params(axis='y', which='minor')
    # axes[0].yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    # plt.tick_params(axis='y', which='major')
    # axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # figfname = f"Figures/{YYYY}_{MM:02}_{dd:02}/{YYYY}_{MM:02}_{dd:02}_Horizontal.png"
    # plt.savefig(figfname, dpi='figure')

    plt.tight_layout()
    plt.show(block=False)
