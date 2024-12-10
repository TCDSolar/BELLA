#
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#
import astropy.units as u
from astropy.time import Time

import argparse

import pyspedas  # pip install git+https://github.com/STBadman/pyspedas
from pytplot import get_data
from radiospectra.spectrogram import Spectrogram  # in the process of updating old spectrogram

# from matplotlib.ticker import FormatStrFormatter,LogFormatter
#
# from datetime import datetime
# from datetime import timedelta
#
# from astropy.constants import c, m_e, R_sun, e, eps0, au
# r_sun = R_sun.value
# AU=au.value
#
from sunpy.net import attrs as a

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.family': "Times New Roman"})

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

def rpw_spec(start, endt, datatype='hfr-surv', bg_subtraction=False):
    """
    Downloads and processes rpw spectrogram data from the SolO mission.

    Parameters
    ----------
    start : datetime
        The start time of the data range to be downloaded.
    endt : datetime
        The end time of the data range to be downloaded.
    datatype : str, optional
        The type of SolO data to use. Default is 'hfr'. Alternative is 'lfr'
    bg_subtraction : bool, optional
        Whether or not to perform background subtraction on the spectrogram. Default is False.

    Returns
    -------
    rpw_spec : `~sunpy.timeseries.Spectrogram`
        A `sunpy.timeseries.Spectrogram` object containing the SolO spectrogram data. please see radiospectra https://github.com/sunpy/radiospectra

    Notes
    -----
    This function uses the `pyspedas` and sunpy's 'radiospectra' libraries to download and process the SolO data. The resulting
    spectrogram object includes metadata such as the observatory, instrument, detector, frequencies, and time range.
    """

    if datatype == 'hfr':
        datatype='hfr-surv'
    elif datatype == 'lfr':
        datatype = 'lfr-surv'

    solo_variables = pyspedas.solo.rpw([start.strftime("%m/%d/%Y %H:%M:%S"), endt.strftime("%m/%d/%Y %H:%M:%S")], datatype = datatype)
    print(f"solo {datatype.upper()} downloaded: solo_variables:")
    print(solo_variables)

    if datatype == 'rpw_hfr':
        rpw_data = get_data('AGC1')

    elif datatype == 'rpw_lfr':
        rpw_data = get_data('solo_fld_l2_rpw_lfr_auto_averages_ch0_V1V2')

    rpw_freqs_MHz = rpw_data.v[0]/ 1e6 * u.MHz

    rpw_timestamps = rpw_data.times
    rpw_times = Time(Time(np.array([datetime.utcfromtimestamp(t_) for t_ in rpw_timestamps])).isot)

    rpw_psdarray = rpw_data.y

    meta = {
        'observatory': "SolO",
        'instrument': "FIELDS",
        'detector': datatype.upper(),
        'freqs': rpw_freqs_MHz,
        'times': rpw_times,
        'wavelength': a.Wavelength(rpw_freqs_MHz[0], rpw_freqs_MHz[-1]),
        'start_time': rpw_times[0],
        'end_time': rpw_times[-1]
    }
    rpw_spec = Spectrogram(rpw_psdarray.T, meta)

    if bg_subtraction:
        rpw_spec.data = backSub(rpw_spec.data.T).T

    return rpw_spec

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
        YYYY = 2021
        MM = 12
        dd = 4
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
    # start = mintime
    # endt = maxtime


    # solo HFR
    rpw_spec_hfr =rpw_spec(mintime, maxtime, datatype='hfr', bg_subtraction=True)
    # solo LFR
    rpw_spec_lfr = rpw_spec(mintime, maxtime, datatype='lfr', bg_subtraction=True)


    rpw_mm_l = np.percentile(rpw_spec_lfr.data, [40, 99.99])
    rpw_mm_h = np.percentile(rpw_spec_hfr.data, [10, 99.99])


    # ---------------------------------------------------------------- #
    # Plotting TYPE IIIs
    # ---------------------------------------------------------------- #
    # solo
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(25, 9))
    rpw_spec_lfr.plot(axes=axes,norm=LogNorm(vmin=rpw_mm_l[0], vmax=rpw_mm_l[1]), cmap="jet")
    rpw_spec_hfr.plot(axes=axes,norm=LogNorm(vmin=rpw_mm_h[0], vmax=rpw_mm_h[1]), cmap="jet")

    axes.set_ylabel("Frequency (MHz)")
    axes.set_title("SOLO, FIELDS, HFR + LFR")

    axes.set_ylim(reversed(axes.get_ylim()))
    axes.set_yscale('log')
    axes.set_xlim(datetime(YYYY, MM, dd, HH_0,mm_0), datetime(YYYY, MM, dd, HH_1,mm_1))
    # axes.set_ylim([freqlimmax, freqlimmin])
    plt.subplots_adjust(hspace=0.31)
    plt.tight_layout()
    plt.show(block=False)
