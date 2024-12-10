#
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt

#
import astropy.units as u
from astropy.time import Time

# import pickle
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

# def rfs_spec(start, endt, datatype='rfs_hfr',stokes="I", bg_subtraction=False, lighttravelshift=0):
#     """
#     Downloads and processes RFS spectrogram data from the PSP mission.
#
#     Parameters
#     ----------
#     start : datetime
#         The start time of the data range to be downloaded.
#     endt : datetime
#         The end time of the data range to be downloaded.
#     datatype : str, optional
#         The type of PSP data to use. Default is 'hfr'. Alternative is 'lfr'
#     bg_subtraction : bool, optional
#         Whether or not to perform background subtraction on the spectrogram. Default is False.
#
#     Returns
#     -------
#     psp_spec : `~sunpy.timeseries.Spectrogram`
#         A `sunpy.timeseries.Spectrogram` object containing the PSP spectrogram data. please see radiospectra https://github.com/sunpy/radiospectra
#
#     Notes
#     -----
#     This function uses the `pyspedas` and sunpy's 'radiospectra' libraries to download and process the PSP data. The resulting
#     spectrogram object includes metadata such as the observatory, instrument, detector, frequencies, and time range.
#     """
#     ['psp_fld_l2_rfs_hfr_auto_averages_ch0_V1V2',
#      'psp_fld_l2_rfs_hfr_auto_averages_ch1_V3V4',
#      'psp_fld_l2_rfs_hfr_auto_peaks_ch0_V1V2',
#      'psp_fld_l2_rfs_hfr_auto_peaks_ch1_V3V4',
#      'psp_fld_l2_rfs_hfr_cross_im_V1V2_V3V4',
#      'psp_fld_l2_rfs_hfr_cross_re_V1V2_V3V4',
#      'psp_fld_l2_rfs_hfr_coher_V1V2_V3V4',
#      'psp_fld_l2_rfs_hfr_phase_V1V2_V3V4',
#      'psp_fld_l2_rfs_hfr_averages',
#      'psp_fld_l2_rfs_hfr_peaks',
#      'psp_fld_l2_rfs_hfr_ch0',
#      'psp_fld_l2_rfs_hfr_ch1']
#
#     if datatype == 'hfr':
#         datatype='rfs_hfr'
#     elif datatype == 'lfr':
#         datatype = 'rfs_lfr'
#
#     psp_variables = pyspedas.psp.fields([start.strftime("%m/%d/%Y %H:%M:%S"), endt.strftime("%m/%d/%Y %H:%M:%S")], datatype = datatype)
#     print(f"psp {datatype.upper()} downloaded: psp_variables:")
#     print(psp_variables)
#
#     if datatype == 'rfs_hfr':
#         rfs_data_V1V2 = get_data('psp_fld_l2_rfs_hfr_auto_averages_ch0_V1V2')
#         rfs_data_V3V4 = get_data('psp_fld_l2_rfs_hfr_auto_averages_ch0_V3V4')
#
#     elif datatype == 'rfs_lfr':
#         rfs_data_V1V2 = get_data('psp_fld_l2_rfs_lfr_auto_averages_ch0_V1V2')
#         rfs_data_V3V4 = get_data('psp_fld_l2_rfs_lfr_auto_averages_ch0_V3V4')
#
#     if stokes=="I":
#         stokesI = np.squared(rfs_data_V1V2.y)+np.squared(rfs_data_V3V4.y)
#     if stokes == "V":
#         stokesV = np.squared(rfs_data_V1V2.y)-np.squared(rfs_data_V3V4.y)
#         rfs_psdarray = stokesV
#
#     rfs_freqs_MHz = rfs_data_V1V2.v[0]/ 1e6 * u.MHz
#
#     rfs_timestamps = rfs_data_V1V2.times
#     rfs_times = Time(Time(np.array([(datetime.utcfromtimestamp(t_)+timedelta(seconds=lighttravelshift)) for t_ in rfs_timestamps])).isot)
#
#
#
#     meta = {
#         'observatory': f"psp",
#         'instrument': "FIELDS",
#         'detector': datatype.upper(),
#         'freqs': rfs_freqs_MHz,
#         'times': rfs_times,
#         'wavelength': a.Wavelength(rfs_freqs_MHz[0], rfs_freqs_MHz[-1]),
#         'start_time': rfs_times[0],
#         'end_time': rfs_times[-1]
#     }
#     rfs_spec = Spectrogram(rfs_psdarray.T, meta)
#
#     if bg_subtraction:
#         rfs_spec.data = backSub(rfs_spec.data.T).T
#
#     return rfs_spec
def rfs_spec(start, endt, datatype='rfs_hfr', bg_subtraction=False, lighttravelshift=0):
    """
    Downloads and processes RFS spectrogram data from the PSP mission.

    Parameters
    ----------
    start : datetime
        The start time of the data range to be downloaded.
    endt : datetime
        The end time of the data range to be downloaded.
    datatype : str, optional
        The type of PSP data to use. Default is 'hfr'. Alternative is 'lfr'
    bg_subtraction : bool, optional
        Whether or not to perform background subtraction on the spectrogram. Default is False.

    Returns
    -------
    psp_spec : `~sunpy.timeseries.Spectrogram`
        A `sunpy.timeseries.Spectrogram` object containing the PSP spectrogram data. please see radiospectra https://github.com/sunpy/radiospectra

    Notes
    -----
    This function uses the `pyspedas` and sunpy's 'radiospectra' libraries to download and process the PSP data. The resulting
    spectrogram object includes metadata such as the observatory, instrument, detector, frequencies, and time range.
    """
    ['psp_fld_l2_rfs_hfr_auto_averages_ch0_V1V2',
     'psp_fld_l2_rfs_hfr_auto_averages_ch1_V3V4',
     'psp_fld_l2_rfs_hfr_auto_peaks_ch0_V1V2',
     'psp_fld_l2_rfs_hfr_auto_peaks_ch1_V3V4',
     'psp_fld_l2_rfs_hfr_cross_im_V1V2_V3V4',
     'psp_fld_l2_rfs_hfr_cross_re_V1V2_V3V4',
     'psp_fld_l2_rfs_hfr_coher_V1V2_V3V4',
     'psp_fld_l2_rfs_hfr_phase_V1V2_V3V4',
     'psp_fld_l2_rfs_hfr_averages',
     'psp_fld_l2_rfs_hfr_peaks',
     'psp_fld_l2_rfs_hfr_ch0',
     'psp_fld_l2_rfs_hfr_ch1']

    if datatype == 'hfr':
        datatype='rfs_hfr'
    elif datatype == 'lfr':
        datatype = 'rfs_lfr'

    psp_variables = pyspedas.psp.fields([start.strftime("%m/%d/%Y %H:%M:%S"), endt.strftime("%m/%d/%Y %H:%M:%S")], datatype = datatype)
    print(f"psp {datatype.upper()} downloaded: psp_variables:")
    print(psp_variables)

    if datatype == 'rfs_hfr':
        rfs_data = get_data('psp_fld_l2_rfs_hfr_auto_averages_ch0_V1V2')

    elif datatype == 'rfs_lfr':
        rfs_data = get_data('psp_fld_l2_rfs_lfr_auto_averages_ch0_V1V2')


    rfs_freqs_MHz = rfs_data.v[0]/ 1e6 * u.MHz

    rfs_timestamps = rfs_data.times
    rfs_times = Time(Time(np.array([(datetime.utcfromtimestamp(t_)+timedelta(seconds=lighttravelshift)) for t_ in rfs_timestamps])).isot)

    rfs_psdarray = rfs_data.y
    meta = {
        'observatory': "psp",
        'instrument': "FIELDS",
        'detector': datatype.upper(),
        'freqs': rfs_freqs_MHz,
        'times': rfs_times,
        'wavelength': a.Wavelength(rfs_freqs_MHz[0], rfs_freqs_MHz[-1]),
        'start_time': rfs_times[0],
        'end_time': rfs_times[-1]
    }
    rfs_spec = Spectrogram(rfs_psdarray.T, meta)

    if bg_subtraction:
        rfs_spec.data = backSub(rfs_spec.data.T).T

    return rfs_spec

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


    # PSP HFR
    rfs_spec_hfr =rfs_spec(mintime, maxtime, datatype='hfr', bg_subtraction=True)
    # PSP LFR
    rfs_spec_lfr = rfs_spec(mintime, maxtime, datatype='lfr', bg_subtraction=True)


    rfs_mm_l = np.percentile(rfs_spec_lfr.data, [40, 99.99])
    rfs_mm_h = np.percentile(rfs_spec_hfr.data, [10, 99.99])

    # start = mintime
    # endt = maxtime
    # datatype = 'rfs_hfr'
    # bg_subtraction = False
    # lighttravelshift = 0
    #
    #
    # psp_variables = pyspedas.psp.fields([start.strftime("%m/%d/%Y %H:%M:%S"), endt.strftime("%m/%d/%Y %H:%M:%S")], datatype = datatype)
    # print(f"psp {datatype.upper()} downloaded: psp_variables:")
    # print(psp_variables)
    #
    # if datatype == 'rfs_hfr':
    #     rfs_data_V1V2 = get_data('psp_fld_l2_rfs_hfr_auto_averages_ch0_V1V2')
    #     rfs_data_V3V4 = get_data('psp_fld_l2_rfs_hfr_auto_averages_ch1_V3V4')
    #     rfs_data_coher = get_data('psp_fld_l2_rfs_hfr_coher_V1V2_V3V4')
    #     rfs_data_phase = get_data('psp_fld_l2_rfs_hfr_phase_V1V2_V3V4')
    #
    #
    # elif datatype == 'rfs_lfr':
    #     rfs_data_V1V2 = get_data('psp_fld_l2_rfs_lfr_auto_averages_ch0_V1V2')
    #     rfs_data_V3V4 = get_data('psp_fld_l2_rfs_lfr_auto_averages_ch0_V3V4')
    #
    #
    # stokesI = np.power(rfs_data_V1V2.y,2)+np.power(rfs_data_V3V4.y,2)
    # stokesV = 2*np.multiply(rfs_data_V1V2.y,rfs_data_V3V4.y)*np.sin(rfs_data_coher.y)
    #
    #
    #
    # rfs_freqs_MHz = rfs_data_V1V2.v[0]/ 1e6 * u.MHz
    #
    # rfs_timestamps = rfs_data_V1V2.times
    # rfs_times = Time(Time(np.array([(datetime.utcfromtimestamp(t_)+timedelta(seconds=lighttravelshift)) for t_ in rfs_timestamps])).isot)
    #
    # rfs_psdarray = stokesV
    # meta = {
    #     'observatory': f"psp",
    #     'instrument': "FIELDS",
    #     'detector': datatype.upper(),
    #     'freqs': rfs_freqs_MHz,
    #     'times': rfs_times,
    #     'wavelength': a.Wavelength(rfs_freqs_MHz[0], rfs_freqs_MHz[-1]),
    #     'start_time': rfs_times[0],
    #     'end_time': rfs_times[-1]
    # }
    # rfs_spec = Spectrogram(rfs_psdarray.T, meta)
    #
    # if bg_subtraction:
    #     rfs_spec.data = backSub(rfs_spec.data.T).T
    #
    # # rfs_spec_lfr
    # rfs_spec_hfr= rfs_spec
    # # rfs_mm_l = np.percentile(rfs_spec_lfr.data, [40, 99.99])
    # rfs_mm_h = np.percentile(rfs_spec_hfr.data[~np.isnan(rfs_spec_hfr.data)], [50, 99.99])
    #
    #

    # ---------------------------------------------------------------- #
    # Plotting TYPE IIIs
    # ---------------------------------------------------------------- #
    # PSP
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(25, 9))
    # rfs_spec_lfr.plot(axes=axes,norm=LogNorm(vmin=rfs_mm_l[0], vmax=rfs_mm_l[1]), cmap="jet")
    # rfs_spec_hfr.plot(axes=axes,norm=LogNorm(vmin=rfs_mm_h[0], vmax=rfs_mm_h[1]), cmap="jet")
    rfs_spec_hfr.plot(axes=axes,vmin=rfs_mm_h[0], vmax=rfs_mm_h[1], cmap="jet")

    axes.set_ylabel("Frequency (MHz)")
    axes.set_title("PSP, FIELDS, HFR + LFR")

    axes.set_ylim(reversed(axes.get_ylim()))
    axes.set_yscale('log')
    axes.set_xlim(datetime(YYYY, MM, dd, HH_0,mm_0), datetime(YYYY, MM, dd, HH_1,mm_1))
    # axes.set_ylim([freqlimmax, freqlimmin])
    plt.subplots_adjust(hspace=0.31)
    plt.tight_layout()
    plt.show(block=False)
