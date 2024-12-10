import os
import pickle
import argparse
from datetime import datetime

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from radiospectra.spectrogram import Spectrogram
from sunpy_soar.attrs import Identifier

import astropy.units as u
from astropy.time import Time
from astropy.visualization import ImageNormalize, PercentileInterval

from sunpy.net import Fido
from sunpy.net import attrs as a


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
def mkdirectory(directory):
    dir = directory
    isExist = os.path.exists(dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir)
        print("The new directory is created!")
    return dir

if __name__=="__main__":
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
    background_subtraction = False
    savedata  = True
    mintime = datetime(YYYY, MM, dd, HH_0, mm_0)
    maxtime = datetime(YYYY, MM, dd, HH_1, mm_1)
    timelims = [mintime, maxtime]
    # start = mintime
    # endt = maxtime


    query = Fido.search(a.Time(f"{YYYY}-{MM:02}-{dd:02}T{HH_0:02}:{mm_0:02}", f"{YYYY}-{MM:02}-{dd:02}T{HH_1:02}:{mm_1:02}"),
                        a.Instrument('RPW'), a.Level(2), Identifier('RPW-HFR-SURV'))
    downloaded = Fido.fetch(query[0])
    spec = Spectrogram(downloaded[0])

    if spec[0] != []:
        spectra = spec[0]
        rpw_freqs_MHz = spectra.frequencies.value / 1e3 * u.MHz
        rpw_times = Time(Time(np.array([t_.datetime for t_ in spectra.times])).isot)

        rpw_psdarray = spectra.data

        meta = {
            'observatory': "SolO",
            'instrument': "RPW",
            'detector': "RPW-HFR-SURV",
            'freqs': rpw_freqs_MHz,
            'times': rpw_times,
            'wavelength': a.Wavelength(rpw_freqs_MHz[0], rpw_freqs_MHz[-1]),
            'start_time': rpw_times[0],
            'end_time': rpw_times[-1]
        }
        rpw_AGC1 = Spectrogram(rpw_psdarray, meta)

        if background_subtraction:
            rpw_AGC1.data = backSub(rpw_AGC1.data)

    if spec[1] != []:
        spectra = spec[1]
        rpw_freqs_MHz = spectra.frequencies.value / 1e3 * u.MHz
        rpw_times = Time(Time(np.array([t_.datetime for t_ in spectra.times])).isot)

        rpw_psdarray = spectra.data

        meta = {
            'observatory': "SolO",
            'instrument': "RPW",
            'detector': "RPW-HFR-SURV",
            'freqs': rpw_freqs_MHz,
            'times': rpw_times,
            'wavelength': a.Wavelength(rpw_freqs_MHz[0], rpw_freqs_MHz[-1]),
            'start_time': rpw_times[0],
            'end_time': rpw_times[-1]
        }
        rpw_AGC2 = Spectrogram(rpw_psdarray, meta)

        if background_subtraction:
            rpw_AGC2.data = backSub(rpw_AGC2.data)


    # rpw_AGC1_mm = np.percentile(rpw_AGC1.data, [10,99.999])
    # rpw_AGC2_mm = np.percentile(rpw_AGC2.data, [10,99.999])
    rpw_AGC1_mm = ImageNormalize(rpw_AGC1.data, interval=PercentileInterval(90.5))
    rpw_AGC2_mm = ImageNormalize(rpw_AGC2.data, interval=PercentileInterval(90.5))

    # ---------------------------------------------------------------- #
    # JOINT PLOT
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #
    # JOINT PLOT Horizontal
    # ---------------------------------------------------------------- #

    my_cmap = mpl.cm.jet
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 9))

    rpw_AGC1.plot(axes=axes[0], norm=rpw_AGC1_mm, cmap=my_cmap)
    rpw_AGC2.plot(axes=axes[1], norm=rpw_AGC2_mm, cmap=my_cmap)

    axes[0].set_title("SolO, RPW, AGC1")
    axes[1].set_title("SolO, RPW, AGC2")

    axes[0].set_ylabel("Frequency (MHz)")
    axes[1].set_ylabel("Frequency (MHz)")

    # # # by default y-axis low to height flip so moving away from sun with time
    axes[0].set_ylim(reversed(axes[0].get_ylim()))
    axes[1].set_ylim(reversed(axes[1].get_ylim()))

    #
    # log y-axis
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')

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

    data = [rpw_AGC1, rpw_AGC2]
    if savedata is True:
        directory = mkdirectory("solo_data/RPW/")
        with open(f'{directory}rpw_extracted_radiospectra_{YYYY}{MM:02}{dd:02}.pkl', 'wb') as output_file:
            pickle.dump(data, output_file)
