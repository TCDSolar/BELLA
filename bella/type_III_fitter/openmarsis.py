from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from radiospectra.spectrogram import Spectrogram  # in the process of updating old spectrogram
from scipy.interpolate import RegularGridInterpolator

import astropy.units as u
from astropy.time import Time

from sunpy.net import attrs as a


def resample_spectra(spectra, new_cadence=None, new_freq_resolution=None):
    """
    Resample a spectra object to a specified cadence and/or frequency resolution.

    Parameters:
    spectra (Spectra object): The spectra object containing data, times, and frequencies.
    new_cadence (float, optional): The desired cadence in seconds.
    new_freq_resolution (float, optional): The desired frequency resolution.

    Returns:
    tuple: (new_data, new_time, new_frequency) - The resampled dynamic spectra data,
           the corresponding new time array, and the new frequency array.
    """
    # Convert Time objects to timestamps for interpolation
    time_stamps = spectra.times.unix

    # Ensure data dimensions match time and frequency dimensions
    if spectra.data.shape != (len(time_stamps), len(spectra.frequencies)):
        # Assuming the data needs to be transposed
        spectra.data = spectra.data.T

    # Handle new cadence
    if new_cadence is not None:
        new_time_stamps = np.arange(time_stamps[0], time_stamps[-1], new_cadence)
    else:
        new_time_stamps = time_stamps

    # Convert new timestamps back to Time objects
    new_time = Time(new_time_stamps, format='unix')

    # Handle new frequency resolution
    if new_freq_resolution is not None:
        new_frequency = np.arange(spectra.frequencies[0], spectra.frequencies[-1], new_freq_resolution)
    else:
        new_frequency = spectra.frequencies

    # Create RegularGridInterpolator
    interpolator = RegularGridInterpolator((time_stamps, spectra.frequencies), spectra.data)

    # Prepare new grid for interpolation
    new_grid = np.meshgrid(new_time_stamps, new_frequency, indexing='ij')
    new_grid_flat = np.array([grid.ravel() for grid in new_grid]).T

    # Interpolate
    new_data = interpolator(new_grid_flat).reshape(len(new_time_stamps), len(new_frequency))

    return new_data, new_time, new_frequency


def marsis_spectra(file_path, quickplot=False, histogram=[], lighttravelshift=0):
    """
    Returns a Radiospectra spectrogram from MARSIS (Mars Express) Radar spectrogram data from a given .dat file (ASCII).

    Args:
        file_path (str): Path to the .dat ASCII file containing MARSIS spectrogram data.
        quickplot (bool, optional): If True, display a quickplot of the spectrogram. Default is False.
        histogram (list, optional): A list of two percentile values (e.g., [5, 95]) to set the color scale
            for the quickplot. Default is an empty list, which will use the default color scale.

    Returns:
        mars_spec (Spectrogram): A radiospectra spectrogram object containing processed MARSIS data.

    Note:
        - The function reads the data from the specified file, processes it, and creates a Spectrogram object.
        - You can adjust the `quickplot` and `histogram` parameters to control the plot's appearance.
    """


    # Open the .dat file in binary mode
    with open(file_path, 'rb') as dat_file:
        # Read the contents of the file
        data = dat_file.read()

    # Now 'data' contains the contents of the .dat file
    # You can process or analyze the data as needed

    # If it's a text file, you can decode it to a string
    # Example for decoding as UTF-8:
    decoded_data = data.decode('ASCII')
    data_ = decoded_data.split('\n')
    data_np = np.array(data_)
    data_np_ = data_np[:-1]  # The last item is blank
    data = data_np_.reshape(int(len(data_np_)/7),7)

    FrameBeginTime = []
    Frequency = []
    BandNumber = []
    ReceiverAttenuation = []
    TransmitPowerLevel = []
    Spectra = []

    date_format = "%Y-%jT%H:%M:%S.%f"


    # Transforming data from string format to their corresponding data types
    for each in data:
        #epoch points stored as datetime
        FrameBeginTime.append(datetime.strptime(each[0][42:-1], date_format))

        # frequency is stored as a float, data is in kHz
        Frequency.append(float(each[1][21:-5]))

        #band number is stored as integer, unused
        BandNumber.append(int(each[2][14:-1]))

        # Receiver attenuation is stored as integer, unused
        ReceiverAttenuation.append(int(each[3][23:-1]))

        # Transmit power level is stored as integer, unused
        TransmitPowerLevel.append(int(each[4][23:-1]))

        # There's "higher time resolution spectra in each epoch point. Averaged this for simplicity.
        Spectra.append(np.array(each[5][:-2].split(' '), dtype=np.float64).mean())

    # Turning data lists into numpy arrays
    FrameBeginTime = np.array(FrameBeginTime)
    Frequency = np.array(Frequency)
    BandNumber = np.array(BandNumber)
    ReceiverAttenuation = np.array(ReceiverAttenuation)
    TransmitPowerLevel = np.array(TransmitPowerLevel)
    Spectra = np.array(Spectra)


    freq_uniq = np.unique(Frequency)      # Frequency array in kHz
    time_uniq = np.unique(FrameBeginTime)  # Epoch array

    # ADDING GAPS in data.
    # 1 - Calculates data cadence.
    # 2 - Searches for any gap in epoch larger than 2 cadences.
    # 3 - Anywhere where there's a gap is filled with 0 values and new epoch points are added.
    # 4 - repeat #3 until all gaps are filled.
    t_diff = []
    for i in range(1, len(time_uniq)):
        t_diff.append(time_uniq[i] - time_uniq[i-1])
    t_diff = np.array(t_diff)
    cadence = np.median(t_diff)

    idx = np.where(t_diff>cadence*2)

    spec = Spectra.reshape(len(time_uniq), len(freq_uniq))

    for i in idx:
        end_date = time_uniq[i+1]
        date_gaps = []
        current_date = time_uniq[i]+cadence
        if current_date.size > 0:
            while current_date <= end_date:
                date_gaps.append(current_date)
                current_date = current_date + cadence
            date_gaps = np.array(date_gaps).reshape(len(date_gaps))
            time_uniq = np.insert(time_uniq, i+1, date_gaps)
            zeros_ = np.zeros((len(date_gaps),len(freq_uniq)))
            spec = np.insert(spec, i, zeros_, axis=0)
    # END of ADDING GAPS in data



    # RADIOSPECTRA
    mars_times = Time(time_uniq) + timedelta(seconds=lighttravelshift)#  Radiospectra likes astropy Time

    mars_freqs = np.array(freq_uniq)*1E-3*u.MHz   # consistent with Solar observations
    mars_spec_bg = spec - spec.min(axis=0)        # Background subtraction to filter background noise
    # mars_spec_bg = spec                         # Bypass back subtraction for testing

    # Radiospectra expects at least the following meta data
    meta = {
        'observatory': 'MarsExpress',
        'instrument': 'MARSIS',
        'detector': 'RDR',
        'freqs': mars_freqs,
        'times': mars_times,
        'wavelength': a.Wavelength(mars_freqs[0], mars_freqs[-1]),
        'start_time': mars_times[0],
        'end_time': mars_times[-1]
    }
    mars_spec = Spectrogram(mars_spec_bg.T, meta)    # Generates radiospectra spectrogram

    if quickplot is True:
        mars_mm_h = np.percentile(mars_spec.data, [histogram[0], histogram[1]])  # histogram levels

        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
        mars_spec.plot(axes=axes, vmin=mars_mm_h[0], vmax=mars_mm_h[1])
        axes.set_ylim(reversed(axes.get_ylim()))   # Solar standards
        axes.set_yscale('log')                     # Solar standards
        plt.subplots_adjust(hspace=0.31)
        plt.show(block=False)

    return mars_spec


if __name__=='__main__':
    # # Specify the path to your .dat file
    # file_path = "./22342/FRM_AIS_RDR_22342_ASCII_.DAT" # 2021/09/06
    # marsis_spectra(file_path, quickplot=True, histogram=[10, 92])
    #
    #
    # file_path = "./22648/FRM_AIS_RDR_22648_ASCII_.DAT" # 2021/12/04
    # marsis_spectra(file_path, quickplot=True, histogram=[10, 94])
    #
    #
    # file_path = "./21342/DATA/ACTIVE_IONOSPHERIC_SOUNDER/FRM_AIS_RDR_21342_ASCII.DAT" # 2020/11/19
    # marsis_spectra(file_path, quickplot=True, histogram=[10, 92])
    #

    #file_path = "./21380/DATA/ACTIVE_IONOSPHERIC_SOUNDER/FRM_AIS_RDR_21380_ASCII.DAT"  # 2020/11/30
    # file_path = # file_path = # file_path = # file_path = #
    file_path = "/Users/canizares/OneDrive/Work/0_PhD/Projects/dynSpectra/MARSIS/22648/FRM_AIS_RDR_22648_ASCII_.DAT"
    # file_path ="/Users/canizares/OneDrive/Work/0_PhD/Projects/dynSpectra/MARSIS/22342/FRM_AIS_RDR_22342_ASCII_.DAT"
    # file_path = "/Users/canizares/OneDrive/Work/0_PhD/Projects/dynSpectra/MARSIS/21380/DATA/ACTIVE_IONOSPHERIC_SOUNDER/FRM_AIS_RDR_21380_ASCII.DAT"
    # file_path = "/Users/canizares/OneDrive/Work/0_PhD/Projects/dynSpectra/MARSIS/21342/DATA/ACTIVE_IONOSPHERIC_SOUNDER/FRM_AIS_RDR_21342_ASCII.DAT"
    # file_path = "/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/2011_06_24/marsis_data/FRM_AIS_RDR_9544_ASCII.DAT"
    # file_path = "/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/BELLA_Projects/2021_12_04/mex_data/marsis/22648/FRM_AIS_RDR_22648_ASCII_.DAT"
    mars_spec = marsis_spectra(file_path, quickplot=True, histogram=[10, 92])

    mars_mm_h = np.percentile(mars_spec.data, [10,97])
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
    mars_spec.plot(axes=axes, vmin=mars_mm_h[0], vmax=mars_mm_h[1], cmap = "jet")
    axes.set_ylim(reversed(axes.get_ylim()))  # Solar standards
    axes.set_yscale('log')  # Solar standards
    plt.subplots_adjust(hspace=0.31)
    plt.show(block=False)

    new_data, new_time, new_frequency = resample_spectra(mars_spec, new_cadence=20, new_freq_resolution=None)


    meta = {
        'observatory': 'MarsExpress',
        'instrument': 'MARSIS',
        'detector': 'RDR',
        'freqs': new_frequency,
        'times': new_time,
        'wavelength': a.Wavelength(new_frequency[0], new_frequency[-1]),
        'start_time': new_time[0],
        'end_time': new_time[-1]
    }
    mars_spec = Spectrogram(new_data.T, meta)    # Generates radiospectra spectrogram

    mars_mm_h = np.percentile(mars_spec.data, [10,97])
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
    mars_spec.plot(axes=axes, vmin=mars_mm_h[0], vmax=mars_mm_h[1], cmap = "jet")
    axes.set_ylim(reversed(axes.get_ylim()))  # Solar standards
    axes.set_yscale('log')  # Solar standards
    plt.subplots_adjust(hspace=0.31)
    plt.show(block=False)
