import datetime as dt

import matplotlib as mpl
from matplotlib import pyplot as plt
from radiospectra.spectrogram import Spectrogram

import astropy.units as u
from astropy.time import Time
from astropy.visualization import ImageNormalize, PercentileInterval

from sunpy.net import attrs as a

from bella.type_III_fitter import dynspec

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.family': "Times New Roman"})

# def open_rpw_l3(cdf_file_path, bg_subtraction=False, lighttravelshift=0):
#     #CDFLIB reads Epoch inccorrectly. DO NOT USE CDFLIB with solo data
#     # Open the CDF file
#     cdf = cdflib.CDF(cdf_file_path)
#
#     # List all variables in the CDF file
#     var_names = cdf.cdf_info()['zVariables']
#
#     print("Variables in the CDF file:", var_names)
#
#     epoch = cdf.varget("Epoch")
#     frequency = cdf.varget("FREQUENCY")
#     background = cdf.varget("BACKGROUND")
#     sensor_config = cdf.varget("SENSOR_CONFIG")
#     channel = cdf.varget("CHANNEL")
#     timing = cdf.varget("TIMING")
#     quality_flag = cdf.varget("QUALITY_FLAG")
#     interpol_flags = cdf.varget("INTERPOL_FLAG")
#     psd_v2 = cdf.varget("PSD_V2")
#     psd_flux = cdf.varget("PSD_FLUX")
#     psd_sfu = cdf.varget("PSD_SFU")
#     lbl1_sc_pos_hci = cdf.varget("LBL1_SC_POS_HCI")
#     sc_pos_hci = cdf.varget("SC_POS_HCI")
#     rep1_sc_pos_hci = cdf.varget("REP1_SC_POS_HCI")
#
#     j2000_start = dt.datetime(2000, 1, 1, 12, 0) + dt.timedelta(seconds=lighttravelshift)
#     epoch_dt = [j2000_start + dt.timedelta(microseconds=tt / 1000) for tt in epoch]
#
#     rpw_freqs_MHz = frequency / 1e6 * u.MHz
#     rpw_times = Time([dt.isoformat() for dt in epoch_dt])
#
#
#     meta = {
#         'observatory': f"SolO",
#         'instrument': "RPW",
#         'detector': "RPW-HFR-SURV",
#         'freqs': rpw_freqs_MHz,
#         'times': rpw_times,
#         'wavelength': a.Wavelength(rpw_freqs_MHz[0], rpw_freqs_MHz[-1]),
#         'start_time': rpw_times[0],
#         'end_time': rpw_times[-1]
#     }
#     rpw_spectro_sfu = Spectrogram(psd_sfu.T, meta)
#     if bg_subtraction:
#         rpw_spectro_sfu.data = dynspec.backSub(rpw_spectro_sfu.data.T).T
#
#     return rpw_spectro_sfu

def open_rpw_l3(cdf_file_path, bg_subtraction=False, lighttravelshift=0):
    from spacepy import pycdf
    print("Opening RPW data with pycdf")
    cdf = pycdf.CDF(cdf_file_path)
    epoch = cdf["Epoch"]
    frequency = cdf["FREQUENCY"]
    cdf["BACKGROUND"]
    cdf["SENSOR_CONFIG"]
    cdf["CHANNEL"]
    cdf["TIMING"]
    cdf["QUALITY_FLAG"]
    cdf["INTERPOL_FLAG"]
    cdf["PSD_V2"]
    cdf["PSD_FLUX"]
    psd_sfu = cdf["PSD_SFU"]
    cdf["LBL1_SC_POS_HCI"]
    cdf["SC_POS_HCI"]
    cdf["REP1_SC_POS_HCI"]


    epoch_dt = epoch[:]

    rpw_freqs_MHz = frequency[:] / 1e6 * u.MHz
    rpw_times = Time([(t_+dt.timedelta(seconds=lighttravelshift)).isoformat() for t_ in epoch_dt])


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
    rpw_spectro_sfu = Spectrogram(psd_sfu[:].T, meta)
    if bg_subtraction:
        rpw_spectro_sfu.data = dynspec.backSub(rpw_spectro_sfu.data.T).T
    return rpw_spectro_sfu


if __name__=="__main__":
    cdf_file_path = '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/dynSpectra/solar_orbiter_data/rpw/science/l3/solo_L3_rpw-hfr-flux_20211204_V01.cdf'
    rpw_spectro_hfr = open_rpw_l3(cdf_file_path)
    rpw_mm_hfr = ImageNormalize(rpw_spectro_hfr.data, interval=PercentileInterval(97.5))

    cdf_file_path = '/Users/canizares/Library/CloudStorage/OneDrive-Personal/Work/0_PhD/Projects/dynSpectra/solar_orbiter_data/rpw/science/l3/solo_L3_rpw-tnr-flux_20211204_V01.cdf'
    rpw_spectro_tnr = open_rpw_l3(cdf_file_path)
    rpw_mm_tnr = ImageNormalize(rpw_spectro_tnr.data, interval=PercentileInterval(97.5))




    YYYY = rpw_spectro_hfr.times[round(len(rpw_spectro_hfr.times)/2)].datetime.year
    MM = rpw_spectro_hfr.times[round(len(rpw_spectro_hfr.times)/2)].datetime.month
    dd = rpw_spectro_hfr.times[round(len(rpw_spectro_hfr.times)/2)].datetime.day
    HH_0 = 0
    mm_0 = 0
    HH_1 = 23
    mm_1 = 59

    my_cmap = mpl.cm.jet


    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(20, 9))
    rpw_spectro_hfr.plot(axes=axes, norm=rpw_mm_hfr, cmap=my_cmap)
    # rpw_spectro_tnr.plot(axes=axes, norm=rpw_mm_tnr, cmap=my_cmap)

    axes.set_title("SolO, RPW, TNR+HFR")

    axes.set_ylabel("Frequency (MHz)")

    # # # by default y-axis low to height flip so moving away from sun with time
    axes.set_ylim(reversed(axes.get_ylim()))

    #
    # log y-axis
    axes.set_yscale('log')

    axes.set_xlim(dt.datetime(YYYY, MM, dd, HH_0, mm_0), dt.datetime(YYYY, MM, dd, HH_1, mm_1))
    plt.subplots_adjust(left=0.041, bottom=0.096, right=0.984, top=0.93, wspace=0.132, hspace=0.31)

    # plt.tick_params(axis='y', which='minor')
    # axes.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    # plt.tick_params(axis='y', which='major')
    # axes.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # figfname = f"Figures/{YYYY}_{MM:02}_{dd:02}/{YYYY}_{MM:02}_{dd:02}_Horizontal.png"
    # plt.savefig(figfname, dpi='figure')

    plt.tight_layout()
    plt.show(block=False)
