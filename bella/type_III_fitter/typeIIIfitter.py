#!/usr/bin/env python

import os

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from scipy.optimize import curve_fit



ln2 = np.log(2)
PI = np.pi
##########################################################################
#               Type III fitting functions
##########################################################################

def quadratic_func(x,a,b,c):
    return a*(x**2) + b*x + c

def biquadratic_func(x, a,b,c,d,e):
    return  a*(x**4) + b*(x**3) + c*(x**2) + d*x + e

def typeIII_func(times, popt, pcov, xref, num=100):

    xdata_buff = np.array(times) - xref
    xdatafit = np.linspace(xdata_buff[0], xdata_buff[-1], num=num)

    ydatafit = exp_fn2(xdatafit, *popt)

    xdatafit_corrected = xdatafit + xref

    xdatafit_corrected_dt = timestamp2datetime(xdatafit_corrected)

    return xdatafit_corrected_dt, ydatafit

def log_func(x,a,b,c):
    return (-1/b) * np.log((x-c)/a)

def log_func2(x,a,b,c,d):
    return (1/b) * (d - np.log((x-c)/a))

def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def exponential_func2(x, a, b, c,d):
    return a * np.exp((-b * x) + d) + c

def exponential_func3(x, a, b, c,d,const ):
    return a * const**((-b * x) + d) + c

def exp_fn2(x,a,b,c,d):
    return a*np.exp((b-x)/c) + d

def log_fn2(x,a,b,c,d):
    return b-(c*np.log((x-d)/a))

def reciprocal_3rdorder(x, a0, a1, a2, a3):
    return a0 + a1 / x + a2 / x ** 2+ a3 / x ** 3

def reciprocal_2ndorder(x, a0, a1, a2):
    return a0 + a1 / x + a2 / x ** 2

def fittypeIII(times, ydata):
    "Not in use"
    xdata = datetime2timestamp(times)
    xref = xdata[-1]
    xdata_buff = np.array(xdata) - xref

    popt, pcov = curve_fit(exp_fn2, xdata_buff, ydata)

    return popt, pcov, xref

def typeIIIfitting(risetimes,testfreq, fitfreqs,freqs4tri, plot_residuals=False):
    """This function takes discrete frequencies and timestamps and returns fitted time data for input.
    fitfreqs  -  frequencies for extrapolated and interpolated fitting
    freqs4tri -  the frequencies that will be used for multilateration"""
    # Turning extracted datetimes into timestamps.
    # Timestamps are subtracted from reference point xref for management of data.
    xdata, xref = epoch2time(risetimes)
    # Frequencies extracted from data in MHz
    ydata = testfreq

    # Fitting discrete DATA into smooth Type III
    popt, pcov = curve_fit(reciprocal_2ndorder, ydata, xdata)

    # Time output used locally, this includes extrapolation of burst.
    fittimes = reciprocal_2ndorder(fitfreqs, *popt)

    # Time output used for multilateration
    times4tri = reciprocal_2ndorder(freqs4tri, *popt)

    # Extrapolated data might result in nan values
    notnan = np.where(~np.isnan(fittimes))
    fitfreqs_local = fitfreqs[notnan]
    fittimes_notnan = fittimes[notnan]

    # Convert into datetime
    times4tri_dt = time2epoch(times4tri, xref)
    fittimes_corrected = time2epoch(fittimes_notnan, xref)

    # residuals
    fittimes_for_residuals = reciprocal_2ndorder(np.array(testfreq), *popt)
    residuals = np.subtract(xdata, fittimes_for_residuals)

    if plot_residuals is True:
        plt.figure()
        plt.plot(residuals, "r.")
        plt.title("residuals WAVES LEADING EDGE")
        plt.xlabel("index")
        plt.ylabel("difference")
        plt.show(block=False)

    return times4tri_dt, fitfreqs_local, fittimes_corrected



##########################################################################
#               Additional functions
##########################################################################

def datetime2timestamp(dates):
   timestamps=[]
   for each in dates:
      timestamps.append(datetime.timestamp(each))
   return timestamps

def timestamp2datetime(stamps):
   dates=[]
   for each in stamps:
      dates.append(datetime.fromtimestamp(each))
   return dates

def epoch2time(epoch):
    times_buff = datetime2timestamp(epoch)
    tref = min(times_buff)
    times = np.array(times_buff) - tref
    return times, tref

def time2epoch(times, xref):
    epoch_corrected = np.array(times) + xref
    epoch_dt = timestamp2datetime(epoch_corrected)
    return epoch_dt

def mkdirectory(directory):
    dir = directory
    isExist = os.path.exists(dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir)
        print("The new directory is created!")

    return dir

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def f_to_angs(f_mhz,c=299792458):
    angstrom = (c / (f_mhz * 10 ** 6)) * 10 ** 10
    return angstrom
