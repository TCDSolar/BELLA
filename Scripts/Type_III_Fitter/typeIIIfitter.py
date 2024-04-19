#!/usr/bin/env python
# ------------------------------------------------------------
# Script which demonstrates how to find the best-fit
# parameters of a Gauss-Hermite line shape model
#
# Vog, 26 Mar 2012
# ------------------------------------------------------------
import numpy as np
from scipy.special import wofz
from scipy.optimize import fsolve
from kapteyn import kmpfit    # https://www.astro.rug.nl/software/kapteyn/intro.html#installinstructions
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
ln2 = np.log(2)
PI = np.pi
from math import sqrt
from datetime import datetime
from scipy.optimize import curve_fit

##########################################################################
#               GH fitting functions
##########################################################################
def gausshermiteh3h4(x, A, x0, s, h3, h4):
    # ------------------------------------------------------------
    # The Gauss-Hermite function is a superposition of functions of the form
    # F = (x-xc)/s
    # E =  A.Exp[-1/2.F^2] * {1 + h3[c1.F+c3.F^3] + h4[c5+c2.F^2+c4.F^4]}
    # ------------------------------------------------------------
    c0 = sqrt(6.0) / 4.0
    c1 = -sqrt(3.0)
    c2 = -sqrt(6.0)
    c3 = 2.0 * sqrt(3.0) / 3.0
    c4 = sqrt(6.0) / 3.0

    F = (x - x0) / s
    E = A * np.exp(-0.5 * F * F) * (1.0 + h3 * F * (c3 * F * F + c1) + h4 * (c0 + F * F * (c2 + c4 * F * F)))
    return E

def hermite2gauss(par, dpar):
    # ------------------------------------------------------------
    # Convert Gauss-Hermite parameters to Gauss(like)parameters.
    #
    # We use the first derivative of the Gauss-Hermite function
    # to find the maximum, usually around 'x0' which is the center
    # of the (pure) Gaussian part of the function.
    # If F = (x-x0)/s then the function for which we want the
    # the zero's is A0+A1*F+A2*F^2+A3*F^3+A4*F^4+A5*F^5 = 0
    # c0 = 1/4sqrt(6) c1 = -sqrt(3) c2 = -sqrt(6)
    # c3 = 2/3sqrt(3) c4 = 1/3sqrt(6)
    # ------------------------------------------------------------
    sqrt2pi = sqrt(2.0 * PI)
    amp, x0, s, h3, h4 = par
    damp, dx0, ds, dh3, dh4 = dpar  # The errors in those parameters
    c0 = sqrt(6.0) / 4.0
    c1 = -sqrt(3.0)
    c2 = -sqrt(6.0)
    c3 = 2.0 * sqrt(3.0) / 3.0
    c4 = sqrt(6.0) / 3.0

    A = np.zeros(6)
    A[0] = -c1 * h3
    A[1] = h4 * (c0 - 2.0 * c2) + 1.0
    A[2] = h3 * (c1 - 3.0 * c3)
    A[3] = h4 * (c2 - 4.0 * c4)
    A[4] = c3 * h3
    A[5] = c4 * h4

    # Define the function that represents the derivative of
    # the GH function. You need it to find the position of the maximum.
    fx = lambda x: A[0] + x * (A[1] + x * (A[2] + x * (A[3] + x * (A[4] + x * A[5]))))
    xr = fsolve(fx, 0, full_output=True)
    xm = s * xr[0] + x0
    ampmax = gausshermiteh3h4(xm, amp, x0, s, h3, h4)

    # Get line strength
    f = 1.0 + h4 * sqrt(6.0) / 4.0
    area = amp * s * f * sqrt2pi
    d_area = sqrt2pi * sqrt(s * s * f * f * damp * damp + \
                            amp * amp * f * f * ds * ds + \
                            3.0 * amp * amp * s * s * dh4 * dh4 / 8.0)

    # Get mean
    mean = x0 + sqrt(3.0) * h3 * s
    d_mean = sqrt(dx0 * dx0 + 3.0 * h3 * h3 * ds * ds + 3.0 * s * s * dh3 * dh3)

    # Get dispersion
    f = 1.0 + h4 * sqrt(6.0)
    dispersion = abs(s * f)
    d_dispersion = sqrt(f * f * ds * ds + 6.0 * s * s * dh4 * dh4)

    # Skewness
    f = 4.0 * sqrt(3.0)
    skewness = f * h3
    d_skewness = f * dh3

    # Kurtosis
    f = 8.0 * sqrt(6.0)
    kurtosis = f * h4
    d_kurtosis = f * dh4

    res = dict(xmax=xm, amplitude=ampmax, area=area, mean=mean, dispersion=dispersion, \
               skewness=skewness, kurtosis=kurtosis, d_area=d_area, d_mean=d_mean, \
               d_dispersion=d_dispersion, d_skewness=d_skewness, d_kurtosis=d_kurtosis)
    return res

def voigt(x, y):
    # The Voigt function is also the real part of
    # w(z) = exp(-z^2) erfc(iz), the complex probability function,
    # which is also known as the Faddeeva function. Scipy has
    # implemented this function under the name wofz()
    z = x + 1j * y
    I = wofz(z).real
    return I

def Voigt(nu, alphaD, alphaL, nu_0, A):
    # The Voigt line shape in terms of its physical parameters
    f = np.sqrt(ln2)
    x = (nu - nu_0) / alphaD * f
    y = alphaL / alphaD * f
    V = A * f / (alphaD * np.sqrt(np.pi)) * voigt(x, y)
    return V

def funcV(p, x):
    # Compose the Voigt line-shape
    alphaD, alphaL, nu_0, I, z0 = p
    return Voigt(x, alphaD, alphaL, nu_0, I) + z0

def funcG(p, x):
    # Model function is a gaussian
    A, mu, sigma, zerolev = p
    return (A * np.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma)) + zerolev)

def funcGH(p, x):
    # Model is a Gauss-Hermite function
    A, xo, s, h3, h4, zerolev = p
    return gausshermiteh3h4(x, A, xo, s, h3, h4) + zerolev

def residualsV(p, data):
    # Return weighted residuals of Voigt
    x, y, err = data
    return (y - funcV(p, x)) / err

def residualsG(p, data):
    # Return weighted residuals of Gauss
    x, y, err = data
    return (y - funcG(p, x)) / err

def residualsGH(p, data):
    # Return weighted residuals of Gauss-Hermite
    x, y, err = data
    return (y - funcGH(p, x)) / err

# def fit_lc(x,y,freq,sigma=1, sdauto=2, h3guess=0.1, h4guess=0, z0guess = 1,method="sigma", plotresults=False, saveplots=False, dir="/lightcurves", fname="freq"):
#     """sdauto picks n number of points from max point as initial guess for standard deviation"""
#     """ Recomended sigma 1 or 2"""
#     """ h3 - skewness ,  h4 - kurtosis"""
#     N = len(y)
#     err = np.ones(N)
#
#     if not type(sdauto) is int:
#         raise TypeError("sdauto: Only integers are allowed")
#     # Fit the Gauss-Hermite model
#     # Initial estimates for A, xo, s, h3, h4, z0
#
#     Aguess = np.max(y)
#     xguess = x[np.where(y==Aguess)[0][0]]
#     sguess = xguess - x[np.where(y==Aguess)[0][0]-sdauto]
#     # h3guess = h3guess
#     # h4guess = h4guess
#     # z0guess = z0guess
#     p0 = [Aguess, xguess, sguess, h3guess, h4guess, z0guess]
#     fitterGH = kmpfit.Fitter(residuals=residualsGH, data=(x, y, err))
#     # fitterGH.parinfo = [{}, {}, {}, {}, {}]  # Take zero level fixed in fit
#     fitterGH.fit(params0=p0)
#     print("\n========= Fit results Gaussian profile ==========")
#     print("Initial params:", fitterGH.params0)
#     print("Params:        ", fitterGH.params)
#     print("Iterations:    ", fitterGH.niter)
#     print("Function ev:   ", fitterGH.nfev)
#     print("Uncertainties: ", fitterGH.xerror)
#     print("dof:           ", fitterGH.dof)
#     print("chi^2, rchi2:  ", fitterGH.chi2_min, fitterGH.rchi2_min)
#     print("stderr:        ", fitterGH.stderr)
#     print("Status:        ", fitterGH.status)
#
#     A, x0, s, h3, h4, z0GH = fitterGH.params
#
#
#     # xm, ampmax, area, mean, dispersion, skewness, kurtosis
#     res = hermite2gauss(fitterGH.params[:-1], fitterGH.stderr[:-1])
#     print("Gauss-Hermite max=%g at x=%g" % (res['amplitude'], res['xmax']))
#     print("Area      :", res['area'], '+-', res['d_area'])
#     print("Mean (X0) :", res['mean'], '+-', res['d_mean'])
#     print("Dispersion:", res['dispersion'], '+-', res['d_dispersion'])
#     print("Skewness  :", res['skewness'], '+-', res['d_skewness'])
#     print("Kurtosis  :", res['kurtosis'], '+-', res['d_kurtosis'])
#
#     xd = np.linspace(x.min(), x.max(), 500)
#
#     xd_dt = timestamp2datetime(xd)
#     x_date = timestamp2datetime(x)
#
#
#     yfit = funcGH(fitterGH.params, xd)  # signal model fit
#     ybg = [z0GH] * len(xd)       # background
#
#     maxfit_idx = np.where(yfit==np.max(yfit))[0][0]
#
#     supported_methods = ["sigma", "hwhm", "thirdmax", "peak"]
#     if method == "sigma":
#         # Get an n sigma (standard distributions) off the peak
#         xrise = xd[maxfit_idx] - sigma*s
#     elif method == "hwhm":
#         # half width half maximum LHS
#         # need to find nearest point in model because y is output not input, thus cant import into funcGH.
#         # this is ok because we are under the resolution of the instrument
#         ycheck = yfit[0:maxfit_idx]
#         halfmax = (yfit[maxfit_idx] - z0GH)/2 + z0GH
#         y_hwhm,y_hwhm_idx = find_nearest(ycheck,halfmax)
#         xrise = xd[y_hwhm_idx]
#     elif method == "thirdmax":
#         # half width third maximum LHS
#         # need to find nearest point in model because y is output not input, thus cant import into funcGH.
#         # this is ok because we are under the resolution of the instrument
#         ycheck = yfit[0:maxfit_idx]
#         halfmax = (yfit[maxfit_idx] - z0GH)/3 + z0GH
#         y_hwhm,y_hwhm_idx = find_nearest(ycheck,halfmax)
#         xrise = xd[y_hwhm_idx]
#
#     elif method == "peak":
#         xrise = xd[maxfit_idx]
#
#     else:
#         raise AttributeError(f"method not supported. supported methods: {supported_methods}")
#
#
#
#     yrise = funcGH(fitterGH.params, xrise)
#
#     xrise = datetime.fromtimestamp(xrise)
#
#
#
#     # Plot the result
#     showparamstitle = False
#
#     plt.rcParams.update({'font.size': 15})
#     plt.rc('legend', fontsize=15)
#     fig1 = plt.figure(figsize=(9, 6), dpi=150)
#     frame1 = fig1.add_subplot(1, 1, 1)
#     frame1.plot(x_date, y, 'bo', label="data")
#     label = "G-H Model"
#     frame1.plot(xd_dt, yfit, 'c', ls='--', label=label)
#     frame1.plot(xd_dt,ybg , "y", ls="--", label='G-H Background')
#     frame1.set_xlabel(f"{x_date[0].year}/{x_date[0].month:02}/{x_date[0].day:02} [UTC]",fontsize=15)
#     frame1.set_ylabel("Relative power",fontsize=15)
#     title = ""#"Profile with Gauss-Hermite model\n"
#     t = (res['area'], res['mean'], res['dispersion'], res['skewness'], res['kurtosis'])
#     title += "GH: $\gamma_{gh}$=%.1f $x_{0_{gh}}$=%.1f $\sigma_{gh}$ = %.2f $\\xi_1$=%.2f  $\\xi_f$=%.2f\n" % t
#     title += f"freq: {freq}MHz"
#     frame1.plot(xrise, yrise, 'r*', label="detection")
#
#     frame1.grid(False)
#     leg = plt.legend(loc='upper right')
#     frame1.add_artist(leg)
#
#     if showparamstitle == True:
#         frame1.set_title(title, fontsize=15)
#     else:
#         plt.text(.01, .99, f"freq: {freq:.2f} MHz", ha='left', va='top', transform=frame1.transAxes)
#
#     frame1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#     plt.gcf().autofmt_xdate()
#
#
#
#
#     if saveplots==True:
#         mkdirectory(dir)
#         fid = f"{dir}/{fname}.jpg"
#         plt.gcf().savefig(fid, dpi=150)
#         print(f"{fid} saved")
#
#     if plotresults == True:
#         plt.show(block=False)
#     else:
#         plt.close()
#
#     return [xrise, yrise]

def fit_lc(x,y,freq,sigma=1, sdauto=2, h3guess=0.1, h4guess=0, z0guess = 1,method="sigma",profile="LE", plotresults=False, saveplots=False, dir="/lightcurves", fname="freq"):
    """sdauto picks n number of points from max point as initial guess for standard deviation"""
    """ Recomended sigma 1 or 2"""
    """ h3 - skewness ,  h4 - kurtosis"""
    N = len(y)
    err = np.ones(N)

    if not type(sdauto) is int:
        raise TypeError("sdauto: Only integers are allowed")
    # Fit the Gauss-Hermite model
    # Initial estimates for A, xo, s, h3, h4, z0

    Aguess = np.max(y)
    xguess = x[np.where(y==Aguess)[0][0]]
    sguess = xguess - x[np.where(y==Aguess)[0][0]-sdauto]
    # h3guess = h3guess
    # h4guess = h4guess
    # z0guess = z0guess
    p0 = [Aguess, xguess, sguess, h3guess, h4guess, z0guess]
    fitterGH = kmpfit.Fitter(residuals=residualsGH, data=(x, y, err))
    # fitterGH.parinfo = [{}, {}, {}, {}, {}]  # Take zero level fixed in fit
    fitterGH.fit(params0=p0)
    print("\n========= Fit results Gaussian profile ==========")
    print("Initial params:", fitterGH.params0)
    print("Params:        ", fitterGH.params)
    print("Iterations:    ", fitterGH.niter)
    print("Function ev:   ", fitterGH.nfev)
    print("Uncertainties: ", fitterGH.xerror)
    print("dof:           ", fitterGH.dof)
    print("chi^2, rchi2:  ", fitterGH.chi2_min, fitterGH.rchi2_min)
    print("stderr:        ", fitterGH.stderr)
    print("Status:        ", fitterGH.status)

    A, x0, s, h3, h4, z0GH = fitterGH.params


    # xm, ampmax, area, mean, dispersion, skewness, kurtosis
    res = hermite2gauss(fitterGH.params[:-1], fitterGH.stderr[:-1])
    print("Gauss-Hermite max=%g at x=%g" % (res['amplitude'], res['xmax']))
    print("Area      :", res['area'], '+-', res['d_area'])
    print("Mean (X0) :", res['mean'], '+-', res['d_mean'])
    print("Dispersion:", res['dispersion'], '+-', res['d_dispersion'])
    print("Skewness  :", res['skewness'], '+-', res['d_skewness'])
    print("Kurtosis  :", res['kurtosis'], '+-', res['d_kurtosis'])

    xd = np.linspace(x.min(), x.max(), 500)

    xd_dt = timestamp2datetime(xd)
    x_date = timestamp2datetime(x)


    yfit = funcGH(fitterGH.params, xd)  # signal model fit
    ybg = [z0GH] * len(xd)       # background

    maxfit_idx = np.where(yfit==np.max(yfit))[0][0]

    supported_methods = ["sigma", "hwhm", "thirdmax","quartermax", "peak"]
    if method == "sigma":
        # Get an n sigma (standard distributions) off the peak
        xrise = xd[maxfit_idx] - sigma*s

    elif method == "hwhm":
        # half width half maximum LHS (LE) and RHS(TE)
        # need to find nearest point in model because y is output not input, thus cant import into funcGH.
        if profile == "LE":
            ycheck = yfit[0:maxfit_idx]
            ydetection = (yfit[maxfit_idx] - z0GH)/2 + z0GH
            y_quarter, y_quarter_idx = find_nearest(ycheck, ydetection)
            xrise = xd[y_quarter_idx]
        elif profile == "TE":
            ycheck = yfit[maxfit_idx:]
            ydetection = (yfit[maxfit_idx] - z0GH)/2 + z0GH
            y_quarter, y_quarter_idx = find_nearest(ycheck, ydetection)
            xrise = xd[maxfit_idx+y_quarter_idx]
    elif method == "thirdmax":
        # half width half maximum LHS (LE) and RHS(TE)
        # need to find nearest point in model because y is output not input, thus cant import into funcGH.
        if profile == "LE":
            ycheck = yfit[0:maxfit_idx]
            ydetection = (yfit[maxfit_idx] - z0GH)/3 + z0GH
            y_quarter, y_quarter_idx = find_nearest(ycheck, ydetection)
            xrise = xd[y_quarter_idx]
        elif profile == "TE":
            ycheck = yfit[maxfit_idx:]
            ydetection = (yfit[maxfit_idx] - z0GH)/3 + z0GH
            y_quarter, y_quarter_idx = find_nearest(ycheck, ydetection)
            xrise = xd[maxfit_idx+y_quarter_idx]
    elif method == "quartermax":
        # half width half maximum LHS (LE) and RHS(TE)
        # need to find nearest point in model because y is output not input, thus cant import into funcGH.
        if profile == "LE":
            ycheck = yfit[0:maxfit_idx]
            ydetection = (yfit[maxfit_idx] - z0GH)/4 + z0GH
            y_quarter, y_quarter_idx = find_nearest(ycheck, ydetection)
            xrise = xd[y_quarter_idx]
        elif profile == "TE":
            ycheck = yfit[maxfit_idx:]
            ydetection = (yfit[maxfit_idx] - z0GH)/4 + z0GH
            y_quarter, y_quarter_idx = find_nearest(ycheck, ydetection)
            xrise = xd[maxfit_idx+y_quarter_idx]

    elif method == "peak":
        xrise = xd[maxfit_idx]

    else:
        raise AttributeError(f"method not supported. supported methods: {supported_methods}")



    yrise = funcGH(fitterGH.params, xrise)

    xrise = datetime.fromtimestamp(xrise)



    # Plot the result
    showparamstitle = False

    plt.rcParams.update({'font.size': 15})
    plt.rc('legend', fontsize=15)
    fig1 = plt.figure(figsize=(9, 6), dpi=150)
    frame1 = fig1.add_subplot(1, 1, 1)
    frame1.plot(x_date, y, 'bo')#, label="Data")
    label = "G-H Model"
    frame1.plot(xd_dt, yfit, 'c', ls='--', label=label)
    frame1.plot(xd_dt,ybg , "y", ls="--", label='G-H Background')
    frame1.set_xlabel(f"{x_date[0].year}/{x_date[0].month:02}/{x_date[0].day:02} [UTC]",fontsize=15)
    frame1.set_ylabel("Relative power",fontsize=15)
    title = ""#"Profile with Gauss-Hermite model\n"
    t = (res['area'], res['mean'], res['dispersion'], res['skewness'], res['kurtosis'])
    title += "GH: $\gamma_{gh}$=%.1f $x_{0_{gh}}$=%.1f $\sigma_{gh}$ = %.2f $\\xi_1$=%.2f  $\\xi_f$=%.2f\n" % t

    title += f"freq: {freq}MHz"

    frame1.plot(xrise, yrise, 'r*')#, label="Detection")
    frame1.axvline(x=xrise, color='red', linestyle='--')
    # plt.text(.11, .45, f"Detection", ha='left', va='top', transform=frame1.transAxes)

    # Add an arrow annotation
    arrow_props = {
        'arrowstyle': '->',  # Arrow style
        'color': 'red',  # Arrow color
        'linewidth': 2,  # Arrow linewidth
    }

    ymidpoint = (yfit.max() + yfit.min())/2
    frame1.annotate('Detection', xy=(xrise, ymidpoint), xytext=(datetime.fromtimestamp(xd[3]), ymidpoint), arrowprops=arrow_props)

    frame1.grid(False)
    leg = plt.legend(loc='upper right')
    frame1.add_artist(leg)

    if showparamstitle == True:
        frame1.set_title(title, fontsize=15)
    else:
        freq_lab = f"{freq:.2f}"
        if freq_lab == "0.28":
            plt.text(.01, .95, f"(a) {freq:.2f} MHz", ha='left', va='top', transform=frame1.transAxes)
        elif freq_lab == "1.42":
            plt.text(.01, .95, f"(b) {freq:.2f} MHz", ha='left', va='top', transform=frame1.transAxes)
        elif freq_lab == "1.98":
            plt.text(.01, .95, f"(c) {freq:.2f} MHz", ha='left', va='top', transform=frame1.transAxes)
        else:
            plt.text(.01, .95, f"freq: {freq:.2f} MHz", ha='left', va='top', transform=frame1.transAxes)

    frame1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()

    if saveplots==True:
        mkdirectory(dir)
        fid = f"{dir}{fname}.jpg"
        plt.gcf().savefig(fid, dpi=150)
        print(f"{fid} saved")

    if plotresults == True:
        plt.show(block=False)
    else:
        plt.close()

    return [xrise, yrise]

def auto_rise_times(spectra, freqlims=[], timelims=[],sigma=1, sdauto=2, h3guess=0.1, h4guess=0, z0guess=1, method="sigma",profile="LE", plotresults=False, saveplots=False):

    freqs = spectra.frequencies
    times = spectra.times
    data = spectra.data

    if freqlims==[]:
        freqlimmin = freqs[0]
        freqlimmax = freqs[-1]
    else:
        freqlimmin, freqlimmax  = freqlims

    minfreq, minfreqidx = find_nearest(freqs, freqlimmin)
    maxfreq, maxfreqidx = find_nearest(freqs, freqlimmax)

    if timelims==[]:
        mintime = times[0]
        maxtime = times[-1]
    else:
        mintime, maxtime = timelims

    time_idx = np.where(np.logical_and(times >= mintime, times <= maxtime))

    times_dt = []
    for each in times[time_idx]:
        times_dt.append(each.datetime)


    dir = f"lightcurves/lc_{times_dt[0].year}_{times_dt[0].month:02}_{times_dt[0].day:02}/{spectra.observatory}/{method}_{profile}/"

    risetimes = []
    riseval = []
    testfreq = []
    x = np.array(datetime2timestamp(times_dt))
    for f in range(minfreqidx,maxfreqidx): #f = 1
        testdata = data[f, time_idx][0]
        normfact = np.abs(testdata[0])
        if normfact != 0:
            y = np.array(testdata / normfact) # normalised
            fname = f"{spectra.observatory.replace(' ','')}_{method}_{freqs[f].value:.2f}"
            risex, risey = fit_lc(x, y,freq=freqs[f].value, sigma=sigma,profile=profile, sdauto=sdauto, h3guess=h3guess, h4guess=h4guess, z0guess=z0guess,method=method, plotresults=plotresults, saveplots=saveplots, dir=dir, fname=fname )
            risetimes.append(risex)
            riseval.append(risey)
            testfreq.append(freqs[f].value)

    return risetimes, riseval, testfreq


def rise_times_atmax(spectra, freqlims=[], timelims=[]):

    freqs = spectra.frequencies
    times = spectra.times
    data = spectra.data

    if freqlims==[]:
        freqlimmin = freqs[0]
        freqlimmax = freqs[-1]
    else:
        freqlimmin, freqlimmax  = freqlims

    minfreq, minfreqidx = find_nearest(freqs, freqlimmin)
    maxfreq, maxfreqidx = find_nearest(freqs, freqlimmax)

    if timelims==[]:
        mintime = times[0]
        maxtime = times[-1]
    else:
        mintime, maxtime = timelims

    time_idx = np.where(np.logical_and(times >= mintime, times <= maxtime))

    times_dt = []
    for each in times[time_idx]:
        times_dt.append(each.datetime)

    risetimes = []
    riseval = []
    testfreq = []
    x = np.array(datetime2timestamp(times_dt))
    for f in range(minfreqidx,maxfreqidx): #f = 1
        testdata = data[f, time_idx][0]
        y_max = testdata.max()
        t = np.where(testdata==y_max)
        risex = timestamp2datetime(x[t])[0]
        risetimes.append(risex)
        riseval.append(y_max)
        testfreq.append(freqs[f].value)

    return risetimes, riseval, testfreq


def testmethod(spectra, freqlims=[], timelims=[],sigma=1, sdauto=2, h3guess=0.1, h4guess=0, z0guess=1, plotresults=False, saveplots=False):

    freqs = spectra.frequencies
    times = spectra.times
    data = spectra.data

    if freqlims==[]:
        freqlimmin = freqs[0]
        freqlimmax = freqs[-1]
    else:
        freqlimmin, freqlimmax  = freqlims

    minfreq, minfreqidx = find_nearest(freqs, freqlimmin)
    maxfreq, maxfreqidx = find_nearest(freqs, freqlimmax)

    if timelims==[]:
        mintime = times[0]
        maxtime = times[-1]
    else:
        mintime, maxtime = timelims

    time_idx = np.where(np.logical_and(times >= mintime, times <= maxtime))

    times_dt = []
    for each in times[time_idx]:
        times_dt.append(each.datetime)



    dir=f"lightcurves/{spectra.observatory}"



    risetimes = []
    riseval = []
    testfreq = []
    x = np.array(datetime2timestamp(times_dt))
    for f in range(minfreqidx,maxfreqidx): #f = 1
        testdata = data[f, time_idx][0]
        normfact = np.abs(testdata[0])
        if normfact != 0:
            y = np.array(testdata / normfact) # normalised
            risex, risey = fit_lc(x, y,freq=freqs[f].value, sigma=sigma, sdauto=sdauto, h3guess=h3guess, h4guess=h4guess, z0guess=z0guess,plotresults=plotresults, saveplots=saveplots, dir=dir, fname=f"{f}" )
            risetimes.append(risex)
            riseval.append(risey)
            testfreq.append(freqs[f].value)

    return risetimes, riseval, testfreq




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

def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def exponential_func2(x, a, b, c,d):
    return a * np.exp((-b * x) + d) + c

def log_func2(x,a,b,c,d):
    return (1/b) * (d - np.log((x-c)/a))

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
    freqs4tri -  the frquencies that will be used for multilateration"""
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

    if plot_residuals == True:
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







if __name__=="__main__":


    spectra = solo_spec  # spectra from radiospectra
    freqs = spectra.frequencies
    times = spectra.times
    data = spectra.data

    minfreq, minfreqidx = find_nearest(freqs, freqlimmin)
    maxfreq, maxfreqidx = find_nearest(freqs, freqlimmax)

    mintime = datetime(YYYY, MM, dd, HH_0,mm_0)
    maxtime =  datetime(YYYY, MM, dd, 3,30)
    time_idx = np.where(np.logical_and(times >= mintime, times <= maxtime))

    times_dt = []
    for each in times[time_idx]:
        times_dt.append(each.datetime)


    risetimes = []
    riseval = []
    testfreq = []
    x = np.array(datetime2timestamp(times_dt))
    for f in range(minfreqidx,maxfreqidx): #f = 1
        testdata = data[f, time_idx][0]
        normfact = np.abs(testdata[0])
        if normfact != 0:
            y = np.array(testdata / normfact) # normalised
            risex, risey = fit_lc(x, y, sdauto=1, h3guess=0.1, h4guess=0, z0guess=0, plotresults=False,saveplots=True,dir=f"lightcurves/{spectra.observatory}", fname=f"{f}" )
            risetimes.append(risex)
            riseval.append(risey)
            testfreq.append(freqs[f].value)



    mm = np.percentile(spectra.data, [10, 99])

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
    spectra.plot(axes=axes, vmin=mm[0], vmax=mm[1])
    axes.plot(risetimes, testfreq, 'r*')
    axes.set_ylim(reversed(axes.get_ylim()))
    axes.set_yscale('log')
    axes.set_ylim([freqlimmax, freqlimmin])
    axes.set_xlim(datetime(YYYY, MM, dd, HH_0,mm_0), datetime(YYYY, MM, dd, 3,0))
    plt.subplots_adjust(hspace=0.31)
    plt.show(block=False)







