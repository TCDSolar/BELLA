#!/usr/bin/env python3
"""
Created on Mon Jun 17 15:27:04 2019
@authors: Luis Alberto Canizares (adapted from benmosley)


"""

import os
import datetime as dt
import multiprocessing

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from astropy.constants import R_sun, au, c


class BayesianTOAPositioner:
    """Class for carrying out Bayesian TOA positioning.
    Requires at least 3 stations (to solve for x,y,v,t1). 4 recommended.
    """

    def __init__(self,
                 stations,
                 x_lim=1.5*au.value,# maximum box size (m)
                 v_mu=c.value,# max velocity prior (m/s)
                 v_sd=(0.00001)*c.value,# standard deviation of velocity prior (m/s)
                 t_cadence=60): # Spacecraft Cadence in seconds


        t_sd = t_cadence / 2  # standard deviation of observed values (s)

        t_lim = 24*60*60# np.sqrt(2)*x_lim/v_mu# resulting max toa value, used for t1 limit (s)     24*60*60     #

        # check if well posed
        if len(stations)<4:
            print("WARNING: at least 4 stations recommended for bayesian toa positioning!")


        self.x_lim = x_lim
        self.v_mu = v_mu
        self.v_sd = v_sd
        self.t_cadence = t_cadence
        self.t_sd = t_sd
        self.t_lim = t_lim
        self.stations = stations

    def sample(self, toa, draws=2000, tune=2000, chains=4,cores=4, init='jitter+adapt_diag', progressbar=True, verbose=True):
        "Carry out Bayesian inference"

        x_lim = self.x_lim
        v_mu = self.v_mu
        v_sd = self.v_sd
        t_sd = self.t_sd
        t_lim = self.t_lim
        stations = self.stations

        # assert correct number of observations
        if len(toa) != len(stations):
            raise Exception(f"ERROR: number of observations must match number of stations! ({len(toa)}, {len(stations)})")

        # assert max toa is not larger than t_lim
        if np.max(toa) > t_lim:
            raise Exception("ERROR: toa > t_lim")

        with pm.Model():  # CONTEXT MANAGER

            # Priors
            # v = pm.TruncatedNormal("v", mu=v_mu, sigma=v_sd, upper=v_mu+v_sd)
            v = pm.Normal("v", mu=v_mu, sigma=v_sd)
            # x = pm.Uniform("x", lower=-x_lim, upper=x_lim, shape=2)          # prior on the source location (m)
            x = pm.Normal("x", mu=0, sigma=x_lim/4, shape=2)                   # prior on the source location (m)
            t0 = pm.Uniform("t0", lower=-t_lim, upper=t_lim)                   #

            # Physics model
            d = pm.math.sqrt(pm.math.sum((stations - x)**2, axis=1))         # distance between source and receivers
            t1 = d/v                                                         # time of arrival of each receiver

            t = t1-t0                                                        # TOA dt

            # Observations
            print(f"\nt: {t} \n t_sd: {t_sd} \n toa: {toa}")
            Y_obs = pm.Normal('Y_obs', mu=t, sd=t_sd, observed=toa)          # DATA LIKELIHOOD function


            # Posterior sampling
            #step = pm.HamiltonianMC()
            trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, target_accept=0.95, init=init, progressbar=progressbar,return_inferencedata=False)#, step=step)# i.e. tune for 1000 samples, then draw 5000 samples

            summary = az.summary(trace)

        mu = np.array(summary["mean"])
        sd = np.array(summary["sd"])
        # v_propa = np.array(summary[""])

        if verbose:
            print("Percent divergent traces: %.2f %%"%(trace['diverging'].nonzero()[0].size / len(trace) * 100))

        return trace, summary, mu, sd

    def fit_xy_posterior(self, trace):
        """ returns position and uncertainty given a trace"""
        x_arr = trace['x'][:, 0]
        y_arr = trace['x'][:, 1]
        np_mu_x = np.mean(x_arr)
        np_sd_x = np.std(x_arr)
        np_mu_y = np.mean(y_arr)
        np_sd_y = np.std(y_arr)

        mu = [np_mu_x, np_mu_y]
        sd = [np_sd_x, np_sd_y]

        return mu, sd

    def forward(self, x, v=c.value):
        "predict time of flight for given source position"
        d = np.linalg.norm(self.stations-x, axis=1)
        t1 = d/v# time of flight values
        return t1

    def t_emission(self, trace):
        t = np.mean(trace['t0'])
        t_sigma = np.std(trace['t0'])
        return [t,t_sigma]

    def v_emission(self, trace):
        v = np.mean(trace['v'])
        v_sigma = np.std(trace['v'])
        return [v,v_sigma]


def mkdirectory(directory):
    dir = directory
    isExist = os.path.exists(dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir)
        print("The new directory is created!")
        print(dir)
    return dir


def triangulate(coords, times,t_cadence=60,v_sd=3E5, chains=4, cores=4, N_SAMPLES=2000, progressbar=True, report=0, plot=0, traceplot=False, savetraceplot = False, showplot=False, traceplotdir="", traceplotfn=""):
    """"
    Input: spacecraft : array (Nx2) N is number of stations
        First column is time, Second column is location.
    """
    B = BayesianTOAPositioner(coords, t_cadence=t_cadence, v_sd=v_sd)
    trace, summary, _, _ = B.sample(times, draws=N_SAMPLES, tune=N_SAMPLES, chains=chains, cores=cores, progressbar=progressbar)



    # analysis
    mu, sd = B.fit_xy_posterior(trace)
    t1_pred = B.forward(mu)
    t_emission = B.t_emission(trace)
    v_analysis = B.v_emission(trace)


    if report == 1:
        # report
        print(summary)


    if traceplot is True:
        # trace plot
        left = 0.04  # the left side of the subplots of the figure
        right = 0.972  # the right side of the subplots of the figure
        bottom = 0.076  # the bottom of the subplots of the figure
        top = 0.966  # the top of the subplots of the figure
        wspace = 0.166  # the amount of width reserved for blank space between subplots
        hspace = 0.498  # the amount of height reserved for white space between subplots

        from scipy.stats import norm
        csfont = {'fontname': 'Times New Roman'}

        ax = az.plot_trace(trace, compact=False,var_names=('x','v','t0'), figsize=(11,9))
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


        # X0 CHAINS PLOT LINES
        sd_y_axis_XCOORD = norm.pdf(mu[0]-sd[0], mu[0], sd[0])
        # ax[0, 0].hlines(sd_y_axis_XCOORD, mu[0] - sd[0], mu[0] + sd[0])                     # HORIZONTAL stedv line
        x_axis = np.arange(trace['x'][:,0].min(), trace['x'][:,0].max(), 0.1*R_sun.value)   #
        ax[0,0].plot(x_axis, norm.pdf(x_axis, mu[0], sd[0]))                                # GAUSSIAN FIT OF ALL CHAINS

        ax[0, 0].vlines(mu[0],0, ax[0, 0].get_ylim()[1])                                    # MEAN VALUE

        ax[0, 0].vlines(mu[0] - sd[0],0, sd_y_axis_XCOORD)                                  # -stdev
        ax[0, 0].vlines(mu[0] + sd[0],0, sd_y_axis_XCOORD)                                  # +stdev


        # X1 CHAINS PLOT LINES
        sd_y_axis_YCOORD = norm.pdf(mu[1]-sd[1], mu[1], sd[1])
        # ax[1, 0].hlines(sd_y_axis_YCOORD, mu[1] - sd[1], mu[1] + sd[1])                     # HORIZONTAL stedv line
        x_axis = np.arange(trace['x'][:,1].min(), trace['x'][:,1].max(), 0.1*R_sun.value)   #
        ax[1,0].plot(x_axis, norm.pdf(x_axis, mu[1], sd[1]))                                # GAUSSIAN FIT OF ALL CHAINS

        ax[1, 0].vlines(mu[1],0, ax[1, 0].get_ylim()[1])                                    # MEAN VALUE

        ax[1, 0].vlines(mu[1] - sd[1],0, sd_y_axis_YCOORD)                                  # -stdev
        ax[1, 0].vlines(mu[1] + sd[1],0, sd_y_axis_YCOORD)                                  # +stdev

        # V CHAINS PLOT LINES
        v_sampled, v_sd_sampled = [np.mean(trace['v']),np.std(trace['v'])]
        sd_y_axis_VCOORD = norm.pdf(v_sampled-v_sd_sampled, v_sampled, v_sd_sampled)
        # ax[1, 0].hlines(sd_y_axis_YCOORD, mu[1] - sd[1], mu[1] + sd[1])                     # HORIZONTAL stedv line
        x_axis = np.linspace(trace['v'].min(), trace['v'].max(), 50)   #
        ax[2,0].plot(x_axis, norm.pdf(x_axis, v_sampled, v_sd_sampled))                               # GAUSSIAN FIT OF ALL CHAINS

        maxval = ax[2, 0].get_ylim()[1]
        ax[2, 0].vlines(c.value+c.value*0.001, 0, maxval, 'r', alpha=0.3)                     # c+1% VALUE
        ax[2, 0].vlines(c.value, 0, maxval, 'r', alpha=0.6)                                   # c VALUE
        ax[2, 0].vlines(c.value-c.value*0.001, 0,maxval, 'r', alpha=0.3)                     # c-1% C VALUE



        ax[2, 0].vlines(v_sampled,0,norm.pdf(v_sampled, v_sampled, v_sd_sampled))            # MEAN VALUE

        ax[2, 0].vlines(v_sampled - v_sd_sampled,0, sd_y_axis_VCOORD)                                  # -stdev
        ax[2, 0].vlines(v_sampled + v_sd_sampled,0, sd_y_axis_VCOORD)                                  # +stdev

        ax[2, 1].hlines(c.value,0, ax[2, 1].get_xlim()[1], 'r', alpha=0.6)                    # C VALUE




        # t0 CHAINS PLOT LINES
        t_sampled, t_sd_sampled = [np.mean(trace['t0']), np.std(trace['t0'])]
        sd_y_axis_tCOORD = norm.pdf(t_sampled-t_sd_sampled, t_sampled, t_sd_sampled)

        x_axis = np.linspace(trace['t0'].min(), trace['t0'].max(), 50)   #
        ax[3,0].plot(x_axis, norm.pdf(x_axis, t_sampled, t_sd_sampled))                       # GAUSSIAN FIT OF ALL CHAINS

        ax[3, 0].vlines(t_sampled,0,norm.pdf(t_sampled, t_sampled, t_sd_sampled))            # MEAN VALUE

        ax[3, 0].vlines(t_sampled - t_sd_sampled,0, sd_y_axis_tCOORD)                                  # -stdev
        ax[3, 0].vlines(t_sampled + t_sd_sampled,0, sd_y_axis_tCOORD)                                  # +stdev

        ax[0, 0].title.set_text('')
        ax[1, 0].title.set_text('')
        ax[2, 0].title.set_text('')
        ax[3, 0].title.set_text('')
        ax[0, 1].title.set_text('')
        ax[1, 1].title.set_text('')
        ax[2, 1].title.set_text('')
        ax[3, 1].title.set_text('')

        ax[0, 0].set_xlabel(r'HEE - X ($R_{\odot}$)')
        ax[1, 0].set_xlabel(r'HEE - Y ($R_{\odot}$)')
        ax[2, 0].set_xlabel('v (c normalised)')
        ax[3, 0].set_xlabel('t0 (s)')



        # ax[0, 0].set_ylabel('N ',rotation=0)
        # ax[1, 0].set_ylabel('N ',rotation=0)
        # ax[2, 0].set_ylabel('N ',rotation=0)
        # ax[3, 0].set_ylabel('N ',rotation=0)

        # ax[0, 0].yaxis.set_ticks([0, 1])


        ax[0, 1].set_xlabel('HEE - X sampling')
        ax[1, 1].set_xlabel('HEE - Y sampling')
        ax[2, 1].set_xlabel('v sampling')
        ax[3, 1].set_xlabel('t0 sampling')

        # ax[0, 0].text(0.01, 0.78, "(a)", fontsize=25, transform=ax[0, 0].transAxes)
        # ax[1, 0].text(0.01, 0.78, "(b)", fontsize=25, transform=ax[1, 0].transAxes)
        # ax[2, 0].text(0.01, 0.78, "(c)", fontsize=25, transform=ax[2, 0].transAxes)
        # ax[3, 0].text(0.01, 0.78, "(d)", fontsize=25, transform=ax[3, 0].transAxes)

        ax[0, 0].tick_params(axis='both', which='major', labelsize=15)
        ax[1, 0].tick_params(axis='both', which='major', labelsize=15)
        ax[2, 0].tick_params(axis='both', which='major', labelsize=15)
        ax[3, 0].tick_params(axis='both', which='major', labelsize=15)
        ax[0, 1].tick_params(axis='both', which='major', labelsize=15)
        ax[1, 1].tick_params(axis='both', which='major', labelsize=15)
        ax[2, 1].tick_params(axis='both', which='major', labelsize=15)
        ax[3, 1].tick_params(axis='both', which='major', labelsize=15)

        scale_x = R_sun.value

        # ax[0, 0].xaxis.set_ticks(scale_x*np.arange((mu[0]/scale_x).astype(int)-50,(mu[0]/scale_x).astype(int)+60,10))
        ticks = ax[0, 0].get_xticks() / scale_x
        np.around(ticks, decimals=0, out=ticks)
        ax[0, 0].set_xticklabels(ticks.astype(int))

        # ax[1, 0].xaxis.set_ticks(scale_x*np.arange((mu[1]/scale_x).astype(int)-50,(mu[1]/scale_x).astype(int)+60,10))
        ticks = ax[1, 0].get_xticks() / scale_x
        np.around(ticks, decimals=0, out=ticks)
        ax[1, 0].set_xticklabels(ticks.astype(int))

        # ax[2, 0].xaxis.set_ticks(c.value*np.arange(997, 1004, 1)*0.001)
        ticks = ax[2, 0].get_xticks() / c.value
        np.around(ticks, decimals=3, out=ticks)
        ax[2, 0].set_xticklabels(ticks)


        ax[3, 0].xaxis.set_ticks(np.arange((t_sampled-2.9*t_sd_sampled).astype(int),(t_sampled+2.9*t_sd_sampled).astype(int),15))
        ticks = (ax[3, 0].get_xticks() *-1).astype(int)
        # np.around(ticks, decimals=0, out=ticks).astype(int)
        ax[3, 0].set_xticklabels(ticks)
        ax[3, 0].set_xlim([t_sampled-2.9*t_sd_sampled, t_sampled+2.9*t_sd_sampled])



        ticks = ax[0, 1].get_yticks() / scale_x
        np.around(ticks, decimals=0, out=ticks)
        ax[0, 1].set_yticklabels(ticks.astype(int))

        ticks = ax[1, 1].get_yticks() / scale_x
        np.around(ticks, decimals=0, out=ticks)
        ax[1, 1].set_yticklabels(ticks.astype(int))

        # ax[2, 1].yaxis.set_ticks(c.value*np.arange(980, 1020, 5)*0.001)
        ticks = ax[2, 1].get_yticks() / c.value
        np.around(ticks, decimals=3, out=ticks)
        ax[2, 1].set_yticklabels(ticks)

        ax[3, 1].yaxis.set_ticks(np.arange((t_sampled-2.9*t_sd_sampled).astype(int),(t_sampled+2.9*t_sd_sampled).astype(int),15))
        ticks = (ax[3, 1].get_yticks() *-1).astype(int)
        # np.around(ticks, decimals=0, out=ticks).astype(int)
        ax[3, 1].set_yticklabels(ticks)
        ax[3, 1].set_ylim([t_sampled-2.9*t_sd_sampled, t_sampled+2.9*t_sd_sampled])
        # plt.show(block=False)

        if savetraceplot is True:
            t0 = times[0]
            if traceplotdir=="":
                direct = mkdirectory("./Traceplots/")
            else:
                direct = mkdirectory(f"./Traceplots/{traceplotdir}")

            if traceplotfn == "":
                plt.savefig(direct + "/bella_traceplot.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
            else:
                plt.savefig(direct + f"/traceplot_{traceplotfn}", bbox_inches='tight', pad_inches=0.01, dpi=300)
                print(f"saved {direct}/traceplot_{traceplotfn}")


        if showplot is True:
            plt.show(block=False)
        else:
            plt.close()
        # pm.autocorrplot(trace)
        # pm.plot_posterior(trace)

    if plot == 1:
        # local map
        plt.figure(figsize=(5, 5))
        spacecraft = coords / R_sun.value

        plt.scatter(spacecraft[:, 0], spacecraft[:, 1], marker="^", s=80, label="Spacecraft")
        ell = matplotlib.patches.Ellipse(xy=(mu[0] / R_sun.value, mu[1] / R_sun.value),
                                         width=4 * sd[0] / R_sun.value, height=4 * sd[1] / R_sun.value,
                                         angle=0., color='black', lw=1.5)
        plt.plot(mu[0] / R_sun.value, mu[1] / R_sun.value, 'k*')
        ell.set_facecolor('none')
        plt.gca().add_patch(ell)
        # plt.legend(loc=1)
        # plt.xlim(-1050, 1050)
        # plt.ylim(-1050, 1050)
        plt.xlabel(r"'HEE - X / $R_{\odot}$'")
        plt.ylabel(r"'HEE - Y / $R_{\odot}$'")
        # plt.savefig("bayes_positioner_result2.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
        if showplot is True:
            plt.show(block=False)
        else:
            plt.close()
    return mu, sd, t1_pred, trace, summary, t_emission, v_analysis


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    print(f'Running on PyMC3 v{pm.__version__}')
    save_vid = False

    ncores_cpu = multiprocessing.cpu_count()

    # Date of the observation
    day = 11
    month = 7
    year = 2020
    date_str = f"{year}_{month:02d}_{day:02d}"
    spacecraft = ["wind", "stereo", "psp", "solo"]


    stations = np.array([[1, 0], [0, 1], [-1, 0],[0,-1]])*au.value
    stations_rsun = stations/R_sun.value



    # TEST SOURCE
    xx, yy = -10.00900424292337, -21.632364008898904    #  n = 2
    # xx, yy = -7.3, 8.1        #  n = 2
    x_true = np.array([ xx*R_sun.value,yy*R_sun.value])# true source position (m)
    v_true = c.value# speed of light (m/s)
    t0_norm = dt.datetime(year,month,day, 0,0,0) # Midnight
    t0_true = dt.datetime(year,month,day, 2,17,2)# source time. can be any constant, as long as it is within the uniform distribution prior on t0
    d_true = np.linalg.norm(stations-x_true, axis=1)
    t1_true = d_true/v_true# true time of flight values
    # t_obs = t1_true-t0_true# true time difference of arrival values

    t0_from_midnight = t0_true-t0_norm
    t_obs = t0_from_midnight.seconds + t1_true


    # np.random.seed(1)
    percent = 1
    NoiseLevel = percent/100
    t_obs = t_obs+np.random.normal(loc=0.0, scale=t_obs.mean()*NoiseLevel, size=len(t_obs))# noisy observations

    # t_obs = [288, 248, 210, 96]         # 0.7
    # # t_obs =[167, 80, 100, 20]           # 3
    mu, sd, t1_pred, trace, summary, t_emission, v_analysis = triangulate(stations, t_obs, report=1, plot=1, t_cadence = 60)




    # report
    print(summary)
    print(f"t0_true: {t0_true}")
    print(f"t1_true: {t1_true}")
    print(f"t_obs:    {t_obs}")
    # print(f"t1_pred: {t1_pred}")
    print(f"stations: {stations}")
    print(f"Source  : ({xx}, {yy})")
    print(f"Detected: ({mu[0]/R_sun.value}, {mu[1]/R_sun.value})")








    # FIGURE
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    plt.subplots_adjust(top=1, bottom=0)
    earth_orbit = plt.Circle((0, 0), au/R_sun, color='k', linestyle="dashed", fill=None)
    ax.add_patch(earth_orbit)
    ax.scatter(stations_rsun[:,0], stations_rsun[:,1],color = "w", marker="^",edgecolors="k", s=80, label="Spacecraft")
    ax.set_aspect('equal')
    ax.set_xlim(-250,250)
    ax.set_ylim(-250,250)
    ax.plot(au / R_sun, 0, 'bo', label="Earth")
    ax.plot(0, 0, 'yo', label="Sun")
    ax.plot(x_true[0] / R_sun.value, x_true[1] / R_sun.value, "r*", label="source")
    ell = matplotlib.patches.Ellipse(xy=(mu[0] / R_sun.value, mu[1] / R_sun.value),
                                     width=2 * sd[0] / R_sun.value, height=2 * sd[1] / R_sun.value,
                                     angle=0., color='black', lw=1.5)
    plt.plot(mu[0] / R_sun.value, mu[1] / R_sun.value, 'k*', label="detection")

    ell.set_facecolor('none')
    plt.gca().add_patch(ell)

    ax.legend(loc=1)

    ax.set_xlabel(r"'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel(r"'HEE - Y / $R_{\odot}$'", fontsize=22)

    plt.show(block=False)
