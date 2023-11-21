============
BayEsian LocaLisation Algorithm - BELLA
============

.. image:: http://img.shields.io/badge/powered%20by-SunPy-orange.svg?style=flat
    :target: http://www.sunpy.org
    :alt: Powered by SunPy Badge
    
.. image:: https://img.shields.io/badge/powered%20by-PyMC-blue
    :target: http://www.pymc.io
    :alt: Powered by PyMC Badge
    
.. image:: https://img.shields.io/badge/powered%20by-SolarMAP-orange
    :target: https://pypi.org/project/solarmap/
    :alt: Powered by SolarMAP Badge
    
    
BELLA uses Bayesian statistics to Localise sources of EM emission within one astronomical unit of the Sun.

Features:

- Type III Fitter: Fits a Type III radio burst and returns scatter data and/or fitted data, prepared for BELLA multilateration. Uses Kapteyn to fit the lightcurve of a Type III solar radio burst using Gaussian-Hermite polynomials.

    typeIIIfitter: file with functions to fit a Type III lightcurve and fit the Type III morphology.

    stacked_dyn_spectra: example file of data extraction of an event.


- BELLA Multilaterate: Uses pymc3 to perform Bayesian multilateration.

    bayes_positioner: Bayesian Multilateration class and multilateration function.

    bayesian_tracker: Functions for performing BELLA multilateration.

    bella_plotter: Functions for plotting BELLA multilateration results.

    bella_triangulation: Example of BELLA multilateration event.

    positioner_mapping_parallel: Generates simulated data of the uncertainty map given a spacecraft configuration.

    density_models: support file with density models and plasma physics functions.


Contributions and comments are welcome using Github at: 
https://github.com/TCDSolar/BELLA_Multilateration/




Please note that BELLA requires:

- PyMC3
- SunPy 
- Astropy
- Solarmap
- Kapteyn
- Pyspedas
- Radiospectra


Installation
============
You must run Fitter and Multilaterate in **different** virtual environments. This is due to package incompatibilities.
It is also recommended to make aliases in order to quickly change between environments.

Some packages such as Kapteyn or pyspedas, may give an error when installing in python different than python 3.8.
You must install virtual environments with python 3.8.

Install BELLA Type III Fitter
----

1 - Make a virtual env:

.. code-block::

    python3.8 -m venv ./bellaenv_fitter
    source ./bellaenv_fitter/bin/activate

2 - Install HDF5:

.. code-block::

        brew install hdf5

3 - Install the following packages:

.. code-block::

    pip install cython numpy h5py solarmap pyspedas


4 - Install Kapteyn:

Kapteyn is a package developed and managed by the University of Groningen. It is recommended to install Kapteyn using
their local instructions. Note: "pip install kapteyn" gives error:

https://www.astro.rug.nl/software/kapteyn/intro.html#installinstructions

.. code-block::

    pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.4.tar.gz


5 - Install radiospectra via pip. Note the current version of radiospectra gives error when running BELLA,
for now install this stable version:

.. code-block::

    pip install git+https://github.com/samaloney/radiospectra.git@6c1faa39d9eba52baec7f7bdc75966e5d8da3b81

6 - (Optional) Add alias to .bashrc // .bash_profile

.. code-block::

    echo 'alias fitter="source $(pwd)/bellaenv_fitter/bin/activate"' > $(HOME)/.bashrc

or

.. code-block::

    echo 'alias fitter="source $(pwd)/bellaenv_fitter/bin/activate"' > $(HOME)/.bash_profile


Install BELLA Multilaterate
----

1 - Make a virtual env:

.. code-block::

    python3.8 -m venv ./bellaenv_multilat
    source ./bellaenv_multilat/bin/activate

2 - Install packages via pip:

.. code-block::

    pip install Theano-PyMC astropy joblib solarmap termcolor pymc3


3 - (Optional) Add alias to .bashrc // .bash_profile

.. code-block::

    echo 'alias multilat="source $(pwd)/bellaenv_multilat/bin/activate"' > $(HOME)/.bashrc

or

.. code-block::

    echo 'alias multilat="source $(pwd)/bellaenv_multilat/bin/activate"' > $(HOME)/.bash_profile


Usage
=====

1 -  In **fitter** environment open **Type_III_Fitter/stacked_dyn_spectra_....py**

    -  Select date and time range. The code has been tested to run with leadingedge. (Running backbone might need the code to be updated.)

    .. code-block::

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

    - Follow the code and comments to adapt the code to your needs. You might consider changing:

        - Histogram levels - > Make Type III visible or improve contrast.
        - Automatic detection settings - > Change initial inputs for automatic detection.
        - Fine tuning of detected points. - > Fix outliers that make unphysical morphologies.

    - Run stacked_dyn_spectra_YYYY_MM_dd.py

    .. code-block::

        cd PATH/TO/Type_III_Fitter
        python stacked_dyn_spectra_YYYY_MM_dd.py

    - Once the stacked file has run. There should be two files generated in PATH/TO/Type_III_Fitter/Data/TypeIII/YYYY_MM_dd. These files are the extracted data, ready for multilateration.

    - The output of stacked should show all the dynamic spectra with solid black line as the fit and dashed lines representing the cadence chosen for the multilateration:

    .. image:: ./Figures_readme/stackedoutput.png
        :align: center

    - A directory showing all the lightcurve fits and automatic detections should have been generated in PATH/TO/Type_III_Fitter/lightcurves:

    .. image:: ./Figures_readme/STEREOA_sigma_0.98.jpg
        :align: center


2 - In **multilat** environment open **Multilaterate/positioner_mapping_parallel.py** to generate background uncertainty map.

    - Select the date. If "surround", "test" or "manual" are selected in date string you may manually input any location for any amount of spacecraft. Note: surround is a particular orbital configuration, see https://www.dias.ie/surround/ for more information.

    .. code-block::

        day = 7
        month = 6
        year = 2012
        date_str = f"{year}_{month:02d}_{day:02d}"
        # date_str = f"surround"

        if date_str == "surround":
            # SURROUND
            #############################################################################
            theta_sc = int(sys.argv[1])

            print(f"theta_sc:    {theta_sc}")
            L1 = [0.99*(au/R_sun),0]
            L4 = [(au/R_sun)*np.cos(radians(60)),(au/R_sun)*np.sin(radians(60))]
            L5 = [(au/R_sun)*np.cos(radians(60)),-(au/R_sun)*np.sin(radians(60))]
            # ahead = [(au/R_sun)*np.cos(radians(theta_sc)),(au/R_sun)*np.sin(radians(theta_sc))]
            # behind = [(au/R_sun)*np.cos(radians(theta_sc)),-(au/R_sun)*np.sin(radians(theta_sc))]

            dh = 0.01
            # theta_AB_deg = 90
            theta_AB = np.radians(theta_sc)
            ahead =  pol2cart((1-dh)*(au / R_sun), theta_AB)
            behind = pol2cart((1+dh)*(au / R_sun),-theta_AB)



            stations_rsun = np.array([L1, ahead, behind])
            #############################################################################
        elif date_str == "test":
            stations_rsun = np.array([[200, 200], [-200, -200], [-200, 200], [200, -200]])
        elif date_str == "manual":
            stations_rsun = np.array([[45.27337378, 9.90422281],[-24.42715218,-206.46280171],[ 212.88183411,0.]])
            date_str = f"{year}_{month:02d}_{day:02d}"
        else:
            solarsystem = solarmap.get_sc_coord(date=[year, month, day], objects=["stereo_b", "stereo_a", "earth"])
            stations_rsun = np.array(solarsystem.locate_simple())


    - Select the spacecraft. Note for this particular date we use "earth" instead of "wind". The reason is Wind ephemeris is not available prior to A.D. 2019-OCT-08 00:01:09.1823 TD on Horizons. So 99% of Sun-Earth distance is assumed.

    .. code-block::

            solarsystem = solarmap.get_sc_coord(date=[year, month, day], objects=["stereo_b", "stereo_a", "earth"])
            stations_rsun = np.array(solarsystem.locate_simple())

    Redefine earth as Wind.

    .. code-block::

        spacecraft = ["stereo_b", "stereo_a", "wind"] # redefining wind as the name of the spacecraft
        stations_rsun[2][0] = 0.99 * stations_rsun[2][0]


    - Make the grid. **CAREFULLY** make your grid in Rsun units. The finer the grid (smaller xres) the longer it will take to run. An estimate of how long the code will take to run will be shown. You may improve this estimate by changing the time per loop "tpl_l" and "tpl_h" based on your machine performance.

    .. code-block::

        # Making grid
        xrange = [-250,250]
        xres = 10
        yrange = [-250, 250]
        yres = xres
        xmapaxis = np.arange(xrange[0], xrange[1], xres)
        ymapaxis = np.arange(yrange[0], yrange[1], yres)


    - Select the cadence. A smaller cadence will lead to lower uncertainty results but will also lead to divergencies. Here we pick the conservative 60s cadence.

    .. code-block::

        cadence = 60


    - Run **positioner_mapping_parallel.py**. Depending on your grid size, resolution and machine specs this step may take a few hours.

    .. code-block::

        cd PATH/TO/Multilaterate
        python positioner_mapping_parallel.py

    - A file with the uncertainty bg results should be available in **PATH/TO/Multilaterate/Data/YYYY_MM_dd/bg/**

    - If the showfigure=True then your ouput should look like:

    .. image:: ./Figures_readme/bayes_positioner_map_median_-250_250_-250_250_10_10_3.jpg
        :align: center


3 - In multilat environment open **Multilaterate/bella_triangulation_YYYY_MM_dd.py**

    - Follow the code and adjust settings according to your needs.

    - Run **bella_triangulation_YYYY_MM_dd.py**. This step may take from minutes to hours depending on your frequency range and resolution.

    .. code-block::

        cd PATH/TO/Multilaterate
        python bella_triangulation_YYYY_MM_dd.py


    - A file with the multilateration results should be available in **PATH/TO/Multilaterate/Data/YYYY_MM_dd/**

4 - In multilat environment open **Multilaterate/bella_plotter.py**

    - Follow the code and adjust settings accordingly. Make sure that the data filenames are correct.

    - Run bella_plotter.py

    .. code-block::

        cd PATH/TO/Multilaterate
        python bella_plotter.py

    .. image:: ./Figures_readme/bellaplotteroutput.png
    :align: center


Documentation
=============
BELLA uses a class in **bayes_positioner.py** called **BayesianTOAPositioner** adapted from benmoseley (https://github.com/benmoseley).
This class sets up a context manager for pymc3. This is where you can define your prior distributions.
Note v can be a Normal Distribution or Truncated Normal depending on whether you want to test if v is converging at c or whether you want to make c a limit.

.. code-block::

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


The function "triangulate()" (soon to be multilaterate) found in bayes_positioner.py allows for one pymc3 multilateration loop, generates traceplots and also a quickview plot of the results. Some important information:

    - cores=4 is the maximum pymc3 will allow. run cores=0 if triangulate is already running in a parallel process.
    - chains=4. Generally recommended to use 4 chains.
    - t_cadence=60. It is recommended to use a cadence that is equal or slightly worse than the instruments cadence. Otherwise divergences may occur.
    - N_SAMPLES=2000. The number of samples and tunning values are 2000 because a larger number becomes computationally expensive. Tuning values and Samples are chosen to be equal but this is not necessary. Change if you need to.



bella_triangulation_YYYY_MM_dd.py or bayesian_tracker.py run triangulate() in a for loop in parallel. Make sure cores=0 in triangulate() if running triangulate() in a for loop in parallel.




Bugs and Warnings
===================
WARNING: always import pymc3 before importing theano. If theano is imported first you might have to restart your shell.

WARNING: theano's cache might fill up. This usually happens when running several processes in parallel. To fix this run
this in your bash shell:

.. code-block::

    theano-cache purge

WARNING: Running BELLA scripts often requires parallelisation. By default BELLA will maximise the number of cores to be used. As a result of this, running several BELLA scripts simultaneously will cause problems.

Disclaimer: BELLA multilateration is relatively computationally expensive and there is room for speeding up the processes. Development of a faster computation is ongoing and contributions to making BELLA faster are welcome.

Please use Github to report bugs, feature requests and submit your code:
https://github.com/TCDSolar/BELLA_Multilateration/

:author: Luis Alberto Canizares
:date: 2022/11/22
