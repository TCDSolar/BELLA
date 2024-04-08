# BayEsian LocaLisation Algorithm - BELLA

![Powered by SunPy Badge](http://img.shields.io/badge/powered%20by-SunPy-orange.svg?style=flat)(http://www.sunpy.org)
![Powered by PyMC Badge](https://img.shields.io/badge/powered%20by-PyMC-blue)(http://www.pymc.io)
![Powered by SolarMAP Badge](https://img.shields.io/badge/powered%20by-SolarMAP-orange)(https://pypi.org/project/solarmap/)

BELLA uses Bayesian statistics to Localise sources of EM emission within one astronomical unit of the Sun.

## Features:

- **Type III Fitter**: Fits a Type III radio burst and returns scatter data and/or fitted data, prepared for BELLA multilateration. Uses Kapteyn to fit the lightcurve of a Type III solar radio burst using Gaussian-Hermite polynomials.
  - `typeIIIfitter`: file with functions to fit a Type III lightcurve and fit the Type III morphology.
  - `stacked_dyn_spectra`: example file of data extraction of an event.

- **BELLA Multilaterate**: Uses pymc3 to perform Bayesian multilateration.
  - `bayes_positioner`: Bayesian Multilateration class and multilateration function.
  - `bayesian_tracker`: Functions for performing BELLA multilateration.
  - `bella_plotter`: Functions for plotting BELLA multilateration results.
  - `bella_triangulation`: Example of BELLA multilateration event.
  - `positioner_mapping_parallel`: Generates simulated data of the uncertainty map given a spacecraft configuration.
  - `density_models`: support file with density models and plasma physics functions.

Contributions and comments are welcome using Github at: 
[https://github.com/TCDSolar/BELLA_Multilateration/](https://github.com/TCDSolar/BELLA_Multilateration/)

Please note that BELLA requires:

- PyMC3
- SunPy 
- Astropy
- Solarmap
- Kapteyn
- Pyspedas
- Radiospectra

## Installation

You must run Fitter and Multilaterate in **different** virtual environments. This is due to package incompatibilities.
It is also recommended to make aliases in order to quickly change between environments.

Some packages such as Kapteyn or pyspedas, may give an error when installing in python different than python 3.8.
You must install virtual environments with python 3.8.

### Install BELLA Type III Fitter

1. Make a virtual env: 

    ```bash
    python3.8 -m venv ./bellaenv_fitter
    source ./bellaenv_fitter/bin/activate
    ```

2. Install HDF5:

    ```bash
    brew install hdf5
    ```

    See [https://github.com/HDFGroup/hdf5](https://github.com/HDFGroup/hdf5) for other OS.

3. Install the following packages:

    ```bash
    pip install -r requirements_fitter.txt
    ```
4. Install Kapteyn:

    Kapteyn is a package developed and managed by the University of Groningen. It is recommended to install Kapteyn using their local instructions. Note: "pip install kapteyn" gives error:

    [https://www.astro.rug.nl/software/kapteyn/intro.html#installinstructions](https://www.astro.rug.nl/software/kapteyn/intro.html#installinstructions)

    ```bash
    pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.4.tar.gz
    ```

    Note: If installation of Kapteyn gives a Numpy error make sure Numpy was deprecated to numpy==1.22


