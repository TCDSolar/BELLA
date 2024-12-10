import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import astropy.constants as const
import astropy.units as u
from scipy.optimize import root_scalar

matplotlib.rcParams.update({'font.size': 26})
###
# LeBlanc Density Model
def Ne_leblanc(rho) :
    '''
    https://ui.adsabs.harvard.edu/abs/1998SoPh..183..165L/exportcitation
    In: rho - heliocentric distance in R_sun
    Out : Density in per cc
    '''
    alpha_1, alpha_2, alpha_3 = 3.3e5, 4.1e6, 8.0e7
    return alpha_1*rho**-2 + alpha_2*rho**-4 + alpha_3*rho**-6

def Ne_leblanc_deriv(rho) :
    '''
    In: rho - heliocentric distance in R_sun
    Out : delta Density in per cc/ delta rho in R_sun
    '''
    alpha_1, alpha_2, alpha_3 = 3.3e5, 4.1e6, 8.0e7
    return -2*alpha_1*rho**-3 + -4*alpha_2*rho**-5 + -6*alpha_3*rho**-7

def Ne_saito(r) :
    '''
    https://link.springer.com/content/pdf/10.1007/BF00150879.pdf
    input : r[R_sun or au if au==True] # Radius from the Sun center
    output : Density in per cc
    '''
    alpha = 2.14
    beta = 6.13
    a = 1.36e6
    b = 1.68e8
    return (a/r**alpha + b/r**beta)

def Ne_saito_deriv(r) :
    '''
    In: r - heliocentric distance in R_sun
    Out : delta Density in per cc/ delta r in R_sun
    '''
    alpha = 2.14
    beta = 6.13
    a = 1.36e6
    b = 1.68e8
    return (-a*alpha/r**(alpha+1) + -beta*b/r**(beta+1))

def Ne_parker(r):
    '''
    Ref : https://ui.adsabs.harvard.edu/abs/1958ApJ...128..664P/abstract
    In : r # Heliocentric radius in R_S
    Out : n_e : heliospheric electorn density in cm^-3
    '''
    m1, m2, m3 = 1.39e6, 3e8, 4.8e9
    alpha, beta, gamma = 2.3, 6, 14
    return m1/r**alpha+m2/r**beta+m3/r**gamma


def Ne_mann(r, N0=8.775e8, D=9.88):
    # r in rsun
    n_e_model_MANN = N0 * np.exp(D * (1 / r - 1))  # Mann & Klassen (2005)
    return n_e_model_MANN

def Ne_allen(r, a0=2.99, a1=1.55, a2=0.036):
    # r in rsun
    n_e_model_ALLEN = 1e8 * ( a0*np.power(r,-16.) +  a1*np.power(r,-6.) + a2*np.power(r,-1.5))  # Allen
    return n_e_model_ALLEN


def newkirk(hr, f, a):
    """ [1] for fundamental backbone, [2] for Harmonic Backbone.
         f: frequency (MHz).
         a: fold (1 - 4, [1] for quiet Sun, [4] for active regions). """
    n_e = ((f*1e6)/(hr*9000))**2
    r_nk = (4.32/np.log10(n_e/(a*4.2e4)))
    return n_e, r_nk

def parker(hr, f, a):
    """ [1] for fundamental backbone, [2] for Harmonic Backbone.
         f: frequency (MHz).
         a: fold (1 - 4, [1] for quiet Sun, [4] for active regions). """
    n_e = Ne(f)*a
    r = R_parker(n_e, calibration_factor=1)

    return n_e, r







def Ne_nk(r,fold=1):
    N0 = 4.2E4
    return (fold * N0 * 10**(4.32/r))




def Ne_parker_deriv(r) :
    m1, m2, m3 = 1.39e6, 3e8, 4.8e9
    alpha, beta, gamma = 2.3, 6, 14
    return -alpha*m1/r**(alpha+1) - beta*m2/r**(beta+1) - gamma*m3/r**(gamma+1)

def Ne_helios(r):
    # https://link.springer.com/content/pdf/10.1007%2FBF00173965.pdf
    n0 = 6.1*u.cm**-3
    if not isinstance(r, u.quantity.Quantity):
        r *= u.R_sun
    return (n0 * (const.au.to('m')/r.to('m'))**2.1).to("cm^-3")

def Ne_moncuquet(r):
    # https://iopscience.iop.org/article/10.3847/1538-4365/ab5a84/pdf
    n0 = 10*u.cm**-3
    return (n0 * (const.au.to('m')/r.to('m'))**2).to("cm^-3")

# Plasma Frequency in MHz
def f_pe(Ne) :
    if not isinstance(Ne, u.quantity.Quantity):
        Ne *= u.cm**-3
    return ((80.6*Ne.to("cm^-3").value)**0.5/1e3)*u.MHz


def Ne(f_pe) :
    if not isinstance(f_pe,u.quantity.Quantity):
        f_pe *=  u.MHz
    return f_pe.to("kHz").value**2/80.6*u.cm**-3

def R_moncuquet(Ne) :
    if not isinstance(Ne,u.quantity.Quantity):
        Ne *= u.cm**-3
    n0 = 10*u.cm**-3
    return (n0/Ne)**0.5 * u.au



def find_radius_parker(target_ne, calibration_factor=1):
    def difference(r):
        return Ne_parker(r)*calibration_factor - target_ne
    return difference
def find_radius_saito(target_ne, calibration_factor=1):
    def difference(r):
        return Ne_saito(r)*calibration_factor - target_ne
    return difference
def find_radius_leblanc(target_ne, calibration_factor=1):
    def difference(r):
        return Ne_leblanc(r)*calibration_factor - target_ne
    return difference


def R_parker(target_ne, calibration_factor=1):
    target_ne = target_ne.value
    sol = root_scalar(find_radius_parker(target_ne, calibration_factor), bracket=[1, 215], method='brentq')
    if sol.converged:
        return sol.root
    else:
        return None  # or raise an error depending on how you want to handle failures

def R_saito(target_ne, calibration_factor=1):
    target_ne = target_ne.value
    sol = root_scalar(find_radius_saito(target_ne, calibration_factor), bracket=[1, 215], method='brentq')
    if sol.converged:
        return sol.root
    else:
        return None  # or raise an error depending on how you want to handle failures

def R_leblanc(target_ne, calibration_factor=1):
    target_ne = target_ne.value
    sol = root_scalar(find_radius_leblanc(target_ne, calibration_factor), bracket=[1, 215], method='brentq')
    if sol.converged:
        return sol.root
    else:
        return None  # or raise an error depending on how you want to handle failures





if __name__ == "__main__":
    fp_parker = []
    fp_saito = []
    fp_leblanc = []
    R = []
    ne_p = []
    ne_s = []
    ne_l = []
    for r in range(1, 600, 1):
        ne_parker = Ne_parker(r)
        ne_saito = Ne_saito(r)
        ne_leblanc = Ne_leblanc(r)

        ne_p.append(ne_parker)
        ne_s.append(ne_saito)
        ne_l.append(ne_leblanc)
        fp_parker.append(f_pe(ne_parker).value)
        fp_saito.append(f_pe(ne_saito).value)
        fp_leblanc.append(f_pe(ne_leblanc).value)
        R.append(r)

    fig, ax1 = plt.subplots(figsize=(10,10))

    ax1.plot(R, fp_parker, 'r-', label="Parker")
    ax1.plot(R, fp_saito, 'b-', label="Saito")
    ax1.plot(R, fp_leblanc, 'g-', label="Leblanc")
    ax1.axhline(y=25, xmin=0, xmax=200)
    ax1.axhline(y=0.02, xmin=0, xmax=200)
    ax1.axvline(x=215, ymin=0, ymax=1, ls='--', c="k")
    ax1.axvline(x=2, ymin=0, ymax=1, ls='--', c="k")
    ax1.text(1.6, 1200, r"2 R$_\odot$")
    ax1.text(150, 1200, r"215 R$_\odot$")
    ax1.text(250, 26, "25 MHz")
    ax1.text(250, 0.03, "0.02 MHz")
    ax1.set_xlabel(r"Distance from Sun (R$_\odot$)", fontsize=26)
    ax1.set_ylabel("Frequency (MHz)", fontsize=26)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.legend(loc='upper right', fontsize=18)


    ax2 = ax1.twinx()

    ax2.plot(R, ne_p, 'r-')
    ax2.plot(R, ne_s, 'b-')
    ax2.plot(R, ne_l, 'g-')

    # ax2.set_ylim(20000 * km3yearToSv, 70000 * km3yearToSv)
    ax2.set_ylabel('Electron density n$_e$ ($cm^{-3}$)', fontsize=26)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show(block=False)



    r = 215
    print(f"Leblanc : {Ne_leblanc(r)} cm^{-3}")
    print(f"Saito : {Ne_saito(r)} cm^{-3}")
    print(f"Parker : {Ne_parker(r)} cm^{-3}")
    print(f"Allen : {Ne_allen(r)} cm^{-3}")

    # print(f"BELLA : {Ne_allen(r, *popt)} cm^{-3}")
