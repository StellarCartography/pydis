"""
Do some basic flux correction tasks I want it to do:

1) fluxcor.normalize
    normalize out the flux continuum, can use spline or polynomial

2) fluxcor.calibrate
    calibrate standard star spectrum against a database spectrum

3) fluxcor.airmass
    correct for airmass extinction
"""

import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy.polynomial import chebyshev


def normalize(wave, flux, spline=False, poly=True, order=3, interac=True):
    if (poly is False) and (spline is False):
        poly=True

    if (poly is True):
        print("yes")

    return


def _mag2flux(mag, zeropt=21.10):
    flux = 10.0**( (mag + zeropt) / (-2.5) )
    return flux


def AirmassCor(obj_wave, obj_flux, airmass):
    dir = os.path.dirname(os.path.realpath(__file__))
    air_wave, air_trans = np.loadtxt(dir+'/resources/apoextinct.dat',unpack=True,skiprows=2)
    airmass_ext = np.interp(obj_wave, air_wave, air_trans) / airmass
    return airmass_ext


def calibrate(stdobs, stdstar='g191b2b', airmass=1.0):
    stdstar = stdstar.lower()
    # important! need to do calibrate in separate steps for airmass, sensfunc, etc
    # also, need to be able to call it within main routines (from autoreduce)

    dir = os.path.dirname(os.path.realpath(__file__))
    onedstdpath = dir + '/resources/onedstds/spec50cal/'

    std_wave0, std_mag, std_wth = np.loadtxt(onedstdpath + stdstar + '.dat',
                                            skiprows=1, unpack=True)
    std_flux0 = _mag2flux(std_mag)


    obj_wave, obj_cts = np.loadtxt(dir+'/G191B2B.0020r.fits.spec',skiprows=1,
                                   unpack=True,delimiter=',')

    # std_wave = np.arange(np.nanmin(obj_wave), np.nanmax(obj_wave),
    #                      np.mean(np.abs(std_wave0[1:]-std_wave0[:-1])))
    # std_flux = np.interp(std_wave, std_wave0, std_flux0)

    std_wave = std_wave0
    std_flux = std_flux0


    # down-sample (ds) the observed counts
    obj_cts_ds = []
    obj_wave_ds = []
    std_flux_ds = []
    for i in range(len(std_wave)):
        rng = np.where((obj_wave>std_wave[i]) &
                       (obj_wave<std_wave[i]+std_wth[i]) )
        if (len(rng[0]) > 1):
            obj_cts_ds.append(np.sum(obj_cts[rng])/std_wth[i])
            obj_wave_ds.append(std_wave[i])
            std_flux_ds.append(std_flux[i])

    plt.figure()
    plt.plot(obj_wave, obj_cts,'b')
    plt.plot(obj_wave_ds, obj_cts_ds, 'ro')
    plt.xlabel('Wavelength')
    plt.show()

    ratio = np.array(std_flux_ds,dtype='float') / np.array(obj_cts_ds,dtype='float')

    ratio_spl = UnivariateSpline(obj_wave_ds, ratio, ext=0, k=3 ,s=0)
    # ratio_fit = np.polyfit(obj_wave_ds, ratio, 11)
    # ratio_chb = chebyshev.chebfit(obj_wave_ds, ratio, 9)

    # the width of each pixel (in angstroms)
    dw_tmp = obj_wave[1:]-obj_wave[:-1]
    dw = np.abs(np.append(dw_tmp, dw_tmp[-1]))

    plt.figure()
    plt.plot(obj_wave_ds, ratio, 'ko')
    plt.plot(obj_wave, ratio_spl(obj_wave),'r')
    # plt.plot(obj_wave, np.polyval(ratio_fit, obj_wave),'g')
    # plt.plot(obj_wave, chebyshev.chebval(obj_wave,ratio_chb),'b')
    plt.ylabel('(erg/s/cm2/A) / (counts/s)')
    plt.show()


    sens = ratio_spl(obj_wave)
    # sens2 = np.polyval(ratio_fit, obj_wave)
    # sens3 = chebyshev.chebval(obj_wave,ratio_chb)


    plt.figure()
    plt.plot(std_wave0, std_flux0,'ko')
    plt.plot(obj_wave, obj_cts/dw * sens,'r',alpha=0.5)
    # plt.plot(obj_wave, obj_cts/dw * sens2,'g')
    # plt.plot(obj_wave, obj_cts/dw * sens3,'b')
    plt.title(stdstar)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux (erg/s/cm2/A)')
    plt.show()


    return
