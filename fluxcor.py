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

def normalize(wave, flux, spline=False, poly=True, order=3, interac=True):
    if (poly is False) and (spline is False):
        poly=True

    if (poly is True):
        print("yes")

    return


def calibrate():
    print('calibrate')
    return
