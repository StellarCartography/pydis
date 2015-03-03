# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:40:42 2015

@author: jradavenport

Try to make a simple one dimensional spectra reduction package

Based heavily on Python4Astronomers tutorial:
    https://python4astronomers.github.io/core/numpy_scipy.html

Plus some simple reduction (flat and bias) methods

Created with the Apache Point Observatory (APO) 3.5-m telescope's
Dual Imaging Spectrograph (DIS) in mind. YMMV

+++
Steps to crappy reduction to 1dspec:

1. flat and bias correct (easy)
2. identify lines in wavelenth cal image (HeNeAr) and define the
    wavelength solution in 2D space
3. trace the object spectrum, define aperture and sky regions
4. extract object, subtract sky, interpolate wavelength space


---
Things I'm not going to worry about:

- cosmic rays
- bad pixels
- overscan
- multiple objects on slit
- extended objects on slit
- flux calibration (yet)


"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdus = fits.open('3c120_stis.fits.gz')
primary = hdus[0].data  # Primary (NULL) header data unit
img = hdus[1].data      # Intensity data

plt.imshow(img, origin = 'lower')
