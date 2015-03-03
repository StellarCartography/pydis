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

e.g. 
- have BLUE/RED channels
- hand-code in that the RED channel wavelength is backwards
- dispersion along the X, spatial along the Y

+++
Steps to crappy reduction to 1dspec:

1. flat and bias correct (easy)
2. identify lines in wavelenth cal image (HeNeAr) and define the
    wavelength solution in 2D space
3. trace the object spectrum, define aperture and sky regions
4. extract object, subtract sky, interpolate wavelength space


---
Things I'm not going to worry about:

- interactive everything (yet)
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
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import scipy.signal

def gaus(x,a,b,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+b

def spec_trace(img, nsteps=100):
    '''
    Aperture finding and tracing

    Parameters
    ----------
    img : 2d numpy array, required
        the input image, already trimmed

    Returns
    -------
    1d numpy array, y positions of the trace

    '''
    #--- find the overall max row, and width
    # comp_y = img.sum(axis=1)
    # peak_y = np.argmax(comp_y)
    # popt,pcov = curve_fit(gaus, np.arange(len(comp_y)), comp_y,
    #                       p0=[np.amax(comp_y), np.median(comp_y), peak_y, 2.])
    # peak_dy = popt[3]    
    

    # median smooth to crudely remove cosmic rays
    img_sm = scipy.signal.medfilt2d(img, kernel_size=(5,5))
    
    xbins = np.linspace(0, img.shape[1], nsteps)
    ybins = np.zeros_like(xbins)
    for i in range(0,len(xbins)-1):
        #-- simply use the max w/i each window
        #ybins[i] = np.argmax(img_sm[:,xbins[i]:xbins[i+1]].sum(axis=1))

        #-- fit gaussian w/i each window
        zi = img_sm[:,xbins[i]:xbins[i+1]].sum(axis=1)
        yi = np.arange(len(zi))
        popt,pcov = curve_fit(gaus, yi, zi,
                              p0=[np.amax(zi), np.median(zi), np.argmax(zi), 2.])
        ybins[i] = popt[2]
        
    mxbins = (xbins[:-1] + xbins[1:])/2.
    mybins = ybins[:-1]
    
    ap_spl = UnivariateSpline(mxbins, mybins, ext=0, k=3)
    
    mx = np.arange(0,img.shape[1])
    my = ap_spl(mx)
    return my



# test data setup
datafile = 'example_data/Gliese176.0052b.fits'
flatfile = 'example_data/flat.0039b.fits'
biasfile = 'example_data/bias.0014b.fits'

hdu = fits.open( datafile )

# trim the data to remove overscan region
datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
d = map(float, datasec)

img = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]]

disp_approx = hdu[0].header['DISPDW']
wcen_approx = hdu[0].header['DISPWC']


plt.figure()
plt.imshow(np.log10(img), origin = 'lower')
plt.plot(np.arange(img.shape[1]), spec_trace(img),'k',lw=3)
plt.show()


