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
from scipy.interpolate import SmoothBivariateSpline
import scipy.signal

#########################
def gaus(x,a,b,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+b

#########################
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

#########################
# def sky_subtract(image, My, apwidth=5, skysep=30, skybin=30):
#     skysubflux = np.zeros_like(My)
#     for i in range(0,len(My)):
#         fit a parabola outside +/- skysep from My skybin wide
#         skysubflux[i] = sum up image within apwidth, subtracting parabola
#         return skysubflux
#
#
##########################
# def HeNeAr_fit(calimage, linelist):
#     take a slice thru calimage, find the peaks
#     trace the peaks vertically, allow curve, spline as before
#
#    disp_approx = hdu[0].header['DISPDW']
#    wcen_approx = hdu[0].header['DISPWC']
#
#     fit against linelist
#
#     # scipy.interpolate.SmoothBivariateSpline
#     treat the wavelenth solution as a surface
#     return wavemap
#
##########################
# def mapwavelength(trace, wavemap):
#     use the wavemap from the HeNeAr_fit routine to determine the wavelength along the trace
#     return trace_wave
#

#########################
def biascombine(biaslist, trim=True):
    # assume biaslist is a simple text file with image names
    # e.g. ls flat.00*b.fits > bflat.lis
    files = np.loadtxt(biaslist,dtype='string')

    for i in range(0,len(files)):
        hdu_i = fits.open(files[i])

        if trim is False:
            im_i = hdu_i[0].data
        if trim is True:
            datasec = hdu_i[0].header['DATASEC'][1:-1].replace(':',',').split(',')
            d = map(float, datasec)
            im_i = hdu_i[0].data[d[2]-1:d[3],d[0]-1:d[1]]

        # create image stack
        if (i==0):
            all_data = im_i
        elif (i>0):
            all_data = np.dstack( (all_data, im_i) )
        hdu_i.close(closed=True)

    # do median across whole stack
    bias = np.median(all_data, axis=2)

    # write output to disk for later use
    hduOut = fits.PrimaryHDU(bias)
    hduOut.writeto('BIAS.fits',clobber=True)
    return bias


#########################
def flatcombine(flatlist, bias, trim=True):
    # read the bias in, BUT we don't know if it's the numpy array or file name
    if isinstance(bias, str):
        # read in file if a string
        bias_im = fits.open(bias)[0].data
    else:
        # assume is proper array from biascombine function
        bias_im = bias

    # assume flatlist is a simple text file with image names
    # e.g. ls flat.00*b.fits > bflat.lis
    files = np.loadtxt(flatlist,dtype='string')

    for i in range(0,len(files)):
        hdu_i = fits.open(files[i])
        if trim is False:
            im_i = hdu_i[0].data - bias_im
        if trim is True:
            datasec = hdu_i[0].header['DATASEC'][1:-1].replace(':',',').split(',')
            d = map(float, datasec)
            im_i = hdu_i[0].data[d[2]-1:d[3],d[0]-1:d[1]] - bias_im

        # assume a median scaling for each flat to account for possible different exposure times
        if (i==0):
            all_data = im_i / np.median(im_i)
        elif (i>0):
            all_data = np.dstack( (all_data, im_i / np.median(im_i)) )
        hdu_i.close(closed=True)

    # do median across whole stack
    flat = np.median(all_data, axis=2)
    # need other scaling?

    # write output to disk for later use
    hduOut = fits.PrimaryHDU(flat)
    hduOut.writeto('FLAT.fits',clobber=True)
    return flat


#########################
#########################
# test data setup
datafile = 'example_data/Gliese176.0052b.fits'
flatfile = 'example_data/flat.0039b.fits'
biasfile = 'example_data/bias.0014b.fits'

hdu = fits.open( datafile )

# trim the data to remove overscan region
datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
d = map(float, datasec)

img = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]]

#### test trace
# plt.figure()
# plt.imshow(np.log10(img), origin = 'lower')
# plt.plot(np.arange(img.shape[1]), spec_trace(img),'k',lw=3)
# plt.show()

bias_done = biascombine('bbias.lis')
flat_done = flatcombine('bflat.lis', bias_done)
flat_done2 = flatcombine('bflat.lis', 'BIAS.fits')

plt.figure()
plt.imshow( flat_done, origin='lower')
plt.show()

plt.figure()
plt.imshow( flat_done2, origin='lower')
plt.show()

