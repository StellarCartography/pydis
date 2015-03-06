# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:40:42 2015

@author: jradavenport, much help by jruan

Try to make a simple one dimensional spectra reduction package

Some ideas from Python4Astronomers tutorial:
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
def ap_trace(img, nsteps=100):
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

    # define the bin edges
    xbins = np.linspace(0, img.shape[1], nsteps)
    ybins = np.zeros_like(xbins)
    for i in range(0,len(xbins)-1):
        #-- simply use the max w/i each window
        #ybins[i] = np.argmax(img_sm[:,xbins[i]:xbins[i+1]].sum(axis=1))

        #-- fit gaussian w/i each window
        zi = img_sm[:,xbins[i]:xbins[i+1]].sum(axis=1)
        yi = np.arange(len(zi))
        popt,pcov = curve_fit(gaus, yi, zi, p0=[np.amax(zi), np.median(zi), np.argmax(zi), 2.])
        ybins[i] = popt[2]

    # recenter the bin positions, trim the unused bin off in Y
    mxbins = (xbins[:-1] + xbins[1:])/2.
    mybins = ybins[:-1]

    # run a cubic spline thru the bins
    ap_spl = UnivariateSpline(mxbins, mybins, ext=0, k=3)

    # interpolate the spline to 1 position per column
    mx = np.arange(0,img.shape[1])
    my = ap_spl(mx)
    return my

#########################
def ap_extract(img, trace, apwidth=5.0):
    # simply add up the total flux around the trace +/- width
    onedspec = np.zeros_like(trace)
    for i in range(0,len(trace)):
        # juuuust in case the trace gets too close to the edge
        # (shouldn't be common)
        widthup = apwidth
        widthdn = apwidth
        if (trace[i]+widthup > img.shape[0]):
            widthup = img.shape[0]-trace[i] - 1
        if (trace[i]-widthdn < 0):
            widthdn = trace[i] - 1

        onedspec[i] = img[trace[i]-widthdn:trace[i]+widthup, i].sum()
    return onedspec


#########################
def sky_fit(img, trace, apwidth=5, skysep=25, skywidth=75, skydeg=2):
    # do simple parabola fit at each pixel
    # (assume wavelength axis is perfectly vertical)
    # return 1-d sky values
    skysubflux = np.zeros_like(trace)
    for i in range(0,len(trace)):
        itrace = int(trace[i])
        y = np.append(np.arange(itrace-skysep-skywidth, itrace-skysep),
                      np.arange(itrace+skysep, itrace+skysep+skywidth))
        z = img[y,i]
        # fit a polynomial to the sky in this column
        pfit = np.polyfit(y,z,skydeg)
        # define the aperture in this column
        ap = np.arange(trace[i]-apwidth,trace[i]+apwidth)
        # evaluate the polynomial across the aperture, and sum
        skysubflux[i] = np.sum(np.polyval(pfit, ap))
    return skysubflux


##########################
def HeNeAr_fit(calimage, linelist=1, trim=True, display=True):
    hdu = fits.open(calimage)
    if trim is False:
        img = hdu[0].data
    if trim is True:
        datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
        d = map(float, datasec)
        img = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]]

    # this approach will be very DIS specific
    disp_approx = hdu[0].header['DISPDW']
    wcen_approx = hdu[0].header['DISPWC']

    # the red chip wavelength is backwards
    clr = hdu[0].header['DETECTOR']
    if (clr.lower()=='red'):
        sign = -1.0
    else:
        sign = 1.0

    # take a slice thru the data (+/- 10 pixels)
    slice = img[img.shape[0]/2-10:img.shape[0]/2+10,:].sum(axis=0)

    # use the header info to do rough solution
    wtemp = (np.arange(len(slice))-len(slice)/2) * disp_approx * sign + wcen_approx

    # sort data, cut top 5% of flux data as peak threshold
    tmp = slice.copy()
    tmp.sort()
    flux_thresh = tmp[int(len(tmp)*0.97)]

    # find flux above threshold
    high = np.where( (slice >= flux_thresh) )

    # find  individual peaks (separated by > 1 pixel)
    pk = high[0][ ( (high[0][1:]-high[0][:-1]) > 1 ) ]

    # the number of pixels around the "peak" to fit over
    pwidth = 10
    # offset from start/end of array by at least same # of pixels
    pk = pk[pk > pwidth]
    pk = pk[pk < (len(slice)-pwidth)]

    if display is True:
        plt.figure()
        plt.plot(wtemp, slice, 'b')
        plt.plot(wtemp, np.ones_like(slice)*np.median(slice))
        plt.plot(wtemp, np.ones_like(slice) * flux_thresh)

    pcent_pix = np.zeros_like(pk)
    wcent_pix = np.zeros_like(pk)
    # for each peak, fit a gaussian to find center
    for i in range(len(pk)):
        xi = wtemp[pk[i]-pwidth:pk[i]+pwidth*1]
        yi = slice[pk[i]-pwidth:pk[i]+pwidth*1]

        popt,pcov = curve_fit(gaus, xi, yi, p0=(np.amax(yi), np.median(slice), xi[np.argmax(yi)], 2.))

        pcent_pix[i] = popt[2]

        if display is True:
            plt.scatter(wtemp[pk], slice[pk], marker='o')
            plt.plot(xi, gaus(xi,*popt), 'r')
    if display is True:
        plt.xlabel('approx. wavelength')
        plt.ylabel('flux')
        plt.show()



    # then match to best guess from linelist


#     trace the peaks vertically, allow curve, spline as before
#
#     # scipy.interpolate.SmoothBivariateSpline
#     treat the wavelenth solution as a surface

    hdu.close(closed=True)
    return #wavemap

##########################
# def mapwavelength(trace, wavemap):
#     use the wavemap from the HeNeAr_fit routine to determine the wavelength along the trace
#     return trace_wave


#########################
def biascombine(biaslist, output='BIAS.fits', trim=True):
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
    hduOut.writeto(output, clobber=True)
    return bias

#########################
def flatcombine(flatlist, bias, output='FLAT.fits', trim=True):
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

        # check for bad regions (not illuminated) in the spatial direction
        ycomp = im_i.sum(axis=1) # compress to y-axis only
        illum_thresh = 0.8 # value compressed data must reach to be used for flat normalization
        ok = np.where( (ycomp>= np.median(ycomp)*illum_thresh) )

        # assume a median scaling for each flat to account for possible different exposure times
        if (i==0):
            all_data = im_i / np.median(im_i[ok,:])
        elif (i>0):
            all_data = np.dstack( (all_data, im_i / np.median(im_i[ok,:])) )
        hdu_i.close(closed=True)

    # do median across whole stack
    flat = np.median(all_data, axis=2)
    # need other scaling?

    # write output to disk for later use
    hduOut = fits.PrimaryHDU(flat)
    hduOut.writeto(output, clobber=True)
    return flat #,ok
# i want to return the flat "OK" mask to use later,
# but only optionally.... need to make a flat class. later...


#########################
def autoreduce(speclist, flatlist, biaslist, trim=True, write_reduced=True, display=True):
    # assume specfile is a list of file names of object
    bias = biascombine(biaslist, trim = True)
    flat = flatcombine(flatlist, bias, trim=True)

    specfile = np.loadtxt(speclist,dtype='string')

    for spec in specfile:
        hdu = fits.open(spec)
        if trim is False:
            raw = hdu[0].data
        if trim is True:
            datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
            d = map(float, datasec)
            raw = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]]
        hdu.close(closed=True)

        # remove bias and flat
        data = (raw - bias) / flat

        # with reduced data, trace the aperture
        trace = ap_trace(data, nsteps=50)

        # extract the spectrum
        ext_spec = ap_extract(data, trace, apwidth=3)

        # measure sky values along trace
        sky = sky_fit(data, trace, apwidth=3)

        if display is True:
            plt.figure()
            plt.imshow(np.log10(data), origin = 'lower',aspect='auto')
            plt.colorbar()
            plt.plot(np.arange(data.shape[1]), trace,'k',lw=1)
            plt.plot(np.arange(data.shape[1]), trace-3.,'r',lw=1)
            plt.plot(np.arange(data.shape[1]), trace+3.,'r',lw=1)
            plt.show()

        # write output file for extracted spectrum
        if write_reduced is True:
            np.savetxt(spec+'.apextract',ext_spec-sky)

    return #ext_spec, sky, data, raw, bias, flat, trace



#########################
#  Test Data / Examples
#########################
datafile = 'example_data/Gliese176.0052b.fits'
flatfile = 'example_data/flat.0039b.fits'
biasfile = 'example_data/bias.0014b.fits'

hdu = fits.open( datafile )

# trim the data to remove overscan region
datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
d = map(float, datasec)

img = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]]

#autoreduce('robj.lis','rflat.lis', 'rbias.lis', trim=True, display=False)

HeNeAr_fit('example_data/HeNeAr.0028b.fits')
HeNeAr_fit('example_data/HeNeAr.0027r.fits')