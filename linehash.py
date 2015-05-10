'''
How to solve the HeNeAr line identify problem for real.

Using inspiration from astrometry.net and geometric hash tables

Goal: clumsy, slow, effective
'''


import pydis
from astropy.io import fits
import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



def _MakeTris(linewave0):
    '''

    :param linewave0:
    :return:
    '''
    linewave = linewave0.copy()
    linewave.sort()

    # might want to change this to some kind of numpy array, not dict...
    d = {}
    ntri = len(linewave)-2
    for k in range(ntri):
        # the 3 lines
        l1,l2,l3 = linewave[k:k+3]
        # the 3 "sides", ratios of the line separations
        s1 = abs( (l1-l3) / (l1-l2) )
        s2 = abs( (l1-l2) / (l2-l3) )
        s3 = abs( (l1-l3) / (l2-l3) )

        sides = [s1,s2,s3]
        lines = [l1,l2,l3]
        ss = np.argsort(sides)

        d.update( {(sides[ss[0]], sides[ss[1]], sides[ss[2]]):
                       (lines[ss[0]], lines[ss[1]], lines[ss[2]])} )
    return d


def LineHash(calimage, trim=True):
    '''
    (REWORD later)
    Find emission lines, match triangles to dictionary (hash table),
    filter out junk, check wavelength order, assign wavelengths!

    Parameters
    ----------
    calimage : str
        the calibration (HeNeAr) image file name you want to solve

    '''

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

    # the red chip wavelength is backwards (DIS specific)
    clr = hdu[0].header['DETECTOR']
    if (clr.lower()=='red'):
        sign = -1.0
    else:
        sign = 1.0
    hdu.close(closed=True)

    # take a slice thru the data (+/- 10 pixels) in center row of chip
    slice = img[img.shape[0]/2-10:img.shape[0]/2+10,:].sum(axis=0)

    # use the header info to do rough solution (linear guess)
    wtemp = (np.arange(len(slice))-len(slice)/2) * disp_approx * sign + wcen_approx

    # the flux threshold to select peaks at
    flux_thresh = np.percentile(slice, 90)

    # find flux above threshold
    high = np.where( (slice >= flux_thresh) )
    # find  individual peaks (separated by > 1 pixel)
    pk = high[0][ ( (high[0][1:]-high[0][:-1]) > 1 ) ]
    # the number of pixels around the "peak" to fit over
    pwidth = 10
    # offset from start/end of array by at least same # of pixels
    pk = pk[pk > pwidth]
    pk = pk[pk < (len(slice)-pwidth)]

    # the arrays to store the estimated peaks in
    pcent_pix = np.zeros_like(pk,dtype='float')
    wcent_pix = np.zeros_like(pk,dtype='float')

    # for each peak, fit a gaussian to find robust center
    for i in range(len(pk)):
        xi = wtemp[pk[i]-pwidth:pk[i]+pwidth*2]
        yi = slice[pk[i]-pwidth:pk[i]+pwidth*2]

        pguess = (np.nanmax(yi), np.median(slice), float(np.nanargmax(yi)), 2.)
        popt,pcov = curve_fit(pydis._gaus, np.arange(len(xi),dtype='float'), yi,
                              p0=pguess)

        # the gaussian center of the line in pixel units
        pcent_pix[i] = (pk[i]-pwidth) + popt[2]
        # and the peak in wavelength units
        wcent_pix[i] = xi[np.nanargmax(yi)]

    # build observed triangles
    tri = _MakeTris(wcent_pix)

    # construct the standard object triangles (maybe could be restructured)
    std = _BuildLineDict(linelist='henear.dat')

    # now step thru each observed "tri", see if it matches any in "std"
    # within some tolerance (maybe say 5% for all 3 ratios?)




def _BuildLineDict(linelist='henear.dat'):
    '''
    Build the dictionary (hash table) of lines from the master file.

    Goal is to do this once, store it in some hard file form for users.
    Users then would only re-run this function if linelist changed, say if
    a different set of lamps were used.
    '''

    dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'

    linewave = np.loadtxt(dir + linelist, dtype='float',
                           skiprows=1, usecols=(0,), unpack=True)

    # sort the lines, just in case the file is not sorted
    linewave.sort()

    d = _MakeTris(linewave)

    # now, how to save this dict? or should we just return it?
    return d
