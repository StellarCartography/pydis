'''
Can we provide a hacky, simple way to solve the HeNeAr problem for just DIS?
Namely, let's try taking the known line list and doing a simple cross-correlation
over chunks of the observed HeNeAr spectrum.
'''


import pydis
from astropy.io import fits
import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def linexcorl(calimage, trim=True, linelist='apohenear.dat',
              fmask=(0,), display=False, noflatmask=False, nbins=10):


    # !!! this should be changed to pydis.OpenImg ?
    hdu = fits.open(calimage)
    if trim is False:
        img = hdu[0].data
    if trim is True:
        datasec = hdu[0].header['DATASEC'][1:-1].replace(':', ',').split(',')
        d = map(float, datasec)
        img = hdu[0].data[d[2] - 1:d[3], d[0] - 1:d[1]]

    # this approach will be very DIS specific
    disp_approx = np.float(hdu[0].header['DISPDW'])
    wcen_approx = np.float(hdu[0].header['DISPWC'])

    # the red chip wavelength is backwards (DIS specific)
    clr = hdu[0].header['DETECTOR']
    if (clr.lower() == 'red'):
        sign = -1.0
    else:
        sign = 1.0
    hdu.close(closed=True)

    if noflatmask is True:
        ycomp = img.sum(axis=1)  # compress to spatial axis only
        illum_thresh = 0.5  # value compressed data must reach to be used for flat normalization
        ok = np.where((ycomp >= np.median(ycomp) * illum_thresh))
        fmask = ok[0]

    slice_width = 5.
    # take a slice thru the data in center row of chip
    slice = img[img.shape[0] / 2. - slice_width:img.shape[0] / 2. + slice_width, :].sum(axis=0)

    # use the header info to do rough solution (linear guess)
    wtemp = (np.arange(len(slice), dtype='float') - np.float(len(slice)) / 2.0) * disp_approx * sign + wcen_approx

    pcent_pix, wcent_pix = pydis.find_peaks(wtemp, slice, pwidth=10, pthreshold=80, minsep=2)

    ### now for the XCORL part:
    # 1. make catalog line "spectrum", use double the resolution of estimated pixels (wtemp)
    #    also assume gaussians of 2 pixels stddev for each line

    # 2. chop the observed slice into segments (nbins)
    #    note: adjust these if not enough lines detected in find_peaks

    # 3. do cross correlation for each segment. Defines a linear correction between observed and catalog

    # 4. do polyfit to solution in middle of bin (or weighted middle from detected lines)
