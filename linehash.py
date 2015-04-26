'''
How to solve the HeNeAr line identify problem for real.

Using inspiration from astrometry.net and geometric hash tables

Goal: clumsy, slow, effective
'''


# import pydis
from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt


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



def _BuildLineDict(linelist='henear.dat'):
    '''
    Build the dictionary (hash table) of lines from the master file.

    Goal is to do this once, store it in some hard file form for users.
    Users then would only re-run this function if linelist changed, say if
    a different set of lamps were used. Thus, these method should be
    chemically agnostic!
    '''

    dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'

    linewave2 = np.loadtxt(dir + linelist, dtype='float',
                           skiprows=1, usecols=(0,), unpack=True)

    # go from blue to red, build huge list of triangles