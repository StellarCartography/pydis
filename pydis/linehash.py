'''
How to solve the HeNeAr line identify problem for real.

Using inspiration from astrometry.net and geometric hash tables

Goal: clumsy, slow, effective
'''


import pydis
from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt

__all__ = ['autoHeNeAr']


def _MakeTris(linewave0):
    '''
    An internal function to be used with _BuildLineDict

    Takes a list of line wavelengths and creates "Tri's", or sets of 3 neighboring
    lines. Ratios of the line separations are computed, and sorted within each Tri
    to make a unique set of ratios for these lines.

    Parameters
    ----------
    linewave0 : float array
        This is the input list of lines

    Returns
    -------
    side_out, line_out
        The sides of the Tri's (line separation ratios) and actual lines
    '''

    linewave = linewave0.copy()
    linewave.sort()

    # print(linewave)
    ntri = len(linewave)-2
    k0 = 0
    for k in range(ntri):
        # the 3 lines
        l1,l2,l3 = linewave[k:k+3]
        # the 3 "sides", ratios of the line separations
        s1 = abs( (l1-l3) / (l1-l2) )
        s2 = abs( (l1-l2) / (l2-l3) )
        s3 = abs( (l1-l3) / (l2-l3) )

        sides = np.array([s1,s2,s3])
        lines = np.array([l1,l2,l3])
        ss = np.argsort(sides)

        if np.isfinite(sides).sum() > 2:
            if (k0==0):
                side_out = sides[ss]
                line_out = lines[ss]
                k0=1
            else:
                side_out = np.vstack((side_out, sides[ss]))
                line_out = np.vstack((line_out, lines[ss]))

    return side_out, line_out


def _BuildLineDict(linelist):
    '''
    Build the dictionary (sorta hash table) of lines from the master file.

    Goal is to do this once, store it in some hard file form for users.
    Users then would only re-run this function if linelist changed, say if
    a different set of lamps were used.
    '''

    linewave = np.loadtxt(linelist, dtype='float',
                          skiprows=1, usecols=(0,), unpack=True)

    # sort the lines, just in case the file is not sorted
    linewave.sort()

    sides, lines = _MakeTris(linewave)

    # now, how to save this dict? or should we just return it?
    return sides, lines


def autoHeNeAr(calimage, trim=True, maxdist=0.5, linelist='apohenear.dat',
               fmask=(0,), display=False, noflatmask=False):
    '''
    (REWORD later)
    Find emission lines, match triangles to dictionary (hash table),
    filter out junk, check wavelength order, assign wavelengths!

    Parameters
    ----------
    calimage : str
        the calibration (HeNeAr) image file name you want to solve

    '''

    # !!! this should be changed to pydis.OpenImg ?
    hdu = fits.open(calimage)
    if trim is False:
        img = hdu[0].data
    if trim is True:
        datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
        d = map(float, datasec)
        img = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]]

    # this approach will be very DIS specific
    disp_approx = np.float(hdu[0].header['DISPDW'])
    wcen_approx = np.float(hdu[0].header['DISPWC'])

    # the red chip wavelength is backwards (DIS specific)
    clr = hdu[0].header['DETECTOR']
    if (clr.lower()=='red'):
        sign = -1.0
    else:
        sign = 1.0
    hdu.close(closed=True)

    if noflatmask is True:
        ycomp = img.sum(axis=1) # compress to spatial axis only
        illum_thresh = 0.5 # value compressed data must reach to be used for flat normalization
        ok = np.where( (ycomp>= np.median(ycomp)*illum_thresh) )
        fmask = ok[0]

    slice_width = 5.
    # take a slice thru the data in center row of chip
    slice = img[img.shape[0]/2.-slice_width:img.shape[0]/2.+slice_width,:].sum(axis=0)

    # use the header info to do rough solution (linear guess)
    wtemp = (np.arange(len(slice),dtype='float') - np.float(len(slice))/2.0) * disp_approx * sign + wcen_approx

    # if display is True:
    #     print('disp_approx',disp_approx)
    #     print('wcen_approx',wcen_approx)
    #     print(wtemp)
    #     print(wtemp[0])
    #     plt.figure()
    #     plt.plot(wtemp, slice)
    #     plt.show()

    pcent_pix, wcent_pix = pydis.find_peaks(wtemp, slice, pwidth=10, pthreshold=80, minsep=2)

    # if display is True:
        # print('>>>>>')
        # print(pcent_pix)
        # print(wcent_pix)
        # print(wcent_pix[1])
        # print('<<<<<')
        # plt.figure()
        # plt.scatter(pcent_pix, wcent_pix, alpha=0.5, s=50)
        # plt.show()

    # build observed triangles from HeNeAr file, in wavelength units
    tri_keys, tri_wave = _MakeTris(wcent_pix)
    '''print'>', (np.shape(wcent_pix),np.shape(tri_keys), np.shape(tri_wave))'''

    # make the same observed tri using pixel units.
    # ** should correspond directly **
    _, tri_pix = _MakeTris(pcent_pix)
    '''print('>> ',np.shape(pcent_pix), np.shape(tri_pix))'''

    # construct the standard object triangles (maybe could be restructured)
    dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'
    std_keys, std_wave = _BuildLineDict(dir + linelist)

    # now step thru each observed "tri", see if it matches any in "std"
    # within some tolerance (maybe say 5% for all 3 ratios?)

    # for each observed tri
    for i in range(tri_keys.shape[0]):
        obs = tri_keys[i,:]
        dist = []
        # search over every library tri, find nearest (BRUTE FORCE)
        for j in range(std_keys.shape[0]):
            ref = std_keys[j,:]
            dist.append( np.sum((obs-ref)**2.)**0.5 )

        if (min(dist)<maxdist):
            indx = dist.index(min(dist))
            # replace the observed wavelengths with the catalog values
            tri_wave[i,:] = std_wave[indx,:]
        else:
            # need to do something better here too
            tri_wave[i,:] = np.array([np.nan, np.nan, np.nan])

    ok = np.where((np.isfinite(tri_wave)))

    out_wave = tri_wave[ok]
    out_pix = tri_pix[ok]

    out_wave.sort()
    out_pix.sort()

    xcent_big, ycent_big, wcent_big = pydis.line_trace(img, out_pix, out_wave,
                                                       fmask=fmask, display=False)

    wfit = pydis.lines_to_surface(img, xcent_big, ycent_big, wcent_big,
                            mode='spline')

    if display is True:
        plt.figure()
        plt.scatter(out_pix, out_wave)
        plt.plot(np.arange(len(slice)),
                 pydis.mapwavelength(np.ones_like(slice)*img.shape[0]/2.,
                                     wfit, mode='spline'))
        plt.title('autoHeNeAr Find')
        plt.xlabel('Pixel')
        plt.ylabel('Wavelength')
        plt.show()

    return wfit

if __name__ == "__main__":
    import sys
    print(sys.argv[1])
    autoHeNeAr(sys.argv[1], display=True, noflatmask=True)
