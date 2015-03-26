# -*- coding: utf-8 -*-
"""
SPECTRA: A simple one dimensional spectra reduction and analysis package

Created with the Apache Point Observatory (APO) 3.5-m telescope's
Dual Imaging Spectrograph (DIS) in mind. YMMV

e.g. DIS specifics:
- have BLUE/RED channels
- hand-code in that the RED channel wavelength is backwards
- dispersion along the X, spatial along the Y axis

"""

import matplotlib
matplotlib.use('TkAgg')
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import SmoothBivariateSpline
from astropy.convolution import convolve, Box1DKernel
import scipy.signal
import datetime
import os
from matplotlib.widgets import Cursor



def _mag2flux(mag, zeropt=21.10):
    flux = 10.0**( (mag + zeropt) / (-2.5) )
    return flux

def _gaus(x,a,b,x0,sigma):
    """ Simple Gaussian function, for internal use only """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+b


def biascombine(biaslist, output='BIAS.fits', trim=True):
    """
    Combine the bias frames in to a master bias image. Currently only
    supports median combine.

    Parameters
    ----------
    biaslist : str
        Path to file containing list of bias images.
    output: str, optional
        Name of the master bias image to write. (Default is "BIAS.fits")
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is True)

    Returns
    -------
    bias : 2-d array
        The median combined master bias image
    """

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


def flatcombine(flatlist, bias, output='FLAT.fits', trim=True,
                display=False, flat_poly=5):
    """
    Combine the flat frames in to a master flat image. Subtracts the
    master bias image first from each flat image. Currently only
    supports median combining the images.

    Parameters
    ----------
    flatlist : str
        Path to file containing list of flat images.
    bias : str or 2-d array
        Either the path to the master bias image (str) or
        the output from 2-d array output from biascombine
    output: str, optional
        Name of the master flat image to write. (Default is "FLAT.fits")
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is True)
    display : bool, optional
        Set to True to show 1d flat, and final flat (Default is False)
    flat_poly : int, optional
        Polynomial order to fit 1d flat curve with. (Default is 5)

    Returns
    -------
    flat : 2-d array
        The median combined master flat
    """
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

    # do median across whole stack of flat images
    flat_stack = np.median(all_data, axis=2)

    xdata = np.arange(all_data.shape[1]) # x pixels

    # sum along spatial axis, smooth w/ 5pixel boxcar, take log of summed flux
    flat_1d = np.log10(convolve(flat_stack.sum(axis=0), Box1DKernel(5)))

    # fit log flux with polynomial
    flat_fit = np.polyfit(xdata, flat_1d, flat_poly)
    # get rid of log
    flat_curve = 10.0**np.polyval(flat_fit, xdata)

    if display is True:
        plt.figure()
        plt.plot(10.0**flat_1d)
        plt.plot(xdata, flat_curve,'r')
        plt.show()

    # divide median stacked flat by this RESPONSE curve
    flat = np.zeros_like(flat_stack)
    for i in range(flat_stack.shape[0]):
        flat[i,:] = flat_stack[i,:] / flat_curve

    # normalize flat
    flat = flat / np.median(flat[ok,:])

    if display is True:
        plt.figure()
        plt.imshow(flat, origin='lower',aspect='auto')
        plt.show()

    # write output to disk for later use
    hduOut = fits.PrimaryHDU(flat)
    hduOut.writeto(output, clobber=True)

    return flat ,ok[0]


def ap_trace(img, fmask=(1,), nsteps=50):
    """
    Trace the spectrum aperture in an image

    Assumes wavelength axis is along the X, spatial axis along the Y.
    Chops image up in bix along the wavelength direction, fits a Gaussian
    within each bin to determine the spatial center of the trace. Finally,
    draws a cubic spline through the bins to up-sample the trace.

    Parameters
    ----------
    img : 2d numpy array
        This is the image, stored as a normal numpy array. Can be read in
        using astropy.io.fits like so:
        >>> hdu = fits.open('file.fits')
        >>> img = hdu[0].data
    nsteps : int, optional
        Keyword, number of bins in X direction to chop image into. Use
        fewer bins if ap_trace is having difficulty, such as with faint
        targets (default is 50, minimum is 4)
    fmask : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.

    Returns
    -------
    my : array
        The spatial (Y) positions of the trace, interpolated over the
        entire wavelength (X) axis
    """
    print('Tracing Aperture using nsteps='+str(nsteps))
    # the valid y-range of the chip
    if (len(fmask)>1):
        ydata = np.arange(img.shape[0])[fmask]
    else:
        ydata = np.arange(img.shape[0])

    # need at least 4 samples along the trace. sometimes can get away with very few
    if (nsteps<4):
        nsteps = 4

    # median smooth to crudely remove cosmic rays
    img_sm = scipy.signal.medfilt2d(img, kernel_size=(5,5))

    #--- find the overall max row, and width
    ztot = img_sm.sum(axis=1)[ydata]
    yi = np.arange(img.shape[0])[ydata]
    peak_y = yi[np.nanargmax(ztot)]
    popt_tot,pcov = curve_fit(_gaus, yi, ztot,
                          p0=[np.nanmax(ztot), np.median(ztot), peak_y, 2.])

    # define the bin edges
    xbins = np.linspace(0, img.shape[1], nsteps)
    ybins = np.zeros_like(xbins)

    for i in range(0,len(xbins)-1):
        #-- simply use the max w/i each window
        #ybins[i] = np.argmax(img_sm[:,xbins[i]:xbins[i+1]].sum(axis=1))
        
        #-- fit gaussian w/i each window
        zi = img_sm[ydata, xbins[i]:xbins[i+1]].sum(axis=1)
        pguess = [np.nanmax(zi), np.median(zi), yi[np.nanargmax(zi)], 2.]
        popt,pcov = curve_fit(_gaus, yi, zi, p0=pguess)

        # if gaussian fits off chip, then use chip-integrated answer
        if (popt[2] <= min(ydata)+25) or (popt[2] >= max(ydata)-25):
            ybins[i] = popt_tot[2]
        else:
            ybins[i] = popt[2]

    # recenter the bin positions, trim the unused bin off in Y
    mxbins = (xbins[:-1]+xbins[1:]) / 2.
    mybins = ybins[:-1]

    # run a cubic spline thru the bins
    ap_spl = UnivariateSpline(mxbins, mybins, ext=0, k=3, s=0)

    # interpolate the spline to 1 position per column
    mx = np.arange(0,img.shape[1])
    my = ap_spl(mx)
    return my



def ap_extract(img, trace, apwidth=5.0):
    """
    Extract the spectrum using the trace. Simply add up all the flux
    around the aperture within a specified +/- width.

    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An important simplification.

    Parameters
    ----------
    img : 2d numpy array
        This is the image, stored as a normal numpy array. Can be read in
        using astropy.io.fits like so:
        >>> hdu = fits.open('file.fits')
        >>> img = hdu[0].data
    trace : 1-d array
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from ap_trace
    apwidth : int, optional
        The width along the Y axis of the trace to extract. Note: a fixed
        width is used along the whole trace. (default is 5 pixels)

    Returns
    -------
    onedspec : array
        The summed flux at each column about the trace. Note: is not
        sky subtracted!
    """

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

        # simply add up the total flux around the trace +/- width
        onedspec[i] = img[trace[i]-widthdn:trace[i]+widthup, i].sum()
    return onedspec



def sky_fit(img, trace, apwidth=5, skysep=25, skywidth=75, skydeg=2):
    """
    Fits a polynomial to the sky at each column

    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An important simplification.

    Parameters
    ----------
    img : 2d numpy array
        This is the image, stored as a normal numpy array. Can be read in
        using astropy.io.fits like so:
        >>> hdu = fits.open('file.fits')
        >>> img = hdu[0].data
    trace : 1-d array
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from ap_trace
    apwidth : int, optional
        The width along the Y axis of the trace to extract. Note: a fixed
        width is used along the whole trace. (default is 5 pixels)
    skysep : int, optional
        The separation in pixels from the aperture to the sky window.
        (Default is 25)
    skywidth : int, optional
        The width in pixels of the sky windows on either side of the
        aperture. (Default is 75)
    skydeg : int, optional
        The polynomial order to fit between the sky windows.
        (Default is 2)

    Returns
    -------
    skysubflux : 1d array
        The integrated sky values along each column, suitable for
        subtracting from the output of ap_extract
    """

    skysubflux = np.zeros_like(trace)
    for i in range(0,len(trace)):
        itrace = int(trace[i])
        y = np.append(np.arange(itrace-apwidth-skysep-skywidth, itrace-apwidth-skysep),
                      np.arange(itrace+apwidth+skysep, itrace+apwidth+skysep+skywidth))

        z = img[y,i]
        # fit a polynomial to the sky in this column
        pfit = np.polyfit(y,z,skydeg)
        # define the aperture in this column
        ap = np.arange(trace[i]-apwidth,trace[i]+apwidth)
        # evaluate the polynomial across the aperture, and sum
        skysubflux[i] = np.sum(np.polyval(pfit, ap))
    return skysubflux



def HeNeAr_fit(calimage, linelist='', interac=True,
               trim=True, fmask=(1,), display=True,
               tol=10,fit_order=2):
    """
    Determine the wavelength solution to be used for the science images.
    Can be done either automatically (buyer beware) or manually. Both the
    manual and auto modes use a "slice" through the chip center to learn
    the wavelengths of specific HeNeAr lines. Emulates the IDENTIFY
    function in IRAF.

    If the automatic mode is selected (interac=False), program tries to
    first find significant peaks in the "slice", then uses a brute-force
    guess scheme based on the grating information in the header. While
    easy, your mileage may vary with this method.

    If the interactive mode is selected (interac=True), you click on
    features in the "slice" and identify their wavelengths.

    Parameters
    ----------
    calimage : str
        Path to the HeNeAr calibration image
    linelist : str, optional
        Path to the linelist file to use. Only needed if using the
        automatic mode.
    interac : bool, optional
        Should the HeNeAr identification be done interactively (manually)?
        (Default is True)
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is True)
    fmask : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.
    display : bool, optional
    tol : int, optional
        When in automatic mode, the tolerance in pixel units between
        linelist entries and estimated wavelengths for the first few
        lines matched... use carefully. (Default is 10)
    fit_order : int, optional
        The polynomial order to use to interpolate between identified
        peaks in the HeNeAr (Default is 2)

    Returns
    -------
    wfit : bivariate spline object
        The wavelength evaluated at every pixel
    """

    print('Running HeNeAr_fit function.')

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

    ######   IDENTIFY   (auto and interac modes)
    if interac is False:
        # find the linelist of choice
        if (len(linelist)==0):
            dir = os.path.dirname(os.path.realpath(__file__))
            linelist = dir + '/resources/dishigh_linelist.txt'

        # import the linelist
        linewave = np.loadtxt(linelist,dtype='float',skiprows=1,usecols=(0,),unpack=True)

        # sort data, cut top x% of flux data as peak threshold
        flux_thresh = np.percentile(slice, 97)

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

        pcent_pix = np.zeros_like(pk,dtype='float')
        wcent_pix = np.zeros_like(pk,dtype='float') # wtemp[pk]
        # for each peak, fit a gaussian to find center
        for i in range(len(pk)):
            xi = wtemp[pk[i]-pwidth:pk[i]+pwidth*2]
            yi = slice[pk[i]-pwidth:pk[i]+pwidth*2]

            pguess = (np.nanmax(yi), np.median(slice), float(np.nanargmax(yi)), 2.)
            popt,pcov = curve_fit(_gaus, np.arange(len(xi),dtype='float'), yi,
                                  p0=pguess)

            # the gaussian center of the line in pixel units
            pcent_pix[i] = (pk[i]-pwidth) + popt[2]
            # and the peak in wavelength units
            wcent_pix[i] = xi[np.nanargmax(yi)]

            if display is True:
                plt.scatter(wtemp[pk][i], slice[pk][i], marker='o')
                plt.plot(xi, _gaus(np.arange(len(xi)),*popt), 'r')
        if display is True:
            plt.xlabel('approx. wavelength')
            plt.ylabel('flux')
            #plt.show()

        if display is True:
            plt.scatter(linewave,np.ones_like(linewave)*np.nanmax(slice),marker='o',c='blue')
            plt.show()

    #   loop thru each peak, from center outwards. a greedy solution
    #   find nearest list line. if not line within tolerance, then skip peak
        pcent = []
        wcent = []

        # find center-most lines, sort by dist from center pixels
        ss = np.argsort(np.abs(wcent_pix-wcen_approx))

        #coeff = [0.0, 0.0, disp_approx*sign, wcen_approx]
        coeff = np.append(np.zeros(fit_order-1),(disp_approx*sign, wcen_approx))

        for i in range(len(pcent_pix)):
            xx = pcent_pix-len(slice)/2
            #wcent_pix = coeff[3] + xx * coeff[2] + coeff[1] * (xx*xx) + coeff[0] * (xx*xx*xx)
            wcent_pix = np.polyval(coeff, xx)

            if display is True:
                plt.figure()
                plt.plot(wtemp, slice, 'b')
                plt.scatter(linewave,np.ones_like(linewave)*np.nanmax(slice),marker='o',c='cyan')
                plt.scatter(wcent_pix,np.ones_like(wcent_pix)*np.nanmax(slice)/2.,marker='*',c='green')
                plt.scatter(wcent_pix[ss[i]],np.nanmax(slice)/2., marker='o',c='orange')

            # if there is a match w/i the linear tolerance
            if (min((np.abs(wcent_pix[ss][i] - linewave))) < tol):
                # add corresponding pixel and *actual* wavelength to output vectors
                pcent = np.append(pcent,pcent_pix[ss[i]])
                wcent = np.append(wcent, linewave[np.argmin(np.abs(wcent_pix[ss[i]] - linewave))] )

                if display is True:
                    plt.scatter(wcent,np.ones_like(wcent)*np.nanmax(slice),marker='o',c='red')

                if (len(pcent)>fit_order):
                    coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)

            if display is True:
                plt.xlim((min(wtemp),max(wtemp)))
                plt.show()

        # the end result is the vector "coeff" has the wavelength solution for "slice"
        # update the "wtemp" vector that goes with "slice" (fluxes)
        wtemp = np.polyval(coeff, (np.arange(len(slice))-len(slice)/2))

    elif interac is True:

        print('')
        print('Using INTERACTIVE HeNeAr_fit mode:')
        print('1) Click on HeNeAr lines in plot window')
        print('2) Enter corresponding wavelength in terminal and press <return>')
        print('   If mis-click or unsure, just press leave blank and press <return>')
        print('3) Close plot window when finished')

        xraw = np.arange(len(slice))
        class InteracWave:
            # http://stackoverflow.com/questions/21688420/callbacks-for-graphical-mouse-input-how-to-refresh-graphics-how-to-tell-matpl
            def __init__(self):
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111)
                self.ax.plot(wtemp, slice, 'b')
                plt.xlabel('Wavelength')
                plt.ylabel('Counts')

                self.pcent = []
                self.wcent = []

                self.cursor = Cursor(self.ax, useblit=False,horizOn=False,
                                     color='red', linewidth=1 )
                self.connect = self.fig.canvas.mpl_connect
                self.disconnect = self.fig.canvas.mpl_disconnect
                self.clickCid = self.connect("button_press_event",self.OnClick)

            def OnClick(self, event):
                # only do stuff if toolbar not being used
                # NOTE: this subject to change API, so if breaks, this probably why
                # http://stackoverflow.com/questions/20711148/ignore-matplotlib-cursor-widget-when-toolbar-widget-selected
                if self.fig.canvas.manager.toolbar._active is None:
                    ix = event.xdata
                    if (ix is not None) and (ix > np.nanmin(slice)) and (ix < np.nanmax(slice)):
                        # disable button event connection
                        self.disconnect(self.clickCid)

                        # disconnect cursor, and remove from plot
                        self.cursor.disconnect_events()
                        self.cursor._update()

                        nearby = np.where((wtemp > ix-10*disp_approx) &
                                          (wtemp < ix+10*disp_approx) )

                        if (len(nearby[0]) > 4):
                            imax = np.nanargmax(slice[nearby])

                            pguess = (np.nanmax(slice[nearby]), np.median(slice), xraw[nearby][imax], 2.)
                            try:
                                popt,pcov = curve_fit(_gaus, xraw[nearby], slice[nearby], p0=pguess)
                                self.ax.plot(wtemp[int(popt[2])], popt[0], 'r|')
                            except ValueError:
                                print('> WARNING: Bad data near this click, cannot centroid line with Gaussian. I suggest you skip this one')
                                popt = pguess
                            except RuntimeError:
                                print('> WARNING: Gaussian centroid on line could not converge. I suggest you skip this one')
                                popt = pguess

                            # using raw_input sucks b/c doesn't raise terminal, but works for now
                            try:
                                number=float(raw_input('> Enter Wavelength: '))
                                self.pcent.append(popt[2])
                                self.wcent.append(number)
                                self.ax.plot(wtemp[int(popt[2])], popt[0], 'ro')
                                print('  Saving '+str(number))
                            except ValueError:
                                print "> Warning: Not a valid wavelength float!"

                        else:
                            print('> Error: No valid data near click!')

                        # reconnect to cursor and button event
                        self.clickCid = self.connect("button_press_event",self.OnClick)
                        self.cursor = Cursor(self.ax, useblit=False,horizOn=False,
                                         color='red', linewidth=1 )
                else:
                    pass

        test = InteracWave()
        plt.show()

        pcent = np.array(test.pcent,dtype='float')
        wcent = np.array(test.wcent, dtype='float')
        print(pcent)
        print(wcent)

        # fit polynomial thru the peak wavelengths
        coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)
        wtemp = np.polyval(coeff, (np.arange(len(slice))-len(slice)/2))

    #-- trace the peaks vertically
    # how far can the trace be bent, i.e. how big a window to fit over?
    maxbend = 10 # pixels (+/-)

    # 3d positions of the peaks: (x,y) and wavelength
    xcent_big = []
    ycent_big = []
    wcent_big = []

    # the valid y-range of the chip
    if (len(fmask)>1):
        ydata = np.arange(img.shape[0])[fmask]
    else:
        ydata = np.arange(img.shape[0])

    # split the chip in to 2 parts, above and below the center
    ydata1 = ydata[np.where((ydata>=img.shape[0]/2))]
    ydata2 = ydata[np.where((ydata<img.shape[0]/2))][::-1]

    img_med = np.median(img)
    # loop over every HeNeAr peak that had a good fit

    for i in range(len(pcent)):
        xline = np.arange(int(pcent[i])-maxbend,int(pcent[i])+maxbend)

        # above center line (where fit was done)
        for j in ydata1:
            yline = img[j, int(pcent[i])-maxbend:int(pcent[i])+maxbend]
            # fit gaussian, assume center at 0, width of 2
            if j==ydata1[0]:
                cguess = pcent[i] # xline[np.argmax(yline)]

            pguess = [np.nanmax(yline),img_med,cguess,2.]
            popt,pcov = curve_fit(_gaus, xline, yline, p0=pguess)
            cguess = popt[2] # update center pixel

            xcent_big = np.append(xcent_big, popt[2])
            ycent_big = np.append(ycent_big, j)
            wcent_big = np.append(wcent_big, wcent[i])
        # below center line, from middle down
        for j in ydata2:
            yline = img[j, int(pcent[i])-maxbend:int(pcent[i])+maxbend]
            # fit gaussian, assume center at 0, width of 2
            if j==ydata1[0]:
                cguess = pcent[i] # xline[np.argmax(yline)]

            pguess = [np.nanmax(yline),img_med,cguess,2.]
            popt,pcov = curve_fit(_gaus, xline, yline, p0=pguess)
            cguess = popt[2] # update center pixel

            xcent_big = np.append(xcent_big, popt[2])
            ycent_big = np.append(ycent_big, j)
            wcent_big = np.append(wcent_big, wcent[i])

    if display is True:
        plt.figure()
        plt.imshow(np.log10(img), origin = 'lower',aspect='auto',cmap=cm.Greys_r)
        plt.colorbar()
        plt.scatter(xcent_big,ycent_big,marker='|',c='r')
        plt.show()

    #-- now the big show!
    #  fit the wavelength solution for the entire chip w/ a 2d spline
    xfitd = 3 # the spline dimension in the wavelength space
    print('Fitting Spline!')
    wfit = SmoothBivariateSpline(xcent_big,ycent_big,wcent_big,kx=xfitd,ky=3,
                                 bbox=[0,img.shape[1],0,img.shape[0]] )
    ## using 2d polyfit
    # wfit = polyfit2d(xcent_big, ycent_big, wcent_big, order=3)
    return wfit


def mapwavelength(trace, wavemap):
    """
    Compute the wavelength along the center of the trace, to be run after
    the HeNeAr_fit routine.

    Parameters
    ----------
    trace : 1-d array
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from ap_trace
    wavemap : bivariate spline object
        The wavelength evaluated at every pixel, output from HeNeAr_fit

    Returns
    -------
    trace_wave : 1d array
        The wavelength vector evaluated at each position along the trace
    """
    # use the wavemap from the HeNeAr_fit routine to determine the wavelength along the trace
    trace_wave = wavemap.ev(np.arange(len(trace)), trace)

    ## using 2d polyfit
    # trace_wave = polyval2d(np.arange(len(trace)), trace, wavemap)
    return trace_wave



def normalize(wave, flux, spline=False, poly=True, order=3, interac=True):
    # not yet
    if (poly is False) and (spline is False):
        poly=True

    if (poly is True):
        print("yes")

    return


def AirmassCor(obj_wave, obj_flux, airmass):
    # read in the airmass curve for APO
    dir = os.path.dirname(os.path.realpath(__file__))
    air_wave, air_trans = np.loadtxt(dir+'/resources/apoextinct.dat',
                                     unpack=True,skiprows=2)

    #this isnt quite right...
    airmass_ext = np.interp(obj_wave, air_wave, air_trans) / airmass
    return airmass_ext * obj_flux


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

    #-- should we down-sample the template?
    # std_wave = np.arange(np.nanmin(obj_wave), np.nanmax(obj_wave),
    #                      np.mean(np.abs(std_wave0[1:]-std_wave0[:-1])))
    # std_flux = np.interp(std_wave, std_wave0, std_flux0)

    #-- don't down-sample the template
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

    # the width of each pixel (in angstroms)
    dw_tmp = obj_wave[1:]-obj_wave[:-1]
    dw = np.abs(np.append(dw_tmp, dw_tmp[-1]))

    plt.figure()
    plt.plot(obj_wave_ds, ratio, 'ko')
    plt.plot(obj_wave, ratio_spl(obj_wave),'r')
    plt.ylabel('(erg/s/cm2/A) / (counts/s)')
    plt.show()

    # this still isnt quite the sensfunc we want...
    sens = ratio_spl(obj_wave)


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

#########################
def autoreduce(speclist, flatlist, biaslist, HeNeAr_file,
               trace1=False, ntracesteps=25,
               apwidth=3,skysep=25,skywidth=75, HeNeAr_interac=False,
               HeNeAr_tol=20, HeNeAr_order=2, displayHeNeAr=False,
               trim=True, write_reduced=True, display=True):
    """
    A wrapper routine to carry out the full steps of the spectral
    reduction and calibration. Steps include:
    1) combines bias and flat images
    2) maps wavelength in the HeNeAr image
    3) perform simple image reduction: Data = (Raw - Bias)/Flat
    4) trace spectral aperture
    5) extract spectrum
    6) measure sky along extracted spectrum
    7) write output files

    Parameters
    ----------
    speclist : str
        Path to file containing list of science images.
    flatlist : str
        Path to file containing list of flat images.
    biaslist : str
        Path to file containing list of bias images.
    HeNeAr_file : str
        Path to the HeNeAr calibration image
    trace1 : bool, optional
        use trace1=True if only perform aperture trace on first object in
        speclist. Useful if e.g. science targets are faint, and first
        object is a bright standard star. Note: assumes star placed at
        same position in spatial direction. (Default is False)
    ntracesteps : int, optional
        Number of bins in X direction to chop image into. Use
        fewer bins if ap_trace is having difficulty, such as with faint
        targets (default here is 25, minimum is 4)
    apwidth : int, optional
        The width along the Y axis of the trace to extract. Note: a fixed
        width is used along the whole trace. (default here is 3 pixels)
    skysep : int, optional
        The separation in pixels from the aperture to the sky window.
        (Default is 25)
    skywidth : int, optional
        The width in pixels of the sky windows on either side of the
        aperture. (Default is 75)
    HeNeAr_interac : bool, optional
        Should the HeNeAr identification be done interactively (manually)?
        (Default here is False)
    HeNeAr_tol : int, optional
        When in automatic mode, the tolerance in pixel units between
        linelist entries and estimated wavelengths for the first few
        lines matched... use carefully. (Default here is 20)
    HeNeAr_order : int, optional
        The polynomial order to use to interpolate between identified
        peaks in the HeNeAr (Default is 2)
    displayHeNeAr : bool, optional
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is True)
    write_reduced : bool, optional
        Set to True to write output files, including the .spec file with
        columns (wavelength, flux); the .trace file with columns
        (X pixel number, Y pixel of trace); .log file with record of
        settings used in this routine for reduction. (Default is True)
    display : bool, optional
        Set to True to display intermediate steps along the way.
        (Default is True)

    """

    # assume specfile is a list of file names of object
    bias = biascombine(biaslist, trim=trim)
    flat,fmask_out = flatcombine(flatlist, bias, trim=trim)

    # do the HeNeAr mapping first, must apply to all science frames
    wfit = HeNeAr_fit(HeNeAr_file, trim=trim, fmask=fmask_out, interac=HeNeAr_interac,
                      display=displayHeNeAr, tol=HeNeAr_tol, fit_order=HeNeAr_order)

    # read in the list of target spectra
    specfile = np.loadtxt(speclist,dtype='string')

    for i in range(len(specfile)):
        spec = specfile[i]

        hdu = fits.open(spec)
        if trim is True:
            datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
            d = map(float, datasec)
            raw = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]]
        else:
            raw = hdu[0].data

        exptime = hdu[0].header['EXPTIME']
        hdu.close(closed=True)

        # remove bias and flat
        data = (raw - bias) / flat

        if display is True:
            plt.figure()
            plt.imshow(np.log10(data), origin = 'lower',aspect='auto',cmap=cm.Greys_r)
            plt.title(spec+' (flat and bias corrected)')
            plt.show()

        # with reduced data, trace the aperture
        if (i==0) or (trace1 is False):
            trace = ap_trace(data,fmask=fmask_out, nsteps=ntracesteps)

        # extract the spectrum
        ext_spec = ap_extract(data, trace, apwidth=apwidth)

        # measure sky values along trace
        sky = sky_fit(data, trace, apwidth=apwidth,skysep=skysep,skywidth=skywidth)

        xbins = np.arange(data.shape[1])
        if display is True:
            plt.figure()
            plt.imshow(np.log10(data), origin = 'lower',aspect='auto',cmap=cm.Greys_r)
            plt.colorbar()
            plt.plot(xbins, trace,'b',lw=1)
            plt.plot(xbins, trace-apwidth,'r',lw=1)
            plt.plot(xbins, trace+apwidth,'r',lw=1)
            plt.plot(xbins, trace-apwidth-skysep,'g',lw=1)
            plt.plot(xbins, trace-apwidth-skysep-skywidth,'g',lw=1)
            plt.plot(xbins, trace+apwidth+skysep,'g',lw=1)
            plt.plot(xbins, trace+apwidth+skysep+skywidth,'g',lw=1)

            plt.title('(with trace, aperture, and sky regions)')
            plt.show()

        # write output file for extracted spectrum
        # if write_reduced is True:
        #     np.savetxt(spec+'.apextract',ext_spec-sky)

        wfinal = mapwavelength(trace, wfit)

        ffinal = (ext_spec - sky) / exptime
        if write_reduced is True:
            # write file with the trace (y positions)
            tout = open(spec+'.trace','w')
            tout.write('#  This file contains the x,y coordinates of the trace \n')
            for k in range(len(trace)):
                tout.write(str(k)+', '+str(trace[k]) + '\n')
            tout.close()

            # write the final spectrum out
            fout = open(spec+'.spec','w')
            fout.write('#  This file contains the final extracted wavelength,counts data \n')
            for k in range(len(wfinal)):
                fout.write(str(wfinal[k]) + ', ' + str(ffinal[k]) + '\n')
            fout.close()

            now = datetime.datetime.now()

            lout = open(spec+'.log','w')
            lout.write('#  This file contains data on the reduction parameters \n'+
                       '#  used for '+spec+'\n')
            lout.write('DATE-REDUCED = '+str(now)+'\n')
            lout.write('HeNeAr_tol   = '+str(HeNeAr_tol)+'\n')
            lout.write('HeNeAr_order = '+str(HeNeAr_order)+'\n')
            lout.write('trace1       = '+str(trace1)+'\n')
            lout.write('ntracesteps  = '+str(ntracesteps)+'\n')
            lout.write('apwidth      = '+str(apwidth)+'\n')
            lout.write('skysep       = '+str(skysep)+'\n')
            lout.write('skywidth     = '+str(skywidth)+'\n')
            lout.write('trim         = '+str(trim)+'\n')
            lout.close()


        # the final figure to plot
        plt.figure()
        plt.plot(wfinal, ffinal)
        plt.xlabel('Wavelength')
        plt.ylabel('Counts / sec')
        plt.title(spec)
        #plot within percentile limits
        plt.ylim( (np.percentile(ffinal,2),
                   np.percentile(ffinal,98)) )
        plt.show()

    return

