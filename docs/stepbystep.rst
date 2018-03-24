.. include:: references.txt

.. _stepbystep:

***************
Reduction Guide
***************

Here is a step-by-step guide to reducing some DIS data from start to finish
with *pyDIS*. Note, this entire process is reproduced in the
`autoreduce <autoreduce>`_ helper function. In this simple example we'll assume
you have all the needed calibration files in the working directory, that you are
on a linux/mac for a couple shell commands, and that there is only 1 science
frame to analyze. We will only do the RED chip in this case. The procedure is
identical for the BLUE chip, but due to different Signal/Noise or color of the
source some parameters may need tweaking.


Organizing the data to reduce
+++++++++++++++++++++++++++++

First we need to create some lists to group the calibration files (flats, biases).
Note, darks are currently not supported explicitly. In your terminal you might say::

    ls flat.0*r.fits > rflat.lis
    ls bias.0*r.fits > rbias.lis


Make master calibration files
+++++++++++++++++++++++++++++

We'll switch to Python from here on. You might want to save these commands in a
script that you could run again. As in IRAF, we must combine the biases and
flats that will be applied to the science frame:

.. code-block:: python

    import pydis
    bias = pydis.biascombine('rbias.lis', trim=True)
    flat, fmask_out = pydis.flatcombine('rflat.lis', bias, trim=True,
                                          mode='spline', response=True, display=True)

Now there should be newly created files named **FLAT.fits** and **BIAS.fits**
in your directory. These default filenames can be changed for both functions
using the `output='FILE.fits'` keyword. Note all the extra keyword arguments
(aka "kwargs") for `flatcombine`, which are used for removing the flatfield
lamp's spectrum (called RESPONSE in IRAF). If you're worried this is not being
removed correctly, be sure to set `display=True` and check!

The output from `biascombine` is simply a 2-d `~numpy.ndarray` that contains the
combined flat. The outputs from `flatcombine` are 1) the combined flat as a
2-d numpy array and 2) the "flat mask", a 1-d array that defines the illuminated
portion of the chip along the y-axis (spatial dimension). You pass this to
several functions later on. It is totally fine to *not* pass it to subsequent
functions, just make sure you're consistent with it! I suggest you do use it.

Next, let's define the wavelength solution that will be used for our science
image. This is the most tedious manual step. We'll mimic IRAF and do the
"identify" manually. While *pyDIS* can do this automatically
(set `interac=False`), the solution is often too crude for science use. If you
have previously done the manual identify and picked out lines, you can skip the
identify step by setting `previous=True`. We'll start with a random guess at the
`fit_order` parameter, but if `interac=True` then we'll be prompted to adjust
this interactively:

.. code-block:: python

    HeNeAr_file = 'HeNeAr.0030r.fits'
    wfit = pydis.HeNeAr_fit(HeNeAr_file, trim=True, fmask=fmask_out,
                              interac=True, mode='poly', fit_order=5)

*pyDIS* will pull the very rough wavelength solution out of the header, take a
slice through the image, and show you the 1-d emission line spectrum. Click on
prominent peaks in the image (sample arc lamp spectra for DIS is provided by
APO `here <http://www.apo.nmsu.edu/arc35m/Instruments/DIS/#4p2>`_), then in the
Python terminal type the wavelength of the line you identified and press
<return>. Each line is fit with a Gaussian to find the exact center. Repeat
this until you've identified as many lines as you can. If you mess one up,
click on it and enter a "d" for the wavelength.

After the lines are identified, you need to fit a smooth function between them.
The polynomial fit and residual will be shown. Close this window and then in the
terminal answer the question: should the polynomial order be changed (if so,
enter an integer number) or is it OK (if so, enter a "d" for done)?

*pyDIS* will then trace these lines along the y-axis, producing a full 2-d
wavelength solution for the image. Be sure to keep this solution `wfit`
around, as we'll need to apply to every image we want a spectrum from
(science and flux standard).


Reduce the science image
++++++++++++++++++++++++

This is the "reduction" step, where we actually remove the bias and flat from
the science image, and divide by the exposure time. The result is an image with
units of counts/second. Everything (almost) is stored in `~numpy.ndarray`, so
performing simple math on them is trivial:

.. code-block:: python

    # the science image file name
    spec = 'object.0035r.fits'

    # open the image, get the data and a couple important numbers
    img = pydis.OpenImg(spec, trim=True)
    raw = img.data
    exptime = img.exptime
    airmass = img.airmass

    # remove bias and flat, divide by exptime
    data = ((raw - bias) / flat) / exptime

You should also read in and reduce the flux standard star the same way:

.. code-block:: python

    # the flux standard file name
    stdspec = 'Feige34.0065r.fits'
    stdraw, stdexptime, stdairmass, _ = pydis.OpenImg(stdspec, trim=True)
    stddata= ((stdraw - bias) / flat) / stdexptime

If we wanted to look at the science image in python, you might do this

.. code-block::python

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(np.log10(data), origin='lower',aspect='auto',cmap=cm.Greys_r)
    plt.show()


Find and trace the spectrum
+++++++++++++++++++++++++++

Examining the image above you should see the bright horizontal spectrum across
the chip. We want to quantify this shape, tracing the flux across the chip along
the wavelength dimension. If the target is bright and the spectrum is not too
curved, this should be pretty simple! If there are multiple objects on the slit
(multiple bright horizontal streaks in the image) then you want to select the
target manually (`interac=True`).

Set `display=True` to see the trace over-plotted on the image. Make sure it
doesn't wander. If it's not accurate enough, adjust the number of steps. For
low Signal/Noise images, sometimes you have to use small values like `nsteps=7`
to trace the spectrum.

.. code-block:: python

    # trace the science image
    trace = pydis.ap_trace(data, fmask=fmask_out, nsteps=50, interac=False, display=True)

    # trace the flux standard image
    stdtrace = pydis.ap_trace(stddata, fmask=fmask_out, nsteps=50, interac=False, display=True)


Now we extract the observed spectrum along this trace, for both the reduced
object and the flux standard pydis. The result is a 1-d spectrum made by summing
up the flux in each column in a range of +/- the aperture width. The sky is
determined by fitting a polynomial along the column in two regions near the trace.
Important consideration: if you choose a large sky region, or separate it a lot
from the aperture, the sky will not be a good fit. This is because the HeNeAr
lines (lines of constant wavelength) are bent and not perfectly vertical on the
chip. Thus it is good to choose a small sky region. The `skydeg` parameter is
the polynomial order to fit between the sky regions in each column. The default
is `skydeg=0`, which is simply a median. A sky value at each pixel along the
trace is returned. This routine also computes a flux error at each pixel along
the trace.

.. code-block::python

    ext_spec, sky, fluxerr = pydis.ap_extract(data, trace, apwidth=5,skysep=1,
                                        skywidth=7, skydeg=0)
    ext_std, stdsky, stderr = pydis.ap_extract(stddata, stdtrace, apwidth=5,
                                         skysep=1, skywidth=7, skydeg=0)

    # subtract the sky from the 1-d spectrum
    flux_red = (ext_spec - sky) # the reduced object
    flux_std = (ext_std - stdsky) # the reduced flux standard


You could make sure these values for the aperture and sky regions were sensible
by plotting the image with the lines overlaid.

.. code-block:: python

    xbins = np.arange(data.shape[1])

    plt.figure()
    plt.imshow(np.log10(data), origin='lower',aspect='auto',cmap=cm.Greys_r)

    # the trace
    plt.plot(xbins, trace,'b',lw=1)

    # the aperture
    plt.plot(xbins, trace-apwidth,'r',lw=1)
    plt.plot(xbins, trace+apwidth,'r',lw=1)

    # the sky regions
    plt.plot(xbins, trace-apwidth-skysep,'g',lw=1)
    plt.plot(xbins, trace-apwidth-skysep-skywidth,'g',lw=1)
    plt.plot(xbins, trace+apwidth+skysep,'g',lw=1)
    plt.plot(xbins, trace+apwidth+skysep+skywidth,'g',lw=1)

    plt.title('(with trace, aperture, and sky regions)')
    plt.show()

Calibrate the spectrum
++++++++++++++++++++++

We have a raw 1-d spectrum now, with flux measured at each pixel along the
x-axis of the image along the trace. However, we want to calibrate the x-axis
to use the wavelength solution we created before. You probably want to use
`mode='poly'`, which is the default and thus optional to write.

The spectrum itself is currently in units of counts/second, and we need to apply
a wavelength dependent correction for the observed airmass. We'll use the
airmass file for APO included with *pyDIS*. Note, this is different from the
file that APO provided prior to April 2015 (an error in this file was noticed
during the development of *pyDIS*).

.. code-block:: python

    # map the wavelength using the HeNeAr fit
    wfinal = pydis.mapwavelength(trace, wfit, mode='poly')
    wfinalstd = pydis.mapwavelength(stdtrace, wfit, mode='poly')

    # correct the object and flux std for airmass extinction
    flux_red_x = pydis.AirmassCor(wfinal, flux_red, airmass,
                                    airmass_file='apoextinct.dat')
    flux_std_x = pydis.AirmassCor(wfinalstd, flux_std, stdairmass,
                                    airmass_file='apoextinct.dat')

Flux calibration
++++++++++++++++

Now that we have the actual wavelength mapped to both the science target and the
flux standard, and we've corrected them for their respective airmass, it's time
for the final step: flux calibration. This is done by comparing the flux
standard observation to a library of standard stars. For *pyDIS* we have
hard-coded the IRAF library "spec50cal". Any other standard could be put in this
directory and used, or a different library could be used with a little hacking
to the code if you're brave or desperate. `DefFluxCal` will try to avoid Balmer
lines, but at present does not have a sophisticated interactive mode where you
can delete bad points. Set `display=True` in `DefFluxCal` to see if the fit
looks smooth and good and does not blow up at the edges.

The sensitivity function is computed for the standard star, and has units of
(erg/s/cm2/A) / (counts/s/cm2/A). The final step is to apply this sensfunc to
the science target, which simply resamples the sensfunc on to the exact
wavelengths as the target and then multiplies the observed 1-d spectrum by the
sensitivity function. Note *pyDIS* works in flux units, not magnitude units
used by IRAF. As a reality check, you can also apply the sensfunc back to the
standard star spectrum to make sure it looks right!

.. code-block:: python

    sensfunc = pydis.DefFluxCal(wfinalstd, flux_std_x, mode='spline',
                                stdstar='spec50cal/feige34.dat')

    # final step in reduction, apply sensfunc
    ffinal,efinal = pydis.ApplyFluxCal(wfinal, flux_red_x, fluxerr,
                                        wfinalstd, sensfunc)

Our final parameters for the science target are now `(wfinal, ffinal, efinal)`,
the wavelength, flux, and flux error. Congratulations, you have fully reduced
one spectrum in Python! If you'd like to view the result, you could do this:

.. code-block:: python

    plt.figure()
    # plt.plot(wfinal, ffinal)
    plt.errorbar(wfinal, ffinal, yerr=efinal)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title(spec)
    #plot within percentile limits
    plt.ylim( (np.percentile(ffinal,2),
               np.percentile(ffinal,98)) )
    plt.show()

Closing thoughts...
+++++++++++++++++++

This procedure replicates (sometimes poorly) IRAF functions. We are also
skipping a couple lesser-used functions, such as the illumination correction.
You should be plotting things every step of the way, always making sure it looks
sensible. Crazy things can happen if you're not careful...

The reduction script can be applied to many science targets from the same night,
using the same calibration files and flux standards. In fact, this exact
procedure is carried out in `autoreduce`.

I would not recommend using calibrations from another night if possible. To get
better wavelength calibrations, use a HeNeAr image taken right before or after
the science target, or even a different HeNeAr for each science image. Future
helper-scripts in *pyDIS* will accommodate automatic solutions of multiple
HeNeAr files.