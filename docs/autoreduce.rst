.. include:: references.txt

.. _autoreduce:

***********************
The autoreduce function
***********************

The goal is to make *pyDIS* as easy to use while observing as possible. The
`~pydis.autoreduce` function makes this possible by wrapping all the components of
the standard DIS reduction, and using basic assumptions. Steps in this
reduction include:

    1. combines bias and flat images
    2. maps wavelength in the HeNeAr image
    3. perform simple image reduction: Data = (Raw - Bias)/Flat
    4. trace spectral aperture
    5. extract spectrum
    6. measure sky along extracted spectrum
    7. apply flux calibration
    8. write output files

Steps (3) - (7) are performed on every target frame and the flux standard star

Here is an example of a script you might run over and over throughout the night
to reduce all your data:

.. code-block:: python

    # if pyDIS isn't in the currnet working directory, add to path
    import sys
    sys.path.append('/path/to/pydis/')

    # must import, of course
    import pydis

    # reduce and extract the data with the fancy autoreduce script
    wave, flux, err = pydis.autoreduce('objlist.r.txt', 'flatlist.r.txt', 'biaslist.r.txt',
                                       'HeNeAr.0005r.fits', stdstar='spec50cal/fiege34.dat')

These are the minimum arguments you need to supply to make autoreduce work.
Some notes on each one:

    - reduce the RED and BLUE channels in DIS separately!
    - the object list should *first* include the flux standard star
      (not RV standard), then all the targets to reduce
    - the keyword arg `stdstar` should be set to the name of the standard star from `spec50cal <https://github.com/TheAstroFactory/pydis/tree/master/pydis/resources/onedstds>`_
    - the flat and bias lists should just be simple lists of all corresponding
      images to combine
    - the HeNeAr lamp image is applied to *all* images, both standard and target
      frames. This does not produce the best wavelength stability, and observers
      will probably want to take HeNeAr frames after each target or regularly
      throughout the night. Supporting multiple HeNeAr images is a future goal!


*Many* keywords are available to customize the `~pydis.autoreduce` function for
most needs. Here is the full definition list with default parameters.

.. code-block:: python

    def autoreduce(speclist, flatlist, biaslist, HeNeAr_file,
                   stdstar='', trace1=False, ntracesteps=15,
                   airmass_file='kpnoextinct.dat',
                   flat_mode='spline', flat_order=9, flat_response=True,
                   apwidth=8,skysep=3,skywidth=7, skydeg=0,
                   HeNeAr_prev=False, HeNeAr_interac=False,
                   HeNeAr_tol=20, HeNeAr_order=3, displayHeNeAr=False,
                   trim=True, write_reduced=True,
                   display=True, display_final=True):


Note: even more args and kwargs are available for the individual functions that
autoreduce calls! Also, this might be slightly out of date. Always check out the
API for `~pydis.autoreduce` for the latest features!