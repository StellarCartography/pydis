# *SPECTRA*
A simple reduction package for one dimensional longslit spectroscopy using Python.

## Examples
### Auto-reduce my data!

The goal is to make SPECTRA as easy to use while observing as possible. Here is an example of a script you might run over and over throughout the night:

````python 
# if SPECTRA isn't in the currnet working directory, add to path
import sys
sys.path.append('/path/to/spectra/')

# must import, of course
import spectra
    
spectra.autoreduce('objlist.txt', 'flatlist.txt', 'biaslist.txt',
                 'HeNeAr.0005r.fits', HeNeAr_interac=False)
````

The `autoreduce` function must be given a list of target objects, a list of flats frames, a list of bias frames, and the path to one HeNeAr calibration frame. In this example, the HeNeAr frame is automatically fit, which usually works reasonably well but should not be trusted for e.g. sub-pixel velocity calibration.

Many keywords are available to customize the `autoreduce` function. Here are the default definitions:


````python
autoreduce(speclist, flatlist, biaslist, HeNeAr_file,
           trace1=False, ntracesteps=25,
           apwidth=3,skysep=25,skywidth=75, HeNeAr_interac=False,
           HeNeAr_tol=20, HeNeAr_order=2, displayHeNeAr=False,
           trim=True, write_reduced=True, display=True)
````

Master flat and bias files (FLAT.fits and BIAS.fits by default), trace files with x,y coordinates (.trace files), and two column wavelength,flux spectra (.spec files), will be written at the end.


### Manually reduce stuff
You can also use each component of the reduction process. For example, if you wanted to combine all your flat and bias frames:

````python 
bias = spectra.biascombine('biaslist.txt')
flat, mask = spectra.flatcombine('flatlist.txt', bias)
````

The resulting flat and bias frames are returned as numpy arrays. By default these functions will write files called **BIAS.fits** and **FLAT.fits**, unless a different name is specified using the `output = 'file.fits'` keyword.
Note also that `flatcombine` returns both the data array and a 1-d "mask" array, which determines from the flat the portion of the CCD that is illuminated.





## About

This is a side project, attempting to create a full Python (IRAF free) reduction and extraction pipeline for low/medium resolution longslit spectra. Currently we are using many simple assumptions to get a quick-and-dirty solution, and modeling the workflow after the robust industry standards set by IRAF.

So far we are only using data from the low/medium resolution [APO 3.5-m](http://www.apo.nmsu.edu) "Dual Imaging Spectrograph" (DIS). Therefore, many instrument specific assumptions are being made.

### Motivation
Really slick tools exist for on-the-fly photometry analysis. However, no turn-key spectra toolkit for Python (without IRAF or PyRAF) is currently available. Here are some mission statements:

- Being able to extract and see data in real time at the telescope would be extremely helpful!
- This pipeline doesn't have to give perfect results to be very useful
- Don't try to build a *One Size Fits All* solution for every possible instrument or science case. We cannot beat IRAF at it's own game. IRAF is the industry standard
- The pipeline does need to handle:
	- Flats 
	- Biases 
	- Spectrum Tracing
	- Wavelength Calibration using HeNeAr arc lamp spectra
	- Sky Subtraction
	- Extraction
- Flux Calibration is a good goal
- The more hands-free the better, a full reduction script needs to be available
- A fully interactive mode (a la IRAF) should be available for each task

So far SPECTRA can do a rough job of all the reduction tasks, except flux calibration, for single point sources objects. We are seeking more data to test it against, to help refine the solution and find bugs. Here is one example of a hands-free reduced M dwarf spectrum versus the manual IRAF reduction (note: pyDIS has become SPECTRA)

![Imgur](http://i.imgur.com/IjXdt39l.png)

### How to Help
Check out the Issues page if you think you can help code! Or if you have some data already reduced that you trust and would be willing to share, let us know!