# *SPECTRA*
A simple reduction package for one dimensional longslit spectroscopy using Python. Also colloqiually known as "pyDIS", as it was designed for DIS

The goal of *SPECTRA* is to provide a turn-key solution for reducing and understanding longslit spectroscopy, which could ideally be done in real time. Currently we are using many simple assumptions to get a quick-and-dirty solution, and modeling the workflow after the robust industry standards set by IRAF. Additionally, we have only used data from the low/medium resolution [APO 3.5-m](http://www.apo.nmsu.edu) "Dual Imaging Spectrograph" (DIS). Therefore, many instrument specific assumptions are being made.



## Examples
#### Auto-reduce my data!

The goal is to make SPECTRA as easy to use while observing as possible. Here is an example of a script you might run over and over throughout the night:

````python 
# if SPECTRA isn't in the currnet working directory, add to path
import sys
sys.path.append('/path/to/spectra/')

# must import, of course
import spectra
    
spectra.autoreduce('objlist.txt', 'flatlist.txt', 'biaslist.txt',
                 'HeNeAr.0005r.fits', stdstar='fiege34')
````

The `autoreduce` function must be given a list of target objects, a list of flats frames, a list of bias frames, and the path to one HeNeAr calibration frame. In this example, the HeNeAr frame is automatically fit, which usually works reasonably well but should not be trusted for e.g. sub-pixel velocity calibration.

*Many* keywords are available to customize the `autoreduce` function for most needs.


#### Manually reduce stuff
You can also use each component of the reduction process. For example, if you wanted to combine all your flat and bias frames:

````python 
bias = spectra.biascombine('biaslist.txt')
flat, mask = spectra.flatcombine('flatlist.txt', bias)
````

The resulting flat and bias frames are returned as numpy arrays. By default these functions will write files called **BIAS.fits** and **FLAT.fits**, unless a different name is specified using the `output = 'file.fits'` keyword.
Note also that `flatcombine` returns both the data array and a 1-d "mask" array, which determines from the flat the portion of the CCD that is illuminated.




## Motivation
Really slick tools exist for on-the-fly photometry analysis. However, no turn-key, easy to use spectra toolkit for Python (without IRAF or PyRAF) was available (that we were aware of). Here are some mission statements:

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
	- basic Flux Calibration
- The more hands-free the better, a full reduction script needs to be available
- A fully interactive mode (a la IRAF) should be available for each task

So far SPECTRA can do a rough job of all the reduction tasks for single point sources objects! We are seeking more data to test it against, to help refine the solution and find bugs. Here is one example of a **totally hands-free reduced M dwarf spectrum** versus the manual IRAF reduction:

![Imgur](http://i.imgur.com/4Y55NZHl.png)

**This spectrum took a few second to reduce, and is good enough for a quick-look!** There are defintely errors in the wavelength, and small offsets in the flux calibration. A (terrible) brute-force wavelength solution, and sometimes fickle flux calibration are being used here. With some minimal parameter tweaking and manual lamp-line identifications the results are even better!



## How to Help

- Check out the Issues page if you think you can help code, or want to requst a feature! 
- If you have some data already reduced in IRAF that you trust and would be willing to share, let us know!