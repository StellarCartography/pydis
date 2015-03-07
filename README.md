# pyDIS

### About
This is a scratch project attempting to create a full Python (IRAF free) reduction and extraction pipeline for low/medium resolution longslit spectra. Currently we are using loads of simple assumptions, and modeling the workflow after the robust industry standards from IRAF.

So far we are only using data from the low/medium resolution [APO 3.5-m](http://www.apo.nmsu.edu) "Dual Imaging Spectrograph" (DIS). Therefore, many instrument specific assumptions are being made.


### Motivation
Really slick tools exist for on-the-fly photometry analysis. However, no turn-key spectra toolkit for Python (without IRAF or PyRAF) is currently available. Here are some mission statements:

- Being able to extract and see data in real time at the telescope would be extremely helpful!
- This pipeline doesn't have to give perfect results to be very useful
- We cannot beat IRAF at it's own game. IRAF is the industry standard
- Don't try to build a One Size Fits All solution for every possible instrument or science case
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

So far `pyDIS` can do a rough job of all the reduction tasks, except flux calibration, for single point sources. No interactive mode is currently available for any task.


### How to Help
Check out the Issues page if you think you can help code! Or if you have some data already reduced that you trust and would be willing to share, let us know!