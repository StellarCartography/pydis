'''
Example 2. Co-adding data

ABOUT:
    In this example we demonstrate the wrapper to reduce multiple exposures
    of the same target. Currently this wrapper only reduces 1 object, not a
    whole night of data.

    ReduceCoAdd works by literally adding the reduced frames together before
    extracting the spectrum. It also assumes the flux standard star provides
    the trace. Both blue and red data are reduced separately, and then manually
    combined to make a single plot.

DATA:
    APO, DIS-low, red & blue

CREDIT:
    Data for this example generously provided by John J. Ruan.
'''

from pydis.wrappers import ReduceCoAdd
import numpy as np
import matplotlib.pyplot as plt

# red channel reduction
wr, fr, er = ReduceCoAdd('robj.lis', 'rflat.lis', 'rbias.lis', # standard input lists
                         'data/HeNeAr.0030r.fits', # explicitly point to the arc to use
                         HeNeAr_interac=True, # do interactive arcs
                         HeNeAr_order=5, # arc fit order to use
                         stdstar='spec50cal/feige34', # what flux standard to use
                         HeNeAr_prev=True, # Use previously found interactive arc line?
                         apwidth=6, # width of spectrum aperture
                         skydeg=0, # polynomial order to use to fit the sky
                         skysep=1, # dist in pixels btwn aperture and sky
                         skywidth=7, # size of sky area to use
                         ntracesteps=15, # number of spline pivot points to trace sky with
                         display=False) # no plots output to screen

# blue channel reduction
wb, fb, eb = ReduceCoAdd('bobj.lis', 'bflat.lis', 'bbias.lis',
                         'data/HeNeAr.0030b.fits', HeNeAr_order=2, HeNeAr_interac=True,
                         stdstar='spec50cal/feige34',
                         HeNeAr_prev=True,
                         skydeg=0, apwidth=6, skysep=1, skywidth=7, ntracesteps=7,
                         display=False)

# the data blows up at the edges - so let's trim them
x1 = np.where((wb<5400) & # blue channel
              (wb>3500))
x2 = np.where((wr>5300)) # red channel

# make a plot of the red + blue channels together
plt.figure()
plt.plot(wb[x1],fb[x1],'b',alpha=0.7)
plt.plot(wr[x2],fr[x2],'r',alpha=0.7)
plt.ylim((-0.2e-15,0.4e-15))
plt.xlim((3500,9800))
plt.xlabel('Wavelength (A)')
plt.ylabel(r'Flux (erg/s/cm$^2$/A)')
plt.savefig('object.png')
plt.show()
