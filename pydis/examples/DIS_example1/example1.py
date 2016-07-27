'''
Example 1. Basic automatic reduction

ABOUT:
    In this example we demonstrate the wrapper to automatically reduce data.
    This example only has the red channel, because the star is very cool.

    autoreduce goes through the standard longslit reduction, and can reduce an
    entire night's worth of data. This wrapper is also the one I suggest you
    modify if you want to write a custom reduction script.

DATA:
    APO, DIS-high, red

CREDIT:
    Data for this example generously provided by Suzanne L. Hawley
'''

from pydis.wrappers import autoreduce

autoreduce('obj.lis', 'flat.lis', 'bias.lis', # the object, flat, and bias lists
           'data/05may31.0035r.fits',  # explicitly point to the arc to use
           stdstar='spec50cal/bd284211',  # what flux standard to use
           apwidth=8, # width of spectrum aperture
           skysep=3, # dist in pixels btwn aperture and sky
           skywidth=7, # size of sky area to use
           skydeg=0, # polynomial order to use to fit the sky
           ntracesteps=15, # number of spline pivot points to trace sky with
           HeNeAr_interac=True, # do interactive arcs?
           HeNeAr_order=5, # arc fit order to use
           HeNeAr_prev=True, # Use previously found interactive arc line?
           display=True) # output lots of plots to the screen