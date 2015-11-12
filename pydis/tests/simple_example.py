import pydis

# This is a simple example of reducing a list of images (bdata.lis)
# in a hands-free manner. A reduced, wavelength calibrated spectrum
# should appear for each file when finished.

pydis.autoreduce('bdata.lis','bflat.lis', 'bbias.lis',
                   'HeNeAr.0028b.fits',
                   apwidth=5, skysep=1, skywidth=15,
                   trace1=True, HeNeAr_interac=False,
                   trim=True, display=False)
