import sys
sys.path.append('../')
import pydis


pydis.autoreduce('bdata.lis','bflat.lis', 'bbias.lis',
                 'HeNeAr.0028b.fits',
                 trim=True, display=False)
