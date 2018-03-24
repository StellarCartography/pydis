'''
This file contains useful wrappers for the primary reduction and analysis
functions of pyDIS. These helper routines enable painless reduction of,
for example, an entire night of simple data (autoreduce).

'''

import pydis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime

from .pydis import (DefFluxCal, HeNeAr_fit, flatcombine, biascombine, OpenImg,
                    ap_trace, ap_extract, mapwavelength, AirmassCor,
                    ApplyFluxCal)

__all__ = ['autoreduce', 'CoAddFinal', 'ReduceCoAdd', 'ReduceTwo']


def _WriteSpec(spec, wfinal, ffinal, efinal, trace):
    # write file with the trace (y positions)
    tout = open(spec+'.trace','w')
    tout.write('#  This file contains the x,y coordinates of the trace \n')
    for k in range(len(trace)):
        tout.write(str(k)+', '+str(trace[k]) + '\n')
    tout.close()

    # write the final spectrum out
    fout = open(spec+'.spec','w')
    fout.write('#  This file contains the final extracted (wavelength,flux,err) data \n')
    for k in range(len(wfinal)):
        fout.write(str(wfinal[k]) + '  ' + str(ffinal[k]) + '  ' + str(efinal[k]) + '\n')
    fout.close()
    return


def autoreduce(speclist, flatlist='', biaslist='', HeNeAr_file='',
               stdstar='', trace_recenter=False, trace_interac=True,
               trace1=False, ntracesteps=15,
               airmass_file='apoextinct.dat',
               flat_mode='spline', flat_order=9, flat_response=True,
               apwidth=8, skysep=3, skywidth=7, skydeg=0,
               HeNeAr_prev=False, HeNeAr_interac=True,
               HeNeAr_tol=20, HeNeAr_order=3, display_HeNeAr=False,
               std_mode='spline', std_order=12, display_std=False,
               trim=True, write_reduced=True,
               display=True, display_final=True,
               HeNeAr_second_pass=True,
               silent=True):
    """
    A wrapper routine to carry out the full steps of the spectral
    reduction and calibration. Steps include:
    1) combines bias and flat images
    2) maps wavelength in the HeNeAr image
    3) perform simple image reduction: Data = (Raw - Bias)/Flat
    4) trace spectral aperture
    5) extract spectrum
    6) measure sky along extracted spectrum
    7) apply flux calibration
    8) write output files

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
    stdstar : str
        Name of the standard star to use for flux calibration. If
        nothing is entered for "stdstar", no flux calibration will be
        computed. (Default is '').
        NOTE1: must include the subdir for the star, e.g. 'spec50cal/feige34'.
        NOTE2: Assumes the first star in "speclist" is the standard star.
    trace1 : bool, optional
        use trace1=True if only perform aperture trace on first object in
        speclist. Useful if e.g. science targets are faint, and first
        object is a bright standard star. Note: assumes star placed at
        same position in spatial direction. (Default is False)
    trace_recenter : bool, optional
        If trace1=True, set this to True to allow for small linear
        adjustments to the trace (default is False)
    trace_interac : bool, optional
        Set to True if user should interactively select aperture center
        for each object spectrum. (Default is True)
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
    display_HeNeAr : bool, optional
    std_mode : str, optional
        Fit mode to use with the flux standard star. Options are 'spline'
        and 'poly' (Default is 'spline')
    std_order : int, optional
        The order of polynomial to fit, if std_mode='poly'. (Default is 12)
    display_std : bool, optional
        If set, display plots of the flux standard being fit (Default is
        False)
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
    display_final : bool, optional
        Set to False to suppress plotting the final reduced spectrum to
        the screen. Useful for running in quiet batch mode. (Default is
        True)

    """

    if (len(biaslist) > 0):
        bias = biascombine(biaslist, trim=trim, silent=silent)
    else:
        bias = 0.0

    if (len(biaslist) > 0) and (len(flatlist) > 0):
        flat,fmask_out = flatcombine(flatlist, bias, trim=trim,
                                           mode=flat_mode,display=False,
                                           flat_poly=flat_order, response=flat_response)
    else:
        flat = 1.0
        fmask_out = (1,)


    if HeNeAr_prev is False:
        prev = ''
    else:
        prev = HeNeAr_file+'.lines'

    # do the HeNeAr mapping first, must apply to all science frames
    if (len(HeNeAr_file) > 0):
        wfit = HeNeAr_fit(HeNeAr_file, trim=trim, fmask=fmask_out,
                                interac=HeNeAr_interac, previous=prev, mode='poly',
                                display=display_HeNeAr, tol=HeNeAr_tol,
                                fit_order=HeNeAr_order, second_pass=HeNeAr_second_pass)


    # read in the list of target spectra
    # assumes specfile is a list of file names of object
    specfile = np.array([np.genfromtxt(speclist, dtype=np.str)]).flatten()

    for i in range(len(specfile)):
        spec = specfile[i]
        
        if silent is False:
            print("> Processing file "+spec+" ["+str(i)+"/"+str(len(specfile))+"]")

        # raw, exptime, airmass, wapprox = pydis.OpenImg(spec, trim=trim)
        img = OpenImg(spec, trim=trim)
        raw = img.data
        exptime = img.exptime
        airmass = img.airmass
        wapprox = img.wavelength


        # remove bias and flat, divide by exptime
        data = ((raw - bias) / flat) / exptime

        if display is True:
            plt.figure()
            plt.imshow(np.log10(data), origin = 'lower',aspect='auto',cmap=cm.Greys_r)
            plt.title(spec+' (flat and bias corrected)')
            plt.show()

        # with reduced data, trace the aperture
        if (i==0) or (trace1 is False):
            trace = ap_trace(data,fmask=fmask_out, nsteps=ntracesteps,
                             recenter=trace_recenter, interac=trace_interac)

        # extract the spectrum, measure sky values along trace, get flux errors
        ext_spec, sky, fluxerr = ap_extract(data, trace, apwidth=apwidth,
                                            skysep=skysep,skywidth=skywidth,
                                            skydeg=skydeg,coaddN=1)

        xbins = np.arange(data.shape[1])
        if display is True:
            plt.figure()
            plt.imshow(np.log10(data), origin='lower',aspect='auto',cmap=cm.Greys_r)
            plt.plot(xbins, trace,'b',lw=1)
            plt.plot(xbins, trace-apwidth,'r',lw=1)
            plt.plot(xbins, trace+apwidth,'r',lw=1)
            plt.plot(xbins, trace-apwidth-skysep,'g',lw=1)
            plt.plot(xbins, trace-apwidth-skysep-skywidth,'g',lw=1)
            plt.plot(xbins, trace+apwidth+skysep,'g',lw=1)
            plt.plot(xbins, trace+apwidth+skysep+skywidth,'g',lw=1)

            plt.title('(with trace, aperture, and sky regions)')
            plt.show()


        if (len(HeNeAr_file) > 0):
            wfinal = mapwavelength(trace, wfit, mode='poly')
        else:
            # if no line lamp given, use approx from the img header
            wfinal = wapprox

        # plt.figure()
        # plt.plot(wfinal,'r')
        # plt.show()

        # subtract local sky level, divide by exptime to get flux units
        #     (counts / sec)
        flux_red = (ext_spec - sky)

        # now correct the spectrum for airmass extinction
        flux_red_x = AirmassCor(wfinal, flux_red, airmass,
                                airmass_file=airmass_file)

        # now get flux std IF stdstar is defined
        # !! assume first object in list is std star !!
        if (len(stdstar) > 0) and (i==0):
            sens_flux = DefFluxCal(wfinal, flux_red_x, stdstar=stdstar,
                                   mode=std_mode, polydeg=std_order, display=display_std)
            sens_wave = wfinal

        elif (len(stdstar) == 0) and (i==0):
            # if 1st obj not the std, then just make array of 1's to multiply thru
            sens_flux = np.ones_like(flux_red_x)
            sens_wave = wfinal

        # final step in reduction, apply sensfunc
        ffinal,efinal = ApplyFluxCal(wfinal, flux_red_x, fluxerr,
                                           sens_wave, sens_flux)


        if write_reduced is True:
            _WriteSpec(spec, wfinal, ffinal, efinal, trace)

            now = datetime.datetime.now()

            lout = open(spec+'.log','w')
            lout.write('#  This file contains the reduction parameters \n'+
                       '#  used in autoreduce for '+spec+'\n')
            lout.write('DATE-REDUCED = '+str(now)+'\n')
            lout.write('HeNeAr_tol   = '+str(HeNeAr_tol)+'\n')
            lout.write('HeNeAr_order = '+str(HeNeAr_order)+'\n')
            lout.write('trace1       = '+str(trace1)+'\n')
            lout.write('ntracesteps  = '+str(ntracesteps)+'\n')
            lout.write('trim         = '+str(trim)+'\n')
            lout.write('response     = '+str(flat_response)+'\n')
            lout.write('apwidth      = '+str(apwidth)+'\n')
            lout.write('skysep       = '+str(skysep)+'\n')
            lout.write('skywidth     = '+str(skywidth)+'\n')
            lout.write('skydeg       = '+str(skydeg)+'\n')
            lout.write('stdstar      = '+str(stdstar)+'\n')
            lout.write('airmass_file = '+str(airmass_file)+'\n')
            lout.close()


        if display_final is True:
            # the final figure to plot
            plt.figure()
            # plt.plot(wfinal, ffinal)
            plt.errorbar(wfinal, ffinal, yerr=efinal)
            plt.xlabel('Wavelength')
            plt.ylabel('Flux')
            plt.title(spec)
            #plot within percentile limits
            plt.ylim( (np.nanpercentile(ffinal,2),
                       np.nanpercentile(ffinal,98)) )
            plt.show()

    return


def ReduceCoAdd(speclist, flatlist, biaslist, HeNeAr_file,
                stdstar='', trace1=False, ntracesteps=15,
                flat_mode='spline', flat_order=9, flat_response=True,
                apwidth=6,skysep=1,skywidth=7, skydeg=0,
                HeNeAr_prev=False, HeNeAr_interac=False,
                HeNeAr_tol=20, HeNeAr_order=2, displayHeNeAr=False,
                HeNeAr_second_pass=True,
                trim=True, write_reduced=True, display=True):
    """
    A special version of autoreduce, that assumes all the target images
    want to be median co-added and then extracted. All images have flat
    and bias removed first, then are combined. Trace and Extraction
    happens only on the final combined image.

    Assumes file names in speclist are the standard star, followed by all the target
    images you want to co-add.


    """
    #-- the basic crap, used for all frames
    bias = biascombine(biaslist, trim=trim)
    flat,fmask_out = flatcombine(flatlist, bias, trim=trim, mode=flat_mode,display=False,
                                 flat_poly=flat_order, response=flat_response)
    if HeNeAr_prev is False:
        prev = ''
    else:
        prev = HeNeAr_file+'.lines'
    wfit = HeNeAr_fit(HeNeAr_file, trim=trim, fmask=fmask_out,
                            interac=HeNeAr_interac, previous=prev, mode='poly',
                            display=displayHeNeAr, tol=HeNeAr_tol,
                            fit_order=HeNeAr_order, second_pass=HeNeAr_second_pass)

    #-- the standard star, set the stage
    specfile = np.array([np.genfromtxt(speclist, dtype=np.str)]).flatten()
    spec = specfile[0]
    # raw, exptime, airmass, wapprox = pydis.OpenImg(spec, trim=trim)
    img = OpenImg(spec, trim=trim)
    raw = img.data
    exptime = img.exptime
    airmass = img.airmass
    wapprox = img.wavelength

    data = ((raw - bias) / flat) / exptime

    trace = ap_trace(data,fmask=fmask_out, nsteps=ntracesteps)
    # extract the spectrum, measure sky values along trace, get flux errors
    ext_spec, sky, fluxerr = ap_extract(data, trace, apwidth=apwidth,
                                        skysep=skysep,skywidth=skywidth,
                                        skydeg=skydeg,coaddN=1)
    xbins = np.arange(data.shape[1])
    wfinal = mapwavelength(trace, wfit, mode='poly')
    flux_red_x = (ext_spec - sky)
    sens_flux = DefFluxCal(wfinal, flux_red_x, stdstar=stdstar,
                           mode='spline',polydeg=12)
    sens_wave = wfinal
    ffinal,efinal = ApplyFluxCal(wfinal, flux_red_x, fluxerr, sens_wave, sens_flux)

    #-- the target star exposures, stack and proceed
    for i in range(1,len(specfile)):
        spec = specfile[i]
        # raw, exptime, airmass, wapprox = pydis.OpenImg(spec, trim=trim)
        img = OpenImg(spec, trim=trim)
        raw = img.data
        exptime = img.exptime
        airmass = img.airmass
        wapprox = img.wavelength
        data_i = ((raw - bias) / flat) / exptime
        if (i==1):
            all_data = data_i
        elif (i>1):
            all_data = np.dstack( (all_data, data_i))
    data = np.median(all_data, axis=2)
    # extract the spectrum, measure sky values along trace, get flux errors
    ext_spec, sky, fluxerr = ap_extract(data, trace, apwidth=apwidth,
                                        skysep=skysep,skywidth=skywidth,
                                        skydeg=skydeg,coaddN=len(specfile))
    xbins = np.arange(data.shape[1])

    wfinal = mapwavelength(trace, wfit, mode='poly')
    flux_red_x = (ext_spec - sky)
    ffinal,efinal = ApplyFluxCal(wfinal, flux_red_x, fluxerr, sens_wave, sens_flux)

    if display is True:
        plt.figure()
        plt.plot(wfinal, ffinal)
        plt.title("CO-ADD DONE")
        plt.ylim( (np.nanpercentile(ffinal,5),
                   np.nanpercentile(ffinal,95)) )
        plt.show()

    return wfinal, ffinal, efinal


def CoAddFinal(frames, mode='mean', display=True):
    # co-add FINSIHED, reduced spectra
    # only trick: resample on to wavelength grid of 1st frame
    files = np.genfromtxt(frames, dtype=np.str,unpack=True)

    # read in first file
    wave_0, flux_0 = np.loadtxt(files[0],dtype='float',skiprows=1,
                                unpack=True,delimiter=',')

    for i in range(1,len(files)):
        wave_i, flux_i = np.loadtxt(files[i],dtype='float',skiprows=1,
                                    unpack=True,delimiter=',')

        # linear interp on to wavelength grid of 1st frame
        flux_i0 = np.interp(wave_0, wave_i, flux_i)

        flux_0 = np.dstack( (flux_0, flux_i0))

    if mode == 'mean':
        flux_out = np.squeeze(flux_0.sum(axis=2) / len(files))
    if mode == 'median':
        flux_out = np.squeeze(np.median(flux_0, axis=2))

    if display is True:
        plt.figure()
        plt.plot(wave_0, flux_out)
        plt.xlabel('Wavelength')
        plt.ylabel('Co-Added Flux')
        plt.show()

    return wave_0, flux_out


def ReduceTwo(speclist, flatlist='', biaslist='', HeNeAr_file='',
               stdstar='', trace_recenter=False, ntracesteps=15,
               airmass_file='apoextinct.dat',
               flat_mode='spline', flat_order=9, flat_response=True,
               apwidth=8, skysep=3, skywidth=7, skydeg=0,
               HeNeAr_prev=False, HeNeAr_interac=True,
               HeNeAr_tol=20, HeNeAr_order=3, display_HeNeAr=False,
               std_mode='spline', std_order=12, display_std=False,
               trim=True, write_reduced=True,
               display=True, display_final=True):


    if (len(biaslist) > 0):
        bias = biascombine(biaslist, trim=trim)
    else:
        bias = 0.0

    if (len(biaslist) > 0) and (len(flatlist) > 0):
        flat,fmask_out = flatcombine(flatlist, bias, trim=trim,
                                           mode=flat_mode,display=False,
                                           flat_poly=flat_order, response=flat_response)
    else:
        flat = 1.0
        fmask_out = (1,)

    if HeNeAr_prev is False:
        prev = ''
    else:
        prev = HeNeAr_file+'.lines'

    # do the HeNeAr mapping first, must apply to all science frames
    if (len(HeNeAr_file) > 0):
        wfit = HeNeAr_fit(HeNeAr_file, trim=trim, fmask=fmask_out,
                                interac=HeNeAr_interac, previous=prev,mode='poly',
                                display=display_HeNeAr, tol=HeNeAr_tol,
                                fit_order=HeNeAr_order)

    # read in the list of target spectra
    # assumes specfile is a list of file names of object
    #-> wrap with array and flatten because Numpy sucks with one-element arrays...
    specfile = np.array([np.genfromtxt(speclist, dtype=np.str)]).flatten()

    for i in range(len(specfile)):
        spec = specfile[i]
        print("> Processing file "+spec+" ["+str(i)+"/"+str(len(specfile))+"]")
        # raw, exptime, airmass, wapprox = pydis.OpenImg(spec, trim=trim)
        img = OpenImg(spec, trim=trim)
        raw = img.data
        exptime = img.exptime
        airmass = img.airmass
        wapprox = img.wavelength

        # remove bias and flat, divide by exptime
        data = ((raw - bias) / flat) / exptime

        if display is True:
            plt.figure()
            plt.imshow(np.log10(data), origin = 'lower',aspect='auto',cmap=cm.Greys_r)
            plt.title(spec+' (flat and bias corrected)')
            plt.show()

        # with reduced data, trace BOTH apertures
        trace_1 = ap_trace(data,fmask=fmask_out, nsteps=ntracesteps,
                                 recenter=trace_recenter, interac=True)
        trace_2 = ap_trace(data,fmask=fmask_out, nsteps=ntracesteps,
                                 recenter=trace_recenter, interac=True)


        xbins = np.arange(data.shape[1])
        if display is True:
            plt.figure()
            plt.imshow(np.log10(data), origin='lower',aspect='auto',cmap=cm.Greys_r)
            plt.plot(xbins, trace_1,'b',lw=1)
            plt.plot(xbins, trace_1-apwidth,'r',lw=1)
            plt.plot(xbins, trace_1+apwidth,'r',lw=1)
            plt.plot(xbins, trace_1-apwidth-skysep,'g',lw=1)
            plt.plot(xbins, trace_1-apwidth-skysep-skywidth,'g',lw=1)
            plt.plot(xbins, trace_1+apwidth+skysep,'g',lw=1)
            plt.plot(xbins, trace_1+apwidth+skysep+skywidth,'g',lw=1)

            plt.plot(xbins, trace_2,'b',lw=1)
            plt.plot(xbins, trace_2-apwidth,'r',lw=1)
            plt.plot(xbins, trace_2+apwidth,'r',lw=1)
            plt.plot(xbins, trace_2-apwidth-skysep,'g',lw=1)
            plt.plot(xbins, trace_2-apwidth-skysep-skywidth,'g',lw=1)
            plt.plot(xbins, trace_2+apwidth+skysep,'g',lw=1)
            plt.plot(xbins, trace_2+apwidth+skysep+skywidth,'g',lw=1)

            plt.title('(Both Traces, with aperture and sky regions)')
            plt.show()


        t_indx = 1
        # now do the processing for both traces as if separate stars
        for trace in [trace_1, trace_2]:
            tnum = '_' + str(t_indx)
            t_indx = t_indx + 1

            # extract the spectrum, measure sky values along trace, get flux errors
            ext_spec, sky, fluxerr = ap_extract(data, trace, apwidth=apwidth,
                                            skysep=skysep,skywidth=skywidth,
                                            skydeg=skydeg,coaddN=1)

            if (len(HeNeAr_file) > 0):
                wfinal = mapwavelength(trace, wfit, mode='poly')
            else:
                # if no line lamp given, use approx from the img header
                wfinal = wapprox

            # subtract local sky level, divide by exptime to get flux units
            #     (counts / sec)
            flux_red = (ext_spec - sky)

            # now correct the spectrum for airmass extinction
            flux_red_x = AirmassCor(wfinal, flux_red, airmass,
                                    airmass_file=airmass_file)

            # now get flux std IF stdstar is defined
            # !! assume first object in list is std star !!
            if (len(stdstar) > 0) and (i==0):
                sens_flux = DefFluxCal(wfinal, flux_red_x, stdstar=stdstar,
                                       mode=std_mode, polydeg=std_order, display=display_std)
                sens_wave = wfinal

            elif (len(stdstar) == 0) and (i==0):
                # if 1st obj not the std, then just make array of 1's to multiply thru
                sens_flux = np.ones_like(flux_red_x)
                sens_wave = wfinal

            # final step in reduction, apply sensfunc
            ffinal,efinal = ApplyFluxCal(wfinal, flux_red_x, fluxerr,
                                               sens_wave, sens_flux)

            if write_reduced is True:
                _WriteSpec(spec+tnum, wfinal, ffinal, efinal, trace)

                now = datetime.datetime.now()

                lout = open(spec+tnum+'.log','w')
                lout.write('#  This file contains the reduction parameters \n'+
                           '#  used in autoreduce for '+spec+'\n')
                lout.write('DATE-REDUCED = '+str(now)+'\n')
                lout.write('HeNeAr_tol   = '+str(HeNeAr_tol)+'\n')
                lout.write('HeNeAr_order = '+str(HeNeAr_order)+'\n')
                lout.write('trace1       = '+str(False)+'\n')
                lout.write('ntracesteps  = '+str(ntracesteps)+'\n')
                lout.write('trim         = '+str(trim)+'\n')
                lout.write('response     = '+str(flat_response)+'\n')
                lout.write('apwidth      = '+str(apwidth)+'\n')
                lout.write('skysep       = '+str(skysep)+'\n')
                lout.write('skywidth     = '+str(skywidth)+'\n')
                lout.write('skydeg       = '+str(skydeg)+'\n')
                lout.write('stdstar      = '+str(stdstar)+'\n')
                lout.write('airmass_file = '+str(airmass_file)+'\n')
                lout.close()

            if display_final is True:
                # the final figure to plot
                plt.figure()
                # plt.plot(wfinal, ffinal)
                plt.errorbar(wfinal, ffinal, yerr=efinal)
                plt.xlabel('Wavelength')
                plt.ylabel('Flux')
                plt.title(spec)
                #plot within percentile limits
                plt.ylim( (np.nanpercentile(ffinal,2),
                           np.nanpercentile(ffinal,98)) )
                plt.show()
    return
