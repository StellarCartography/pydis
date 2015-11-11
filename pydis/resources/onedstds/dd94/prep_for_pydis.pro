function flux2mag,wave,flux
  zeropt=48.60
  c = 2.99792458e18 
  mag = -2.5 * alog10(flux / (c / wave^2.0)) - zeropt
  return,mag
end
    

pro prep_for_pydis
  ; prepping spectral standards for PyDIS from Danks & Dennefeld (1994)
  ; data from this paper: http://adsabs.harvard.edu/abs/1994PASP..106..382D
  ; available online: http://vizier.cfa.harvard.edu/viz-bin/Cat?III/179

  
  spawn,'ls *.fit > im.lis'
  readcol,'im.lis', f='(A)', files

  for i = 0l,n_elements(files)-1 do begin
     file = files[i]
     
     t = mrdfits(file, 0, hdr, /silent)
     w = findgen(n_elements(t)) * sxpar(hdr, 'CDELT1') + sxpar(hdr, 'CRVAL1')
     
     f = (t*sxpar(hdr,'BSCALE')+sxpar(hdr,'BZERO')) * 10d-13

     ; things get noisy past here
     remove,where(w gt 9600), t,w,f

     ; what resolution to down-sample at
     binsz = 50.0               ; angstroms
     wout = findgen((max(w)-min(w))/binsz)*binsz+min(w)
     
     ; interpolate on to the down-sampled wavelength range
     fout = interpol(f,w,wout)
     
     ;plot,w,f
     ;oplot,wout,fout,psym=4
     
     bout = fltarr(n_elements(fout))+binsz
     mout = flux2mag(wout,fout)

     mout1 = strtrim(string(mout,f='(F6.2)'),2)
     wout1 = strtrim(string(wout,f='(F6.0)'),2)
     bout1 = strtrim(string(bout,f='(F6.0)'),2)
     
     forprint,comm=file, wout1,mout1,bout1, $
              textout='dd94/'+strmid(file, 0, strpos(file,'.fit'))+'.dat',$
              f='(A," ",A," ",A)'
              
     
     ;plot,wout,mout
  endfor
  stop
  return
end
