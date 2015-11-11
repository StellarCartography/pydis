pro APOext_conv
  ;+
  ; Convert airmass extinction curve to proper units,
  ; down-sample to acceptable gridding for IRAF use.
  ;
  ; author: James R. A. Davenport (2015)
  ;-

  loadct,39,/silent
    !p.thick=2
    !x.thick=2
    !y.thick=2
    !p.charthick=2.5
    !p.font=0
    !p.charsize=1.6

  ; read in extinction curves from various places 
  readcol,'ctioextinct.dat',wc,mc ; via IRAF
  readcol,'kpnoextinct.dat',wk,mk ; via IRAF
  readcol, 'apoextinct_trans.dat', wa,fa ; via APO website

  ; convert APO transmission -> mag/airmass
  ma = -2.5d0 * alog10(fa)

  ; median down-sample APO data in very coarse bins
  medbin, wa, ma, xout, yout, 450, min(wa), max(wa)

  ; select regions to consider differently
  ;-- in APO curve
  x1a = where(wa ge 3000 and wa lt 6200)
  x2a = where(wa ge 6200 and wa lt 8400)
  x3a = where(wa ge 8400 and wa lt 8700)
  x4a = where(wa ge 8700 and wa lt 10000)
  x5a = where(wa ge 10000)

  ;-- in Kitt Peak curve
  x1k = where(wk lt 6200)
  x2k = where(wk ge 6200 and wk lt 8400)
  x3k = where(wk ge 8400 and wk lt 8700)
  x4k = where(wk ge 8700 and wk lt 10000)
  x5k = where(wk ge 10000)

  wa_out = [wa[x1a], wk[x2k], wa[x3a], wa[x5a]]
  ma_out = [ma[x1a], $
            interpol(yout, xout, wk[x2k]), $
            ma[x3a], $
            ma[x5a]]


  set_plot,'ps'

  device,filename='apoextinction.eps',/encap,/color
  loadct,0,/silent
  plot, wc,mc,psym=10,xtitle='!7wavelength',ytitle='mag / airmass',$
        /ylog,/xsty,xrange=[3300, 11000],thick=4,font=0
  oplot,wk,mk,color=150,psym=10,thick=4
  legend,box=0,['!7CTIO','KPNO'],color=[0,150],/bottom,linestyle=0,charsize=1
  
  loadct,39,/silent
  oplot, wa, ma,color=250,thick=2
  oplot, wa_out, ma_out,color=50,psym=-4,thick=4
  legend,box=0,['!7APO old','APO new'],color=[250,50],linestyle=0,/right,charsize=1
  device,/close

  
  device,filename='apo_ext_problem.eps',/encap,/color
  plot,wa_out,10.0^(0.4*ma_out),/ysty,yrange=[0,3],xrange=[3300,10000],$
       ytitle='Flux correction @ X=1',thick=3
  oplot,wa,10.0^(0.4*fa),linesty=2,color=250,thick=3
  legend,['right','wrong'],linestyle=[0,2],/bottom,box=0,$
         charsize=1,color=[0,250]
  device,/close
  
  
  set_plot,'X'

  openw, lun, 'apoextinct.dat', /get_lun
  printf, lun, '# APO airmass extinction curve, prepped for IRAF'
  printf, lun, '# wavelength, mag/airmass  (J.R.A. Davenport 2015)'
  for i=0L,n_elements(wa_out)-1 do $
     printf, lun, strtrim(string(wa_out[i], f='(F10.2)'),2)+'  ' + $
             strtrim(string(ma_out[i], f='(F10.5)'),2)
  close,lun
  
  
  stop
end
