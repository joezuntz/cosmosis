!===========================================================
program test_likelihood

! RB December '05
!===========================================================

use wmap_likelihood_7yr
use wmap_options
use wmap_util

implicit none

real(8), dimension(:), allocatable :: cl_tt,cl_te,cl_ee,cl_bb
#ifdef USE_LOWELL_TBEB
real(8), dimension(:), allocatable :: cl_tb,cl_eb
#endif
character(LEN=128)          :: filename
real(8)                     :: like(num_WMAP),like_tot,expected_like_tot
integer                     :: lun, l, i, olun
integer :: tt_npix, teeebb_npix

!---------------------------------------------------

print *,""
print *,"WMAP 7-year likelihood test program"
print *,"==================================="
print *,""
print *,"NOTE: This code uses a different CMB spectrum than previous versions."
print *,"      The new spectrum (data/test_cls_v4.dat) is a better fit to the data"
print *,"      than the old one (data/test_cls_v3.dat)."
print *,""
print *,"      As before, a constant offset is now being subtracted from ln(L)."
print *,"      The value of the offset is the sum of the determinant"
print *,"      contributions to ln(L) computed for the CMB spectrum in"
print *,"      data/test_cls_v4.dat, ln_det_C_f:"
print *,""
print *,"        -2ln(L) = chi^2 + ln_det_C - ln_det_C_f"
print *,""

!---------------------------------------------------

allocate( cl_tt(ttmin:ttmax) )
allocate( cl_te(ttmin:ttmax) )
allocate( cl_ee(ttmin:ttmax) )
allocate( cl_bb(ttmin:ttmax) )

cl_bb = 0d0

#ifdef USE_LOWELL_TBEB
allocate( cl_tb(ttmin:ttmax) )
allocate( cl_eb(ttmin:ttmax) )
cl_tb(:) = 0d0
cl_eb(:) = 0d0
#endif

!---------------------------------------------------
! read in Cls
!---------------------------------------------------
filename = trim(WMAP_data_dir)//'test_cls_v4.dat'

write(*,*)"Reading in Cls from: ",trim(filename)
call get_free_lun( lun )
open(unit=lun,file=filename,action='read',status='old')

do l=2,ttmax
   read(lun,*)i,cl_tt(l),cl_ee(l),cl_bb(l),cl_te(l)
enddo

close(lun)

!do l = 2,9
!	cl_te(l) = 0d0
!	cl_ee(l) = 0d0
!end do

!---------------------------------------------------
! put in likelihood options
! see PASS2_options module for the options below
!---------------------------------------------------

use_TT               = .true.
use_TE               = .true. 
use_lowl_TT          = .true.
use_lowl_pol         = .true.

!---------------------------------------------------
! get likelihoods
!---------------------------------------------------
like =0.d0
call wmap_likelihood_init
call wmap_likelihood_dof( tt_npix, teeebb_npix )
#ifdef USE_LOWELL_TBEB
call wmap_likelihood_compute(cl_tt,cl_te,cl_tb,cl_ee,cl_eb,cl_bb,like)
#else
call wmap_likelihood_compute(cl_tt,cl_te,cl_ee,cl_bb,like)
#endif
call wmap_likelihood_error_report

like_tot = sum(like(1:num_WMAP))

!---------------------------------------------------
! write outputs
!---------------------------------------------------
  print 1
  print 2
  print 1
  print 3, 'MASTER TTTT             ', 2*like(ttlike),     ttmax-lowl_max
  print 5, 'Beam/ptsrc TT correction', 2*like(beamlike)
if ( use_gibbs ) then
  print 5, 'low-l TTTT gibbs        ', 2*like(ttlowllike)
else
  print 4, 'low-l TTTT chi2         ', 2*like(ttlowllike), tt_npix
  print 5, 'low-l TTTT det          ', 2*like(ttlowldet)
end if
  print 3, 'MASTER TETE chi2        ', 2*like(telike),     temax-23
  print 5, 'MASTER TETE det         ', 2*like(tedet)
  print 4, 'TT/TE/EE/BB low-l chi2  ', 2*like(lowllike),   teeebb_npix
  print 5, 'TT/TE/EE/BB low-l det   ', 2*like(lowldet)
  print 1
  print 5, 'TOTAL -2ln(L)           ', 2*like_tot
  print 1

  if ( use_gibbs ) then
    !expected_like_tot = 7481.131790d0
    expected_like_tot = 7477.656769d0
  else
#ifdef FASTERTT
    !expected_like_tot = 8257.119952d0
    expected_like_tot = 8253.582175d0
#else
    !expected_like_tot = 11068.324977d0
    expected_like_tot = 11064.787184d0
#endif
  end if

  print '(A,F13.6)', "Expected -2ln(L)         = ", expected_like_tot
  print '(A,F13.6)', "      Difference         = ", 2*like_tot-expected_like_tot
  print 1
  print *, ""
  print *, "Differences on the order of O(0.001) are normal between platforms."
  print *, ""

  stop
1 format ('------------------------------------------------------------------')
2 format ('Breakdown of -2ln(L)')
3 format (A24,' = ',F13.6,' for ',I6,' ls')
4 format (A24,' = ',F13.6,' for ',I6,' pixels')
5 format (A24,' = ',F13.6)
 end program test_likelihood
