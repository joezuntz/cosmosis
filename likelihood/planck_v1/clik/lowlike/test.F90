!===========================================================
program test_likelihood

!===========================================================

use planck_likelihood
use planck_options

implicit none

real(8), dimension(:), allocatable :: cl_tt,cl_te,cl_ee,cl_bb
character(LEN=128)          :: filename
real(8)                     :: like(num_pl),like_tot,expected_like_tot
integer                     :: lun, l, i, olun
integer :: tt_npix, teeebb_npix
INTEGER :: clock_start,clock_end,clock_rate
REAL :: elapsed_time

!---------------------------------------------------

print *,""
print *,"Planck low-like test program"
print *,"==================================="
print *,""
!---------------------------------------------------

allocate( cl_tt(ttmin:ttmax) )
allocate( cl_te(ttmin:ttmax) )
allocate( cl_ee(ttmin:ttmax) )
allocate( cl_bb(ttmin:ttmax) )

cl_bb = 0d0

!---------------------------------------------------
! read in Cls
!---------------------------------------------------
filename = trim(Planck_data_dir)//'test_cls_v5.dat'

write(*,*)"Reading in Cls from: ",trim(filename)
call get_free_lun( lun )
open(unit=lun,file=filename,action='read',status='old')

do l=2,ttmax
   read(lun,*)i,cl_tt(l),cl_ee(l),cl_bb(l),cl_te(l)
enddo

close(lun)

!---------------------------------------------------
! put in likelihood options
!---------------------------------------------------

use_lowl_pol         = .true.

!---------------------------------------------------
! get likelihoods
!---------------------------------------------------
like =0.d0
call planck_lowlike_init

CALL SYSTEM_CLOCK(COUNT_RATE=clock_rate) ! Find the rate
CALL SYSTEM_CLOCK(COUNT=clock_start) ! Start timing
do i=1,100
call planck_lowlike_compute(cl_tt,cl_te,cl_ee,cl_bb,like)
enddo
CALL SYSTEM_CLOCK(COUNT=clock_end) ! Stop timing
  ! Calculate the elapsed time in seconds:
  elapsed_time=REAL((clock_end-clock_start)/clock_rate)

 write(*,*) real(elapsed_time), 'secs'


like_tot = sum(like(1:num_pl))

!---------------------------------------------------
! write outputs
!---------------------------------------------------
  print 1
  print 2
  print 1

  print 3, 'TT/TE/EE/BB low-l chi2  ', 2*like(lowllike)
  print 4, 'TT/TE/EE/BB low-l det   ', 2*like(lowldet)
  print 1
  print 4, 'TOTAL -2ln(L)           ', 2*like_tot
  print 1

  expected_like_tot = 1925.697629d0
 
  print '(A,F13.6)', "Expected -2ln(L)         = ", expected_like_tot
  print '(A,F13.6)', "      Difference         = ", 2*like_tot-expected_like_tot
  print 1
  print *, ""
  print *, "Differences on the order of O(0.001) are normal between platforms."
  print *, ""

  stop
1 format ('------------------------------------------------------------------')
2 format ('Breakdown of -2ln(L)')
3 format (A24,' = ',F13.6,' for ',I6,' pixels')
4 format (A24,' = ',F13.6)
 end program test_likelihood
