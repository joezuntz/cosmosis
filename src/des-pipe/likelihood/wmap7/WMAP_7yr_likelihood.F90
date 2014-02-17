! ===========================================================================
MODULE wmap_likelihood_7yr

! This code is the central likelihood routine from which the other subroutines
! are called.
!
! Parameters are defined in associated WMAP_options module
!
! This code was a collaborative effort by the following WMAP Team Members:
! R. Bean
! J. Dunkley
! E. Komatsu
! D. Larson
! M. Nolta
! H. Peiris
! L. Verde
! D. Spergel
!
! E. Komatsu, March 5, 2009
! -- Changed the orders of do-loops for a better performance 
!    [thanks to Raphael Flauger]
! ===========================================================================

  USE WMAP_options
  use wmap_tt_beam_ptsrc_chisq
#ifdef USE_LOWELL_TBEB
  use WMAP_tetbeebbeb_lowl
#else
  use WMAP_teeebb_lowl
#endif
  use wmap_tlike
  use wmap_gibbs

  character(len=*), parameter, public :: wmap_likelihood_version='v4.1'

  INTEGER, PARAMETER :: tt_lmax =1200 
  INTEGER, PARAMETER :: te_lmax = 800
  logical :: initialise_pass2=.true.

  REAL(8) :: R_off_tete(2:te_lmax, 2:te_lmax)
  REAL(8) :: R_off_tttt(2:tt_lmax, 2:tt_lmax)
  REAL(8) :: off_log_curv(2:tt_lmax, 2:tt_lmax)
  REAL(8) :: epsilon(2:tt_lmax,2:tt_lmax)

  REAL(8) :: cltt_dat(2:tt_lmax)
  REAL(8) :: clte_dat(2:te_lmax) 
  REAL(8) :: cltb_dat(2:te_lmax) 
  REAL(8) :: ntt(2:tt_lmax)    
  REAL(8) :: ntt_te(2:te_lmax)  
  REAL(8) :: nee_te(2:te_lmax)
  REAL(8) :: fskytt(2:tt_lmax)
  REAL(8) :: fskyte(2:te_lmax)
  REAL(8) :: ntt_tb(2:te_lmax)  
  REAL(8) :: nbb_tb(2:te_lmax)
  REAL(8) :: fskytb(2:te_lmax)

PRIVATE

	public :: wmap_likelihood_init
	public :: wmap_likelihood_compute
	public :: wmap_likelihood_dof

contains

! ===========================================================================
SUBROUTINE wmap_likelihood_init 
! ===========================================================================

  use wmap_util

  IMPLICIT NONE

  INTEGER  :: l,ll
  integer  :: il,ill,i,j
  REAL(8)  :: dummy
  CHARACTER(LEN=240) :: ttfilename,tefilename,tbfilename,ttofffilename,teofffilename
  LOGICAL  :: good
  integer ::  lun
  real :: rtmp, rtmp2

#ifdef TIMING
	call wmap_timing_start( 'wmap_likelihood_init' )
#endif

  print *, 'Initializing WMAP likelihood, version '//wmap_likelihood_version

!-----------------------------------------------
! initialise beam uncertainties
!-----------------------------------------------

  if ( use_TT_beam_ptsrc ) call init_tt_beam_and_ptsrc_chisq( 2, tt_lmax )

!-----------------------------------------------
! initialise low l codes
!-----------------------------------------------
  if(use_lowl_pol)then
        !write(*,*) use_lowl_pol, 'using low ell'
#ifdef USE_LOWELL_TBEB
        call tetbeebbeb_lowl_like_setup
#else
        call teeebb_lowl_like_setup
#endif
 endif

 if(use_lowl_TT) then
     if (use_gibbs) then
        call setup_for_tt_gibbs()
     else
        call setup_for_TT_exact(lowl_max)
     endif
  endif

!-----------------------------------------------
! set file names
!-----------------------------------------------

  ttfilename = trim(WMAP_data_dir)//'highl/wmap_likelihood_inputs_tt.p4v6.dat'
  tefilename = trim(WMAP_data_dir)//'highl/wmap_likelihood_inputs_te_recalibrated.p4v6.dat'
  tbfilename = trim(WMAP_data_dir)//'highl/wmap_likelihood_inputs_tb_recalibrated.p4v6.dat'
  ttofffilename = trim(WMAP_data_dir)//'highl/wmap_likelihood_inputs_tt_offdiag.p4v4.dat'
  teofffilename = trim(WMAP_data_dir)//'highl/wmap_likelihood_inputs_te_offdiag.p4v4.dat'

!-----------------------------------------------
! get TT diag
!-----------------------------------------------

  inquire(file=ttfilename,exist = good)
  if(.not.good)then
     write(*,*) 'cant find', trim(ttfilename)
     stop
  endif
  call get_free_lun( lun )
  open(unit=lun,file=ttfilename,form='formatted',status='unknown',action='read')

  do l=2,tt_lmax
     read(lun,*) dummy,cltt_dat(l),ntt(l),fskytt(l)
  enddo
  close(lun)

!-----------------------------------------------
! get TE diag
!-----------------------------------------------

  inquire(file=tefilename,exist = good)
  if(.not.good)then
     write(*,*) 'cant find', trim(tefilename)
     stop
  endif

  call get_free_lun( lun )
  open(unit=lun,file=tefilename,form='formatted',status='unknown',action='read')

  do l=2,te_lmax
     read(lun,*) ll,clte_dat(l), dummy, ntt_te(l),nee_te(l),fskyte(l)
  enddo
  close(lun)

!-----------------------------------------------
! get TB diag
!-----------------------------------------------

#ifdef USE_HIGHELL_TB
  inquire(file=tbfilename,exist = good)
  if(.not.good)then
     write(*,*) 'cant find', trim(tbfilename)
     stop
  endif

  call get_free_lun( lun )
  open(unit=lun,file=tbfilename,form='formatted',status='unknown',action='read')

  do l=2,te_lmax
     read(lun,*) ll,cltb_dat(l), dummy, ntt_tb(l),nbb_tb(l),fskytb(l)
  enddo
  close(lun)
#endif

!-----------------------------------------------
! get TT off diag
!-----------------------------------------------

  R_off_tttt = 0.

  inquire(file=ttofffilename,exist = good)

  if(.not.good)then
     write(*,*) 'cant find', trim(ttofffilename)
     stop
  endif

  call get_free_lun( lun )
  open(unit=lun,file=ttofffilename,form='formatted',status='unknown',action='read')
! --- left as is --- could be written better if the files are changed
  do i=2,tt_lmax
     do j=i+1,tt_lmax

        read(lun,*) l,ll,epsilon(i,j),R_off_tttt(i,j)
        R_off_tttt(j,i)=R_off_tttt(i,j)
        epsilon(j,i)=epsilon(i,j)

        if(i.ne.l.or.j.ne.ll)then
           write(*,*)"tt off file misread",i,j,l,ll
           stop
        endif

     enddo
  enddo
! ---
  close(lun)

!-----------------------------------------------
! get TE off diag
!----------------------------------------------

  inquire(file=teofffilename,exist = good)

  if(.not.good)then
     write(*,*) 'cant find', trim(teofffilename)
     stop
  endif

  call get_free_lun( lun )
  open(unit=lun,file=teofffilename,form='formatted',status='unknown',action='read')

! --- left as is --- could be written better if the files are changed
  do i=2,te_lmax-1
     do j=i+1,te_lmax

        read(lun,*) l ,ll,dummy

        if(l.le.te_lmax.and.ll.le.te_lmax)then
           R_off_tete(i,j)=dummy
           R_off_tete(j,i)=R_off_tete(i,j)
        endif

        if(l.ne.i.or.ll.ne.j)then
           write(*,*)"TE off diag misread i,j,l,ll",i,j,l,ll
           stop
        endif

     enddo
  enddo
  close(lun)

  initialise_pass2 = .false.

#ifdef TIMING
	call wmap_timing_end()
#endif

END SUBROUTINE

  subroutine wmap_likelihood_dof( tt_npix, teeebb_npix )

	integer, intent(out) :: tt_npix, teeebb_npix

	tt_npix = tt_pixlike_dof()
#ifdef USE_LOWELL_TBEB
	teeebb_npix = tetbeebbeb_pixlike_dof()
#else
	teeebb_npix = teeebb_pixlike_dof()
#endif

  end subroutine

! ===========================================================================
#ifdef USE_LOWELL_TBEB
SUBROUTINE wmap_likelihood_compute(cltt,clte,cltb,clee,cleb,clbb,like)
#elif USE_HIGHELL_TB
SUBROUTINE wmap_likelihood_compute(cltt,clte,cltb,clee,cleb,clbb,like)
#else 
SUBROUTINE wmap_likelihood_compute(cltt,clte,clee,clbb,like)
#endif
! ===========================================================================

  use wmap_util

  IMPLICIT NONE
  REAL(8), intent(in) :: cltt(2:*), clte(2:*), clee(2:*), clbb(2:*)
#ifdef USE_LOWELL_TBEB
  REAL(8), intent(in) :: cltb(2:*), cleb(2:*)
  real(8) :: ee2, bb2, eb2
#elif USE_HIGHELL_TB
  REAL(8), intent(in) :: cltb(2:*), cleb(2:*)
  real(8) :: ee2, bb2, eb2
#endif
  REAL(8),intent(out) :: like(num_WMAP)
  INTEGER :: il, ill
  REAL(8) :: dlnlike_tot, dlnlike ,dlnlike_beam, ln_det_TETE, ln_det_TBTB
  !REAL(8) :: fisher(2:tt_lmax, 2:tt_lmax)
  REAL(8), allocatable :: fisher(:,:)
  REAL(8) :: tttt(2:tt_lmax),tete(2:te_lmax),ttte(2:te_lmax),tbtb(2:te_lmax)
  REAL(8) :: z(2:tt_lmax), zbar(2:tt_lmax)
  REAL(8) :: cltt_temp(2:tt_lmax)
  real, allocatable,dimension(:) :: lowl_cl
  integer :: tt_hi_l_start, te_hi_l_start
  REAL(8) :: correlation_coefficient_cl,tol=1d-10

#ifdef TIMING
	call wmap_timing_start( 'wmap_likelihood_compute' )
#endif

  call wmap_likelihood_error_init

  if(initialise_pass2)then
     call wmap_likelihood_init
  endif

  Like = 0.d0

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! allocate memory

  allocate( fisher(2:tt_lmax,2:tt_lmax) )

  !--------------------------------------
  ! Are cltt, clte, and clee consistent?
  !--------------------------------------
  do il = 2,ttmax
    if ( abs(clte(il)) > 0d0 ) then
      correlation_coefficient_cl = abs(clte(il))/sqrt(cltt(il)*clee(il))
      if( correlation_coefficient_cl-1d0 > tol ) then
         call wmap_likelihood_error( &
           'unphysical input: TE/sqrt(TT*EE) > 1 at l=', il )
         return
      end if
    end if
  enddo

#ifdef USE_HIGHELL_TB
  do il = 2,ttmax
    if ( abs(cltb(il)) > 0d0 ) then
      correlation_coefficient_cl = abs(cltb(il))/sqrt(cltt(il)*clbb(il))
      if( correlation_coefficient_cl-1d0 > tol )then
         call wmap_likelihood_error( &
           'unphysical input: TB/sqrt(TT*BB) > 1 at l=', il )
         return
      end if
    end if
  enddo
#endif

#ifdef USE_LOWELL_TBEB
  do il = 2,ttmax
    if ( abs(cltb(il)) > 0d0 ) then
      correlation_coefficient_cl = abs(cltb(il))/sqrt(cltt(il)*clbb(il))
      if( correlation_coefficient_cl-1d0 > tol )then
         call wmap_likelihood_error( &
           'unphysical input: TB/sqrt(TT*BB) > 1 at l=', il )
         return
      end if
    end if

    if ( abs(cleb(il)) > 0d0 ) then
      correlation_coefficient_cl = abs(cleb(il))/sqrt(clee(il)*clbb(il))
      if ( correlation_coefficient_cl-1d0 > tol ) then
         call wmap_likelihood_error( &
           'unphysical input: EB/sqrt(EE*BB) > 1 at l=', il )
print*,correlation_coefficient_cl
print*,'unphysical input: EB/sqrt(EE*BB) > 1 at l=', il
         return
      end if
    end if

    if ( abs(cleb(il)) > 0d0 ) then
      ee2 = clee(il) - clte(il)**2/cltt(il)
      bb2 = clbb(il) - cltb(il)**2/cltt(il)
      eb2 = cleb(il) - clte(il)*cltb(il)/cltt(il)
      if ( abs(eb2)/sqrt(ee2*bb2)-1d0 > tol ) then
        call wmap_likelihood_error( &
          "unphysical input: EB'/sqrt(EE'*BB') > 1 at l=", il )
        return
      end if
    end if
  enddo
#endif

  cltt_temp(2:ttmax)=cltt(2:ttmax)

  !---------------------------------------------------------------------------
  ! low l TT likelihood
  !---------------------------------------------------------------------------
  if(use_lowl_TT)then
     if (use_gibbs) then
        call compute_tt_gibbslike(cltt_temp(2:lowl_max),Like(ttlowllike))
        Like(ttlowldet) = 0.0
     else
        allocate(lowl_cl(2:lowl_max))
        lowl_cl(2:lowl_max) = cltt_temp(2:lowl_max)
        call compute_tt_pixlike(lowl_max,lowl_cl,Like(ttlowllike),Like(ttlowldet))
        deallocate(lowl_cl)
     endif
     tt_hi_l_start = lowl_max+1
  else
     tt_hi_l_start = ttmin
  endif

  !---------------------------------------------------------------------------
  ! low l TE/EE/BB likelihood
  !---------------------------------------------------------------------------
  if(use_lowl_pol)then
#ifdef USE_LOWELL_TBEB
     call tetbeebbeb_lowl_likelihood(23,cltt_temp,clte,cltb,clee,clbb,cleb,Like(lowllike),Like(lowldet))
#else
     call teeebb_lowl_likelihood(23,cltt_temp,clte,clee,clbb,Like(lowllike),Like(lowldet))
#endif
     te_hi_l_start = 24
  else
     te_hi_l_start = temin
  endif

  !---------------------------------------------------------------------------
  ! TT and TE covariance terms
  !---------------------------------------------------------------------------
  tttt=0.d0
  tete=0.d0
  ttte=0.d0

  if(use_TT.or.use_TE)then

     do il=ttmin,ttmax
        tttt(il)= 2.*(cltt_temp(il)+ntt(il))**2/((2.0d0*dble(il)+1.0d0)*fskytt(il)**2.)
     enddo

     do il=temin,temax
        tete(il)= ((cltt_temp(il)+ntt_te(il))*(clee(il)+nee_te(il))+clte(il)**2)/&
             ((2.0d0*dble(il)+1.0d0)*fskyte(il)**2.)
        ttte(il)= 2*((cltt_temp(il)+ntt(il))*clte(il))/&
             ((2.0d0*dble(il)+1.0d0)*fskyte(il)*fskytt(il))
     enddo

  endif

  !---------------------------------------------------------------------------
  ! TB covariance terms (diagonal only)
  !---------------------------------------------------------------------------
  tbtb=0.d0

#ifdef USE_HIGHELL_TB
     do il=temin,temax
        tbtb(il)= ((cltt_temp(il)+ntt_tb(il))*(clbb(il)+nbb_tb(il))+cltb(il)**2)/&
             ((2.0d0*dble(il)+1.0d0)*fskytb(il)**2.)
     enddo
#endif

  !---------------------------------------------------------------------------
  !TTTT MASTER likelihood
  !---------------------------------------------------------------------------
  if(use_TT)then

     fisher=0.d0
     off_log_curv=0.d0

     do il=ttmin,ttmax
        z(il)=dlog(cltt_dat(il)+ntt(il))
        zbar(il)=dlog(cltt_temp(il)+ntt(il))
     enddo
! --- original
!     do il=ttmin,ttmax
!        do ill=il,ttmax
! --- RF
     do ill=ttmin,ttmax
        do il=ttmin,ill
! ---
           if(il.eq.ill)then
              if(il.le.te_lmax)then
                 fisher(il,ill) = tete(il)/(tttt(il)*tete(il)-ttte(il)**2)
              else
                 fisher(il,ill) = 1.d0/tttt(il)
              endif
           else
              fisher(il,ill) = -R_off_tttt(il,ill)/sqrt(tttt(il)*tttt(ill))&
                   +epsilon(il,ill)/(tttt(il)*tttt(ill))
           endif
           off_log_curv(il,ill)=(cltt_temp(il)+ntt(il))*fisher(il, ill)*&
                & (cltt_temp(ill)+ntt(ill))
        end do
     end do

     dlnlike_tot = 0.d0

! --- original
!     do il=ttmin,ttmax
!        do ill=il,ttmax
! --- RF 
     do ill=ttmin,ttmax
        do il=ttmin,ill
! ---
           dlnlike = 2.d0/3.d0*(z(il)-zbar(il))*off_log_curv(il,ill)*(z(ill)-zbar(ill))+&
                1.d0/3.d0*(cltt_temp(il)-cltt_dat(il))* &
                fisher(il,ill)*(cltt_temp(ill)-cltt_dat(ill))

           if(il.ge.tt_hi_l_start.or.ill.ge.tt_hi_l_start)then
              if(il.eq.ill)then
                 dlnlike_tot = dlnlike_tot + dlnlike
              else
                 dlnlike_tot = dlnlike_tot + dlnlike*2
              endif
           endif

        end do
     end do
     
     Like(ttlike) = dlnlike_tot/2.d0

  endif
  !---------------------------------------------------------------------------
  !TTTT Beam and point source correction
  !---------------------------------------------------------------------------
  if(use_TT .and. use_TT_beam_ptsrc)then
	dlnlike_beam = compute_tt_beam_and_ptsrc_chisq( ttmin, ttmax, &
		cltt_temp, cltt_dat, ntt, fisher, z, zbar )
     if ( abs(dlnlike_beam) >= dlnlike_tot/4d0 ) then
        call wmap_likelihood_warning( 'beam correction invalid', 0 )
	dlnlike_beam = 0d0
     end if
     like(beamlike) = dlnlike_beam/2.d0
  endif

  !---------------------------------------------------------------------------
  !TETE MASTER likelihood
  !---------------------------------------------------------------------------
  if(use_TE)then
     ln_det_TETE=0.0d0
     dlnlike_tot=0.d0
     fisher=0.d0     
! --- original
!     do il=temin,temax
!        do ill=il,temax
! --- RF
     do ill=temin,temax
        do il=temin,ill
! ---
           if(il.eq.ill)then
              if(il.le.te_lmax)then
                 fisher(il,ill) = tttt(il)/(tttt(il)*tete(il)-ttte(il)**2)
              else
                 fisher(il,ill) = 1.d0/tete(il)
              endif
           else
              fisher(il,ill) = -R_off_tete(il,ill)/sqrt(tete(il)*tete(ill))
           endif
        enddo
     enddo
! --- original
!     do il=temin,temax
!        do ill=il,temax
! --- RF
     do ill=temin,temax
        do il=temin,ill
! ---
           dlnlike = (clte(il)-clte_dat(il))*fisher(il,ill)* (clte(ill)-clte_dat(ill))
           
           if(il.ge.te_hi_l_start.or.ill.ge.te_hi_l_start)then
              IF(il.eq.ill) then
                 dlnlike_tot=dlnlike_tot+dlnlike
                 ln_det_TETE=ln_det_TETE-Log(fisher(il,ill))
              else
                 dlnlike_tot=dlnlike_tot+dlnlike*2.
              endif
           endif

        end do
     end do
     Like(telike) = dlnlike_tot/2.d0
     
     Like(tedet) = (ln_det_TETE - te_lndet_offset)/2d0
  endif

  !---------------------------------------------------------------------------
  !TBTB MASTER likelihood
  !---------------------------------------------------------------------------
#ifdef USE_HIGHELL_TB
     ln_det_TBTB=0.0d0
     dlnlike_tot=0.d0
     fisher=0.d0

     do il=temin,temax
                 fisher(il,il) = 1.d0/tbtb(il)
     enddo

     do il=temin,temax

           dlnlike = (cltb(il)-cltb_dat(il))**2.*fisher(il,il)

           if(il.ge.te_hi_l_start)then
                 dlnlike_tot=dlnlike_tot+dlnlike
                 ln_det_TBTB=ln_det_TBTB-Log(fisher(il,il))
           endif

     end do
     Like(tblike) = dlnlike_tot/2.d0

     Like(tbdet) = (ln_det_TBTB - tb_lndet_offset)/2d0
print*,ln_det_TBTB
#endif

10 continue

  ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ! deallocate memory

  deallocate( fisher )

#ifdef TIMING
	call wmap_timing_end()
#endif

end SUBROUTINE

END MODULE

