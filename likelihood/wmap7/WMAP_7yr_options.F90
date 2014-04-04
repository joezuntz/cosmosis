! ===========================================================================
MODULE WMAP_OPTIONS

! This module contains the options in the likelihood code
!
! ===========================================================================

!---------------------------------------------------
! location of input data
! ---------------------------------------------------
#ifdef WMAP_DATA_DIR	 
  character(len=*), parameter :: WMAP_data_dir =&
 	WMAP_DATA_DIR
#else
  character(len=*), parameter :: WMAP_data_dir = './data/'
#endif




!---------------------------------------------------
! likelihood terms from WMAP
!---------------------------------------------------
#ifdef USE_HIGHELL_TB
  integer,parameter   :: num_WMAP   = 10   ! number of individual chi2 terms in likelihood
#else
  integer,parameter   :: num_WMAP   = 8    ! number of individual chi2 terms in likelihood
#endif

  integer,parameter   :: ttlike     = 1    ! master tttt chisq flag
  integer,parameter   :: ttlowllike = 2    ! low tttt chisq flag
  integer,parameter   :: ttlowldet  = 3    ! low tttt determinant flag
  integer,parameter   :: beamlike   = 4    ! beam/pt source correction to tttt chisq flag
  integer,parameter   :: telike     = 5    ! master tete chisq flag
  integer,parameter   :: tedet      = 6    ! master tete determinant flag
  integer,parameter   :: lowllike   = 7    ! TE/EE/BB lowl chisq flag
  integer,parameter   :: lowldet    = 8    ! TE/EE/BB lowl determinant flag
  integer,parameter   :: tblike     = 9    ! master tbtb chisq flag
  integer,parameter   :: tbdet      = 10   ! master tbtb determinant flag

!---------------------------------------------------
! l range to be used in the likelihood code
! change these to consider a more limited l range in TTTT and TETE
!---------------------------------------------------
  integer :: ttmax                = 1200 ! must be l.le.1200 
  integer :: ttmin                = 2    ! must be l.ge.2 
  integer :: temax                = 800  ! must be l.le.800 
  integer :: temin                = 2    ! must be l.ge.2 

!---------------------------------------------------
! various likelihood options
! change these to include/ exclude various likelihood aspects
!---------------------------------------------------
  logical :: use_lowl_TT          = .true. ! include TT pixel likelihood, for l<=lowl_max
  logical :: use_lowl_pol         = .true. ! include TE,EE,BB pixel likelihood for l<24
  logical :: use_TT               = .true. ! include MASTER TT in likelihood
  logical :: use_TT_beam_ptsrc    = .true. ! include beam/ptsrc errors
  logical :: use_TE               = .true. ! include MASTER TE in likelihood

  logical :: use_gibbs_pol_cleaning = .false.

!---------------------------------------------------
! *** AN IMPORTANT CHANGE WITH REGARD TO THE TT LIKELIHOOD ***
!---------------------------------------------------
! There are two options to choose from for evaluating the low-l temperature
! likelihood. Both options produce the same results.
!
! (1) The direct evaluation of likelihood in pixel space using a resolution 4 temperature map.
! (2) The Gibbs sampling.
!
! The option (2) is much faster to evaluate than the option (1).
!
! To use (1), set "use_gibbs = .false." and "lowl_max = 30".
! To use (2), set "use_gibbs = .true." and "lowl_max = 32".
!
! Note that the resolution 3 option for (1) has been disabled.
!
  logical :: use_gibbs = .true.

!---------------------------------------------------
! (1) Pixel likelihood
!---------------------------------------------------
  integer :: lowl_tt_res          = 4      ! TT map resolution
  integer :: lowl_max             = 32     ! use low l TT code 2<l<lowl_max

!---------------------------------------------------
! (2) Gibbs likelihood
!---------------------------------------------------
! For using different sections of the sigmaElls file,
! adjust gibbs_first_iteration, gibbs_last_iteration,
! and gibbs_skip.
!
! For a 50,000 Gibbs sample file, it may be useful to set
! gibbs_first_iteration = 100
! gibbs_last_iteration = 25000
! gibbs_skip = 3
! for one parameter run (to use every third value from the first half
! (approximately) of the file), and
! gibbs_first_iteration = 25100
! gibbs_last_iteration = 50000
! gibbs_skip = 3
! for another parameter run, to use the second half of the file (every third value).
!
! To get really fast (possibly inaccurate) likelihoods,
! set gibbs_skip to be ~ 0.01 * (gibbs_last_iteration - gibbs_first_iteration)
!
! gibbs_first_iteration must be >= 1

character(len=*), parameter :: gibbs_sigma_filename = WMAP_data_dir//'lowlT/gibbs/sigmaEllsHkeChu_run4_kq85.fits'
integer :: gibbs_first_iteration = 10
integer :: gibbs_last_iteration = 120000
integer :: gibbs_skip = 2

! The sum in the BR estimator goes up to this value:
integer :: gibbs_ell_max = 32 

!---------------------------------------------------
! ln(det) offsets
!---------------------------------------------------
! The value of ln(L) returned by the likelihood code is renormalized
! by subtracting off a constant offset:
!
!   -2ln(L) = chi^2 + ln_det_C - ln_det_C_f
!
! The value of the offset, ln_det_C_f,  is the sum of the determinant
! contributions to -2ln(L) computed for the CMB spectrum in
! data/test_cls_v3.dat:
!
!   ln_det_C_f = tt_pixlike_lndet_offset(lowl_tt_res)
!              + teeebb_pixlike_lndet_offset
!              + te_lndet_offset
!   
  double precision, parameter :: tt_pixlike_lndet_offset(4:4) &
#ifdef FASTERTT
		= (/ 5024.741512d0 /)
#else
		= (/ -29677.056620d0 /)
#endif
  double precision, parameter :: teeebb_pixlike_lndet_offset &
		= 16078.083180d0
  double precision, parameter :: te_lndet_offset &
		= 3584.277805d0
  double precision, parameter :: tb_lndet_offset &
		= 3598.152208d0

contains

  subroutine wmap_print_options()
    print *, "-----------------------------------------------------"
    print *, "WMAP_data_dir = ", trim(WMAP_data_dir) 
    print *, ""
    print *, "ttmax = ", ttmax 
    print *, "ttmin = ", ttmin 
    print *, "temax = ", temax
    print *, "temin = ", temin
    print *, ""
    print *, "use_lowl_TT =       ", use_lowl_TT       
    print *, "use_lowl_pol =      ", use_lowl_pol         
    print *, "use_TT =            ", use_TT
    print *, "use_TT_beam_ptsrc = ", use_TT_beam_ptsrc
    print *, "use_TE =            ", use_TE
    print *, ""
    print *, "lowl_tt_res = ", lowl_tt_res
    print *, "lowl_max =    ", lowl_max
    print *, ""
    print *, "tt_pixlike_lndet_offset =     ", tt_pixlike_lndet_offset
    print *, "teeebb_pixlike_lndet_offset = ", teeebb_pixlike_lndet_offset
    print *, "te_lndet_offset =             ", te_lndet_offset
#ifdef USE_HIGHELL_TB
    print *, "tb_lndet_offset =             ", tb_lndet_offset
#endif
    print *, ""
    print *, "use_gibbs = ", use_gibbs
    print *, "gibbs_sigma_filename = ", trim(gibbs_sigma_filename)
    print *, "gibbs_first_iteration = ", gibbs_first_iteration
    print *, "gibbs_last_iteration =  ", gibbs_last_iteration
    print *, "gibbs_skip =            ", gibbs_skip
    print *, "gibbs_ell_max =         ", gibbs_ell_max
    print *, "-----------------------------------------------------"
  end subroutine wmap_print_options
  
END MODULE WMAP_OPTIONS
