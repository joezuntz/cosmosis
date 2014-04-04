module wmap_gibbs

  use healpix_types
  use br_mod_dist
  use wmap_options
  use wmap_util

  implicit none

  private
  real(dp), dimension(:), allocatable :: cl_tt_fiducial, cl_tt_dummy
  logical, parameter :: gibbs_verbose = .false.

  public :: setup_for_tt_gibbs, compute_tt_gibbslike

contains
  subroutine setup_for_tt_gibbs()
    integer(i4b) :: ell_max, num_chains, num_samples, ell_min, ell, i
    integer(i4b) :: lun
    real(sp), dimension(:,:,:), pointer :: sigmas
    real(dp) :: ln_likelihood, dummy_te, dummy_ee, dummy_bb
    character(len=512) :: filename

    if (.not. use_lowl_TT) then
       print *, "Error: use_lowl_TT == .false. in setup_for_tt_gibbs()"
       print *, "Set use_lowl_TT = .true."
       stop
    endif

    if (.not. use_gibbs) then
       print *, "Error: use_gibbs == .false. in setup_for_tt_gibbs()"
       print *, "Set use_gibbs = .true."
       stop
    endif

    if (lowl_max .lt. 2) then
       print *, "Error: lowl_max is less than 2."
       print *, "Raise it to 2 or greater for use with Gibbs sampling."
       stop
    endif
    
    !---------------------------------------------------
    ! read in sigmas and initialize br_mod
    !---------------------------------------------------

    if ( gibbs_verbose ) then
      print *, "------------------------------------------------------------"
      print *, "Reading from '"//trim(gibbs_sigma_filename)//"'"
    end if
    call get_free_lun(lun)
    call read_gibbs_chain(trim(gibbs_sigma_filename), lun, ell_max, &
         & num_chains, num_samples, sigmas)

    if ( gibbs_verbose ) then
      print *, "minval sigmas = ", minval(sigmas)
      print *, "maxval sigmas = ", maxval(sigmas)

      print *, "Read in sigmas for Gibbs likelihood:"
      print *, "Maximum ell = ", ell_max
      print *, "Number of chains = ", num_chains
      print *, "Number of Gibbs samples = ", num_samples
    end if

    if (lowl_max > ell_max) then
       print *, "Error: lowl_max is too large for use with "//trim(gibbs_sigma_filename)
       print *, "lowl_max: ", lowl_max
       print *, "maximum allowed value: ", ell_max
       stop
    endif

    if (gibbs_ell_max > ell_max) then
       print *, "Error: gibbs_ell_max is too large for use with "//trim(gibbs_sigma_filename)
       print *, "gibbs_ell_max: ", gibbs_ell_max
       print *, "maximum allowed value: ", ell_max
       stop
    endif

    if ( gibbs_verbose ) then
      print *, "Using values of ell from 2 to ", gibbs_ell_max
      print *, "Using Gibbs samples from ", gibbs_first_iteration, "to", gibbs_last_iteration
      print *, "in steps of ", gibbs_skip
      print *, "------------------------------------------------------------"
    end if

    ell_min = 2
    call initialize_br_mod(ell_min, sigmas(2:gibbs_ell_max, 1:num_chains, gibbs_first_iteration:gibbs_last_iteration:gibbs_skip))

    !---------------------------------------------------
    ! read in Cls
    !---------------------------------------------------
    allocate(cl_tt_fiducial(ttmin:ttmax))
    allocate(cl_tt_dummy(ttmin:gibbs_ell_max))
   
    filename = trim(WMAP_data_dir)//'lowlT/gibbs/test_cls.dat'

    if ( gibbs_verbose ) then
      write(*,*)"Reading in Cls from: ",trim(filename)
    end if
    call get_free_lun(lun)
    open(unit=lun, file=filename, action='read', status='old')
    do ell = ttmin, ttmax
       read (lun,*) i, cl_tt_fiducial(ell), dummy_te, dummy_ee, dummy_bb
       if (i .ne. ell) then
          print *, "Error with file format of: "//trim(filename)
          stop
       endif
    enddo
    close(lun)

    !---------------------------------------------------
    ! Initialize br_mod with default spectrum
    !---------------------------------------------------
    cl_tt_dummy(ttmin:gibbs_ell_max) = cl_tt_fiducial(ttmin:gibbs_ell_max)
    call compute_br_estimator(cl_tt_dummy, ln_likelihood)
    if ( gibbs_verbose ) then
      print *, "Initialized log likelihood = ", ln_likelihood
    end if

  end subroutine setup_for_tt_gibbs
  

  ! compute_br_estimator() returns the natural logarithm of the 
  ! likelihood (plus some arbitrary constant).  The code expects
  ! the negative of this quantity (= chisquared/2)
  subroutine compute_tt_gibbslike(cl_in, like)
    real(dp), dimension(2:), intent(in) :: cl_in
    real(dp) :: like
#ifdef TIMING
    call wmap_timing_start('compute_tt_gibbslike')
#endif
    if (gibbs_ell_max .gt. lowl_max) then
       cl_tt_dummy(lowl_max+1:gibbs_ell_max) = cl_tt_fiducial(lowl_max+1:gibbs_ell_max)
    endif
    cl_tt_dummy(ttmin:lowl_max) = cl_in(ttmin:lowl_max)
    call compute_br_estimator(cl_tt_dummy, like)
    like = -like
#ifdef TIMING
    call wmap_timing_end()
#endif
  end subroutine compute_tt_gibbslike
  
end module wmap_gibbs
