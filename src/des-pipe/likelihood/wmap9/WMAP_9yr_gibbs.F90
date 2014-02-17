Module wmap_gibbs

    Use healpix_types
    Use br_mod_dist
    Use wmap_options
    Use wmap_util

    Implicit None

    Private
    Real(dp), Dimension(:), Allocatable :: cl_tt_fiducial, cl_tt_dummy
    Logical, Parameter :: gibbs_verbose = .False.

    Public :: setup_for_tt_gibbs, compute_tt_gibbslike

Contains
    Subroutine setup_for_tt_gibbs()
        Integer(i4b) :: ell_max, num_chains, num_samples, ell_min, ell, i
        Integer(i4b) :: lun
        Real(sp), Dimension(:, :, :), Pointer :: sigmas
        Real(dp) :: ln_likelihood, dummy_te, dummy_ee, dummy_bb, dummy_tt
        Character(Len=512) :: filename, gibbs_file

        If ( .Not. use_lowl_TT) Then
            Print *, "Error: use_lowl_TT == .false. in setup_for_tt_gibbs()"
            Print *, "Set use_lowl_TT = .true."
            Stop
        End If

        If ( .Not. use_gibbs) Then
            Print *, "Error: use_gibbs == .false. in setup_for_tt_gibbs()"
            Print *, "Set use_gibbs = .true."
            Stop
        End If

        If (lowl_max .Lt. 2) Then
            Print *, "Error: lowl_max is less than 2."
            Print *, "Raise it to 2 or greater for use with Gibbs sampling."
            Stop
        End If

        !---------------------------------------------------
        ! read in sigmas and initialize br_mod
        !---------------------------------------------------
        gibbs_file = Trim(WMAP_data_dir) // Trim(gibbs_sigma_filename)

        If (gibbs_verbose) Then
            Print *, "------------------------------------------------------------"
            Print *, "Reading from '" // Trim(gibbs_file) // "'"
        End If
        Call get_free_lun(lun)
        Call read_gibbs_chain(Trim(gibbs_file), lun, ell_max, num_chains, num_samples, sigmas)

        If (gibbs_verbose) Then
            Print *, "minval sigmas = ", Minval(sigmas)
            Print *, "maxval sigmas = ", Maxval(sigmas)

            Print *, "Read in sigmas for Gibbs likelihood:"
            Print *, "Maximum ell = ", ell_max
            Print *, "Number of chains = ", num_chains
            Print *, "Number of Gibbs samples = ", num_samples
        End If

        If (lowl_max > ell_max) Then
            Print *, "Error: lowl_max is too large for use with " // Trim(gibbs_file)
            Print *, "lowl_max: ", lowl_max
            Print *, "maximum allowed value: ", ell_max
            Stop
        End If

        If (gibbs_ell_max > ell_max) Then
            Print *, "Error: gibbs_ell_max is too large for use with " // Trim(gibbs_file)
            Print *, "gibbs_ell_max: ", gibbs_ell_max
            Print *, "maximum allowed value: ", ell_max
            Stop
        End If

        If (gibbs_verbose) Then
            Print *, "Using values of ell from 2 to ", gibbs_ell_max
            Print *, "Using Gibbs samples from ", gibbs_first_iteration, "to", gibbs_last_iteration
            Print *, "in steps of ", gibbs_skip
            Print *, "------------------------------------------------------------"
        End If

        ell_min = ttmin
        Call initialize_br_mod(ell_min, sigmas(ttmin:gibbs_ell_max, 1:num_chains, &
            & gibbs_first_iteration:gibbs_last_iteration:gibbs_skip))

        !---------------------------------------------------
        ! read in Cls
        !---------------------------------------------------
        Allocate(cl_tt_fiducial(ttmin:ttmax))
        Allocate(cl_tt_dummy(ttmin:gibbs_ell_max))

        filename = Trim(WMAP_data_dir) // 'lowlT/gibbs/test_cls.dat'

        If (gibbs_verbose) Then
            Write(*,*) "Reading in Cls from: ", Trim(filename)
        End If
        Call get_free_lun(lun)
        Open(Unit=lun, File=filename, Action='read', Status='old')
        !Do ell = ttmin, ttmax
        Do ell = 2, ttmax
            !Read(lun,*) i, cl_tt_fiducial(ell), dummy_te, dummy_ee, dummy_bb
            Read(lun,*) i, dummy_tt, dummy_te, dummy_ee, dummy_bb
            If (i .Ne. ell) Then
                Print *, "Read in ", i, " was expecting ", ell
                Print *, "Error with file format of: " // Trim(filename)
                Stop
            End If
            If (ell .Ge. ttmin) cl_tt_fiducial(ell) = dummy_tt
        End Do
        Close(lun)

        !---------------------------------------------------
        ! Initialize br_mod with default spectrum
        !---------------------------------------------------
        cl_tt_dummy(ttmin:gibbs_ell_max) = cl_tt_fiducial(ttmin:gibbs_ell_max)
        Call compute_br_estimator(cl_tt_dummy, ln_likelihood)
        If (gibbs_verbose) Then
            Print *, "Initialized log likelihood = ", ln_likelihood
        End If

    End Subroutine setup_for_tt_gibbs


    ! compute_br_estimator() returns the natural logarithm of the
    ! likelihood(plus some arbitrary constant).  The code expects
    ! the negative of this quantity(= chisquared/2)
    Subroutine compute_tt_gibbslike(cl_in, like)
        Real(dp), Dimension(2:), Intent(In) :: cl_in
        Real(dp) :: like
#ifdef TIMING
        Call wmap_timing_start('compute_tt_gibbslike')
#endif
        If (gibbs_ell_max .Gt. lowl_max) Then
            cl_tt_dummy(lowl_max+1:gibbs_ell_max) = cl_tt_fiducial(lowl_max+1:gibbs_ell_max)
        End If
        cl_tt_dummy(ttmin:lowl_max) = cl_in(ttmin:lowl_max)
        Call compute_br_estimator(cl_tt_dummy, like)
        like = - like
#ifdef TIMING
        Call wmap_timing_end()
#endif
    End Subroutine compute_tt_gibbslike

End Module wmap_gibbs
