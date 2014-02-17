! ===========================================================================
Module wmap_likelihood_9yr
    
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

    Use WMAP_options
    Use wmap_tt_beam_ptsrc_chisq
#ifdef USE_LOWELL_TBEB
    Use WMAP_tetbeebbeb_lowl
#else
    Use WMAP_teeebb_lowl
#endif
    Use wmap_tlike
    Use wmap_gibbs

    Character(Len=*), Parameter, Public :: wmap_likelihood_version = 'v5'

    Integer, Parameter :: tt_lmax = 1200
    Integer, Parameter :: te_lmax = 800
    Logical :: initialise_pass2 = .True.

    Real(8) :: R_off_tete(2:te_lmax, 2:te_lmax)
    Real(8) :: R_off_tttt(2:tt_lmax, 2:tt_lmax)
    Real(8) :: off_log_curv(2:tt_lmax, 2:tt_lmax)
    Real(8) :: Epsilon(2:tt_lmax, 2:tt_lmax)

    Real(8) :: cltt_dat(2:tt_lmax)
    Real(8) :: clte_dat(2:te_lmax)
    Real(8) :: cltb_dat(2:te_lmax)
    Real(8) :: ntt(2:tt_lmax)
    Real(8) :: ntt_te(2:te_lmax)
    Real(8) :: nee_te(2:te_lmax)
    Real(8) :: fskytt(2:tt_lmax)
    Real(8) :: fskyte(2:te_lmax)
    Real(8) :: ntt_tb(2:te_lmax)
    Real(8) :: nbb_tb(2:te_lmax)
    Real(8) :: fskytb(2:te_lmax)

    Private

    Public :: wmap_likelihood_init
    Public :: wmap_likelihood_compute
    Public :: wmap_likelihood_dof

Contains

    ! ===========================================================================
    Subroutine wmap_likelihood_init
        ! ===========================================================================

        Use wmap_util

        Implicit None

        Character(Len=128) :: option = 'wmap9_newgibbs_kq85cinv_v3'
        Integer :: l, ll
        Integer :: il, ill, i, j
        Real(8) :: dummy
        !Character(Len=240) :: ttfilename, tefilename, tbfilename, ttofffilename, teofffilename
        Logical :: good
        Integer :: lun
        Real :: rtmp, rtmp2

#ifdef TIMING
        Call wmap_timing_start('wmap_likelihood_init')
#endif

        Print *, 'Initializing WMAP likelihood, version ' // wmap_likelihood_version
        !Print *, 'Initializing WMAP likelihood, version ' // Trim(option)

        Call wmap_set_options(option)

        !-----------------------------------------------
        ! initialise beam uncertainties
        !-----------------------------------------------

        If (use_TT_beam_ptsrc) Call init_tt_beam_and_ptsrc_chisq(2, tt_lmax)

        !-----------------------------------------------
        ! initialise low l codes
        !-----------------------------------------------
        If (use_lowl_pol) Then
            !write(*,*) use_lowl_pol, 'using low ell'
#ifdef USE_LOWELL_TBEB
            Call tetbeebbeb_lowl_like_setup
#else
            Call teeebb_lowl_like_setup
#endif
        End If

        If (use_lowl_TT) Then
            If (use_gibbs) Then
                Call setup_for_tt_gibbs()
            Else
                Call setup_for_TT_exact(lowl_max)
            End If
        End If

        !-----------------------------------------------
        ! set file names
        !-----------------------------------------------

        !ttfilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_tt.p5v1.dat'
        !tefilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_te_recalibrated.p5v1.dat'
        !tbfilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_tb_recalibrated.p5v1.dat'
        !ttofffilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_tt_offdiag.p5v1.dat'
        !teofffilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_te_offdiag.p5v1.dat'

        !-----------------------------------------------
        ! get TT diag
        !-----------------------------------------------

        Inquire(File=ttfilename, Exist=good)
        If ( .Not. good) Then
            Write(*,*) 'cant find', Trim(ttfilename)
            Stop
        End If
        Call get_free_lun(lun)
        Open(Unit=lun, File=ttfilename, Form='formatted', Status='unknown', Action='read')

        Do l = 2, tt_lmax
            Read(lun,*) dummy, cltt_dat(l), ntt(l), fskytt(l)
        End Do
        Close(lun)

        !-----------------------------------------------
        ! get TE diag
        !-----------------------------------------------

        Inquire(File=tefilename, Exist=good)
        If ( .Not. good) Then
            Write(*,*) 'cant find', Trim(tefilename)
            Stop
        End If

        Call get_free_lun(lun)
        Open(Unit=lun, File=tefilename, Form='formatted', Status='unknown', Action='read')

        Do l = 2, te_lmax
            Read(lun,*) ll, clte_dat(l), dummy, ntt_te(l), nee_te(l), fskyte(l)
        End Do
        Close(lun)

        !-----------------------------------------------
        ! get TB diag
        !-----------------------------------------------

#ifdef USE_HIGHELL_TB
        Inquire(File=tbfilename, Exist=good)
        If ( .Not. good) Then
            Write(*,*) 'cant find', Trim(tbfilename)
            Stop
        End If

        Call get_free_lun(lun)
        Open(Unit=lun, File=tbfilename, Form='formatted', Status='unknown', Action='read')

        Do l = 2, te_lmax
            Read(lun,*) ll, cltb_dat(l), dummy, ntt_tb(l), nbb_tb(l), fskytb(l)
        End Do
        Close(lun)
#endif

        !-----------------------------------------------
        ! get TT off diag
        !-----------------------------------------------

        R_off_tttt = 0.

        Inquire(File=ttofffilename, Exist=good)

        If ( .Not. good) Then
            Write(*,*) 'cant find', Trim(ttofffilename)
            Stop
        End If

        Call get_free_lun(lun)
        Open(Unit=lun, File=ttofffilename, Form='formatted', Status='unknown', Action='read')
        ! --- left as is --- could be written better if the files are changed
        Do i = 2, tt_lmax
            Do j = i + 1, tt_lmax

                Read(lun,*) l, ll, Epsilon(i, j), R_off_tttt(i, j)
                R_off_tttt(j, i) = R_off_tttt(i, j)
                Epsilon(j, i) = Epsilon(i, j)

                If (i .Ne. l .Or. j .Ne. ll) Then
                    Write(*,*) "tt off file misread", i, j, l, ll
                    Stop
                End If

            End Do
        End Do
        ! ---
        Close(lun)

        !-----------------------------------------------
        ! get TE off diag
        !----------------------------------------------

        Inquire(File=teofffilename, Exist=good)

        If ( .Not. good) Then
            Write(*,*) 'cant find', Trim(teofffilename)
            Stop
        End If

        Call get_free_lun(lun)
        Open(Unit=lun, File=teofffilename, Form='formatted', Status='unknown', Action='read')

        ! --- left as is --- could be written better if the files are changed
        Do i = 2, te_lmax - 1
            Do j = i + 1, te_lmax

                Read(lun,*) l, ll, dummy

                If (l .Le. te_lmax .And. ll .Le. te_lmax) Then
                    R_off_tete(i, j) = dummy
                    R_off_tete(j, i) = R_off_tete(i, j)
                End If

                If (l .Ne. i .Or. ll .Ne. j) Then
                    Write(*,*) "TE off diag misread i,j,l,ll", i, j, l, ll
                    Stop
                End If

            End Do
        End Do
        Close(lun)

        initialise_pass2 = .False.

#ifdef TIMING
        Call wmap_timing_end()
#endif

    End Subroutine wmap_likelihood_init

    Subroutine wmap_likelihood_dof(tt_npix, teeebb_npix)

        Integer, Intent(Out) :: tt_npix, teeebb_npix

        tt_npix = tt_pixlike_dof()
#ifdef USE_LOWELL_TBEB
        teeebb_npix = tetbeebbeb_pixlike_dof()
#else
        teeebb_npix = teeebb_pixlike_dof()
#endif

    End Subroutine wmap_likelihood_dof

    ! ===========================================================================
#ifdef USE_LOWELL_TBEB
    Subroutine wmap_likelihood_compute(cltt, clte, cltb, clee, cleb, clbb, like)
#elif USE_HIGHELL_TB
    Subroutine wmap_likelihood_compute(cltt, clte, cltb, clee, cleb, clbb, like)
#else
    Subroutine wmap_likelihood_compute(cltt, clte, clee, clbb, like)
#endif
        ! ===========================================================================

        Use wmap_util

        Implicit None
        Real(8), Intent(In) :: cltt(2:*), clte(2:*), clee(2:*), clbb(2:*)
#ifdef USE_LOWELL_TBEB
        Real(8), Intent(In) :: cltb(2:*), cleb(2:*)
        Real(8) :: ee2, bb2, eb2
#elif USE_HIGHELL_TB
        Real(8), Intent(In) :: cltb(2:*), cleb(2:*)
        Real(8) :: ee2, bb2, eb2
#endif
        Real(8), Intent(Out) :: like(num_WMAP)
        Integer :: il, ill
        Real(8) :: dlnlike_tot, dlnlike, dlnlike_beam, ln_det_TETE, ln_det_TBTB
        !REAL(8) :: fisher(2:tt_lmax, 2:tt_lmax)
        Real(8), Allocatable :: fisher(:, :)
        Real(8) :: tttt(2:tt_lmax), tete(2:te_lmax), ttte(2:te_lmax), tbtb(2:te_lmax)
        Real(8) :: z(2:tt_lmax), zbar(2:tt_lmax)
        Real(8) :: cltt_temp(2:tt_lmax)
        Real, Allocatable, Dimension(:) :: lowl_cl
        Integer :: tt_hi_l_start, te_hi_l_start
        Real(8) :: correlation_coefficient_cl, tol = 1d-10

#ifdef TIMING
        Call wmap_timing_start('wmap_likelihood_compute')
#endif

        Call wmap_likelihood_error_init

        If (initialise_pass2) Then
            !Print *, 'Likelihood code needs to be initialized with an option string'
            !Stop
            Call wmap_likelihood_init
        End If

        like = 0.d0

        ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ! allocate memory

        Allocate(fisher(2:tt_lmax, 2:tt_lmax))

        !--------------------------------------
        ! Are cltt, clte, and clee consistent?
        !--------------------------------------
        Do il = 2, ttmax
            If (Abs(clte(il)) > 0d0) Then
                correlation_coefficient_cl = Abs(clte(il)) / Sqrt(cltt(il)*clee(il))
                If (correlation_coefficient_cl-1d0 > tol) Then
                    Call wmap_likelihood_error('unphysical input: TE/sqrt(TT*EE) > 1 at l=', il)
                    Return
                End If
            End If
        End Do

#ifdef USE_HIGHELL_TB
        Do il = 2, ttmax
            If (Abs(cltb(il)) > 0d0) Then
                correlation_coefficient_cl = Abs(cltb(il)) / Sqrt(cltt(il)*clbb(il))
                If (correlation_coefficient_cl-1d0 > tol) Then
                    Call wmap_likelihood_error('unphysical input: TB/sqrt(TT*BB) > 1 at l=', il)
                    Return
                End If
            End If
        End Do
#endif

#ifdef USE_LOWELL_TBEB
        Do il = 2, ttmax
            If (Abs(cltb(il)) > 0d0) Then
                correlation_coefficient_cl = Abs(cltb(il)) / Sqrt(cltt(il)*clbb(il))
                If (correlation_coefficient_cl-1d0 > tol) Then
                    Call wmap_likelihood_error('unphysical input: TB/sqrt(TT*BB) > 1 at l=', il)
                    Return
                End If
            End If

            If (Abs(cleb(il)) > 0d0) Then
                correlation_coefficient_cl = Abs(cleb(il)) / Sqrt(clee(il)*clbb(il))
                If (correlation_coefficient_cl-1d0 > tol) Then
                    Call wmap_likelihood_error('unphysical input: EB/sqrt(EE*BB) > 1 at l=', il)
                    Print *, correlation_coefficient_cl
                    Print *, 'unphysical input: EB/sqrt(EE*BB) > 1 at l=', il
                    Return
                End If
            End If

            If (Abs(cleb(il)) > 0d0) Then
                ee2 = clee(il) - clte(il) ** 2 / cltt(il)
                bb2 = clbb(il) - cltb(il) ** 2 / cltt(il)
                eb2 = cleb(il) - clte(il) * cltb(il) / cltt(il)
                If (Abs(eb2)/Sqrt(ee2*bb2)-1d0 > tol) Then
                    Call wmap_likelihood_error("unphysical input: EB'/sqrt(EE'*BB') > 1 at l=", il)
                    Return
                End If
            End If
        End Do
#endif

        cltt_temp(2:ttmax) = cltt(2:ttmax)

        !---------------------------------------------------------------------------
        ! low l TT likelihood
        !---------------------------------------------------------------------------
        If (use_lowl_TT) Then
            If (use_gibbs) Then
                Call compute_tt_gibbslike(cltt_temp(2:lowl_max), like(ttlowllike))
                like(ttlowldet) = 0.0
            Else
                Allocate(lowl_cl(2:lowl_max))
                lowl_cl(2:lowl_max) = cltt_temp(2:lowl_max)
                Call compute_tt_pixlike(lowl_max, lowl_cl, like(ttlowllike), like(ttlowldet))
                Deallocate(lowl_cl)
            End If
            tt_hi_l_start = lowl_max + 1
        Else
            tt_hi_l_start = ttmin
        End If

        !---------------------------------------------------------------------------
        ! low l TE/EE/BB likelihood
        !---------------------------------------------------------------------------
        If (use_lowl_pol) Then
#ifdef USE_LOWELL_TBEB
            Call tetbeebbeb_lowl_likelihood(23, cltt_temp, clte, cltb, clee, clbb, cleb, like(lowllike), like(lowldet))
#else
            Call teeebb_lowl_likelihood(23, cltt_temp, clte, clee, clbb, like(lowllike), like(lowldet))
#endif
            te_hi_l_start = 24
        Else
            te_hi_l_start = temin
        End If

        !---------------------------------------------------------------------------
        ! TT and TE covariance terms
        !---------------------------------------------------------------------------
        tttt = 0.d0
        tete = 0.d0
        ttte = 0.d0

        If (use_TT .Or. use_TE) Then

            Do il = ttmin, ttmax
                tttt(il) = 2. * (cltt_temp(il)+ntt(il)) ** 2 / ((2.0d0*Dble(il)+1.0d0)*fskytt(il)**2.)
            End Do

            Do il = temin, temax
                tete(il) = ((cltt_temp(il)+ntt_te(il))*(clee(il)+nee_te(il))+clte(il)**2) / &
                     ((2.0d0*Dble(il)+1.0d0)*fskyte(il)**2.)
                ttte(il) = 2 * ((cltt_temp(il)+ntt(il))*clte(il)) / &
                    ((2.0d0*Dble(il)+1.0d0)*fskyte(il)*fskytt(il))
            End Do

        End If

        !---------------------------------------------------------------------------
        ! TB covariance terms(diagonal only)
        !---------------------------------------------------------------------------
        tbtb = 0.d0

#ifdef USE_HIGHELL_TB
        Do il = temin, temax
            tbtb(il) = ((cltt_temp(il)+ntt_tb(il))*(clbb(il)+nbb_tb(il))+cltb(il)**2) / &
                ((2.0d0*Dble(il)+1.0d0)*fskytb(il)**2.)
        End Do
#endif

        !---------------------------------------------------------------------------
        !TTTT MASTER likelihood
        !---------------------------------------------------------------------------
        If (use_TT) Then

            fisher = 0.d0
            off_log_curv = 0.d0

            Do il = ttmin, ttmax
                z(il) = dlog(cltt_dat(il)+ntt(il))
                zbar(il) = dlog(cltt_temp(il)+ntt(il))
            End Do
            ! --- original
            !     do il=ttmin,ttmax
            !        do ill=il,ttmax
            ! --- RF
            Do ill = ttmin, ttmax
                Do il = ttmin, ill
                    ! ---
                    If (il .Eq. ill) Then
                        If (il .Le. te_lmax) Then
                            fisher(il, ill) = tete(il) / (tttt(il)*tete(il)-ttte(il)**2)
                        Else
                            fisher(il, ill) = 1.d0 / tttt(il)
                        End If
                    Else
                        fisher(il, ill) = - R_off_tttt(il, ill) / Sqrt(tttt(il)*tttt(ill)) &
                            + Epsilon(il, ill) / (tttt(il)*tttt(ill))
                    End If
                    off_log_curv(il, ill) = (cltt_temp(il)+ntt(il)) * fisher(il, ill) * &
                        & (cltt_temp(ill)+ntt(ill))
                End Do
            End Do

            dlnlike_tot = 0.d0

            ! --- original
            !     do il=ttmin,ttmax
            !        do ill=il,ttmax
            ! --- RF
            Do ill = ttmin, ttmax
                Do il = ttmin, ill
                    ! ---
                    dlnlike = 2.d0 / 3.d0 * (z(il)-zbar(il)) * off_log_curv(il, ill) * (z(ill)-zbar(ill)) + &
                        1.d0 / 3.d0 * (cltt_temp(il)-cltt_dat(il)) * &
                        fisher(il, ill) * (cltt_temp(ill)-cltt_dat(ill))

                    If (il .Ge. tt_hi_l_start .Or. ill .Ge. tt_hi_l_start) Then
                        If (il .Eq. ill) Then
                            dlnlike_tot = dlnlike_tot + dlnlike
                        Else
                            dlnlike_tot = dlnlike_tot + dlnlike * 2
                        End If
                    End If

                End Do
            End Do

            like(ttlike) = dlnlike_tot / 2.d0

        End If
        !---------------------------------------------------------------------------
        !TTTT Beam and point source correction
        !---------------------------------------------------------------------------
        If (use_TT .And. use_TT_beam_ptsrc) Then
            dlnlike_beam = compute_tt_beam_and_ptsrc_chisq(ttmin, ttmax, &
                cltt_temp, cltt_dat, ntt, fisher, z, zbar)
            If (Abs(dlnlike_beam) >= dlnlike_tot/4d0) Then
                Call wmap_likelihood_warning('beam correction invalid', 0)
                dlnlike_beam = 0d0
            End If
            like(beamlike) = dlnlike_beam / 2.d0
        End If

        !---------------------------------------------------------------------------
        !TETE MASTER likelihood
        !---------------------------------------------------------------------------
        If (use_TE) Then
            ln_det_TETE = 0.0d0
            dlnlike_tot = 0.d0
            fisher = 0.d0
            ! --- original
            !     do il=temin,temax
            !        do ill=il,temax
            ! --- RF
            Do ill = temin, temax
                Do il = temin, ill
                    ! ---
                    If (il .Eq. ill) Then
                        If (il .Le. te_lmax) Then
                            fisher(il, ill) = tttt(il) / (tttt(il)*tete(il)-ttte(il)**2)
                        Else
                            fisher(il, ill) = 1.d0 / tete(il)
                        End If
                    Else
                        fisher(il, ill) = - R_off_tete(il, ill) / Sqrt(tete(il)*tete(ill))
                    End If
                End Do
            End Do
            ! --- original
            !     do il=temin,temax
            !        do ill=il,temax
            ! --- RF
            Do ill = temin, temax
                Do il = temin, ill
                    ! ---
                    dlnlike = (clte(il)-clte_dat(il)) * fisher(il, ill) * (clte(ill)-clte_dat(ill))

                    If (il .Ge. te_hi_l_start .Or. ill .Ge. te_hi_l_start) Then
                        If (il .Eq. ill) Then
                            dlnlike_tot = dlnlike_tot + dlnlike
                            ln_det_TETE = ln_det_TETE - Log(fisher(il, ill))
                        Else
                            dlnlike_tot = dlnlike_tot + dlnlike * 2.
                        End If
                    End If

                End Do
            End Do
            like(telike) = dlnlike_tot / 2.d0

            like(tedet) = (ln_det_TETE-te_lndet_offset) / 2d0
        End If

        !---------------------------------------------------------------------------
        !TBTB MASTER likelihood
        !---------------------------------------------------------------------------
#ifdef USE_HIGHELL_TB
        ln_det_TBTB = 0.0d0
        dlnlike_tot = 0.d0
        fisher = 0.d0

        Do il = temin, temax
            fisher(il, il) = 1.d0 / tbtb(il)
        End Do

        Do il = temin, temax

            dlnlike = (cltb(il)-cltb_dat(il)) ** 2. * fisher(il, il)

            If (il .Ge. te_hi_l_start) Then
                dlnlike_tot = dlnlike_tot + dlnlike
                ln_det_TBTB = ln_det_TBTB - Log(fisher(il, il))
            End If

        End Do
        like(tblike) = dlnlike_tot / 2.d0

        like(tbdet) = (ln_det_TBTB-tb_lndet_offset) / 2d0
        Print *, ln_det_TBTB
#endif

10              Continue

        ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ! deallocate memory

        Deallocate(fisher)

#ifdef TIMING
        Call wmap_timing_end()
#endif

    End Subroutine wmap_likelihood_compute

End Module

