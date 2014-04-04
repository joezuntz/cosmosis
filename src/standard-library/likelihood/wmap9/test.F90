!===========================================================
Program test_likelihood

    ! RB December '05
    !===========================================================

    Use wmap_likelihood_9yr
    Use wmap_options
    Use wmap_util

    Implicit None

    Real(8), Dimension(:), Allocatable :: cl_tt, cl_te, cl_ee, cl_bb, cl_pp
#ifdef USE_LOWELL_TBEB
    Real(8), Dimension(:), Allocatable :: cl_tb, cl_eb
#endif
    Character(Len=128) :: filename
    Real(8) :: like(num_WMAP), like_tot, expected_like_tot
    Integer :: lun, l, i, olun
    Integer :: tt_npix, teeebb_npix

    !---------------------------------------------------

    Print *, ""
    Print *, "WMAP 9-year likelihood test program"
    Print *, "==================================="
    Print *, ""
    Print *, "NOTE: This code uses a different CMB spectrum than previous versions."
    Print *, "      The new spectrum(data/test_cls_v5.dat) is a better fit to the data"
    Print *, "      than the old one(data/test_cls_v4.dat)."
    Print *, ""
    Print *, "      As before, a constant offset is now being subtracted from ln(L)."
    Print *, "      The value of the offset is the sum of the determinant"
    Print *, "      contributions to ln(L) computed for the CMB spectrum in"
    Print *, "      data/test_cls_v5.dat, ln_det_C_f:"
    Print *, ""
    Print *, "        -2ln(L) = chi^2 + ln_det_C - ln_det_C_f"
    Print *, ""

    !---------------------------------------------------

    Allocate(cl_tt(ttmin:ttmax))
    Allocate(cl_te(ttmin:ttmax))
    Allocate(cl_ee(ttmin:ttmax))
    Allocate(cl_bb(ttmin:ttmax))
    Allocate(cl_pp(ttmin:ttmax))

    cl_bb = 0d0

#ifdef USE_LOWELL_TBEB
    Allocate(cl_tb(ttmin:ttmax))
    Allocate(cl_eb(ttmin:ttmax))
    cl_tb(:) = 0d0
    cl_eb(:) = 0d0
#endif

    !---------------------------------------------------
    ! read in Cls
    !---------------------------------------------------
    filename = Trim(WMAP_data_dir) // 'test_cls_v5.dat'

    Write(*,*) "Reading in Cls from: ", Trim(filename)
    Call get_free_lun(lun)
    Open(Unit=lun, File=filename, Action='read', Status='old')

    Do l = 2, ttmax
        Read(lun,*) i, cl_tt(l), cl_ee(l), cl_bb(l), cl_te(l), cl_pp(l)
    End Do

    Close(lun)

    !do l = 2,9
    !	cl_te(l) = 0d0
    !	cl_ee(l) = 0d0
    !end do

    !---------------------------------------------------
    ! put in likelihood options
    ! see PASS2_options module for the options below
    !---------------------------------------------------

    use_TT = .True.
    use_TE = .True.
    use_lowl_TT = .True.
    use_lowl_pol = .True.

    !---------------------------------------------------
    ! get likelihoods
    !---------------------------------------------------
    like = 0.d0
    Call wmap_likelihood_init
    Call wmap_likelihood_dof(tt_npix, teeebb_npix)
#ifdef USE_LOWELL_TBEB
    Call wmap_likelihood_compute(cl_tt, cl_te, cl_tb, cl_ee, cl_eb, cl_bb, like)
#else
    Call wmap_likelihood_compute(cl_tt, cl_te, cl_ee, cl_bb, like)
#endif
    Call wmap_likelihood_error_report


    like_tot = Sum(like(1:num_WMAP))

    !---------------------------------------------------
    ! write outputs
    !---------------------------------------------------
    Print 1
    Print 2
    Print 1
    Print 3, 'MASTER TTTT             ', 2 * like(ttlike), ttmax - lowl_max
    Print 5, 'Beam/ptsrc TT correction', 2 * like(beamlike)
    If (use_gibbs) Then
        Print 5, 'low-l TTTT gibbs        ', 2 * like(ttlowllike)
    Else
        Print 4, 'low-l TTTT chi2         ', 2 * like(ttlowllike), tt_npix
        Print 5, 'low-l TTTT det          ', 2 * like(ttlowldet)
    End If
    Print 3, 'MASTER TETE chi2        ', 2 * like(telike), temax - 23
    Print 5, 'MASTER TETE det         ', 2 * like(tedet)
    Print 4, 'TT/TE/EE/BB low-l chi2  ', 2 * like(lowllike), teeebb_npix
    Print 5, 'TT/TE/EE/BB low-l det   ', 2 * like(lowldet)
    Print 1
    Print 5, 'TOTAL -2ln(L)           ', 2 * like_tot
    Print 1

    If (use_gibbs) Then
        expected_like_tot = 7557.9659905
    Else
#ifdef FASTERTT
        expected_like_tot = 8222.332496
#else
        expected_like_tot = 12349.777782d0
#endif
    End If

    Print '(A,F13.6)', "Expected -2ln(L)         = ", expected_like_tot
    Print '(A,F13.6)', "      Difference         = ", 2 * like_tot - expected_like_tot
    Print 1
    Print *, ""
    Print *, "Differences on the order of O(0.001) are normal between platforms."
    Print *, ""

    Stop
1   Format('------------------------------------------------------------------')
2   Format('Breakdown of -2ln(L)')
3   Format(A24, ' = ', F13.6, ' for ', I6, ' ls')
4   Format(A24, ' = ', F13.6, ' for ', I6, ' pixels')
5   Format(A24, ' = ', F13.6)
End Program test_likelihood
