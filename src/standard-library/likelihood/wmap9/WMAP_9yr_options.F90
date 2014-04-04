! ===========================================================================
Module WMAP_OPTIONS
    
    ! This module contains the options in the likelihood code
    !
    ! ===========================================================================

    !---------------------------------------------------
    ! location of input data
    ! ---------------------------------------------------

#ifdef WMAP9_DATA_DIR   
  character(len=1024) :: WMAP_data_dir =&
  WMAP9_DATA_DIR
#else
  character(len=1024) :: WMAP_data_dir = ''
#endif

    ! For use in WMAP_9yr_likelihood.F90
    Character(Len=240) :: ttfilename, tefilename, tbfilename, ttofffilename, teofffilename

    ! For use in WMAP_9yr_tt_beam_ptsrc_chisq.f90
    Character(Len=128) :: ifn_ptsrc_mode, ifn_beam_modes, ifn_fiducial_cltt

    ! For use in WMAP_9yr_teeebb_pixlike.F90
    Character(Len=256) :: teeebb_filename(0:9), eebbdir, teeebb_maskfile

    !---------------------------------------------------
    ! likelihood terms from WMAP
    !---------------------------------------------------
#ifdef USE_HIGHELL_TB
    Integer, Parameter :: num_WMAP = 10 ! number of individual chi2 terms in likelihood
#else
    Integer, Parameter :: num_WMAP = 8 ! number of individual chi2 terms in likelihood
#endif

    Integer, Parameter :: ttlike = 1 ! master tttt chisq flag
    Integer, Parameter :: ttlowllike = 2 ! low tttt chisq flag
    Integer, Parameter :: ttlowldet = 3 ! low tttt determinant flag
    Integer, Parameter :: beamlike = 4 ! beam/pt source correction to tttt chisq flag
    Integer, Parameter :: telike = 5 ! master tete chisq flag
    Integer, Parameter :: tedet = 6 ! master tete determinant flag
    Integer, Parameter :: lowllike = 7 ! TE/EE/BB lowl chisq flag
    Integer, Parameter :: lowldet = 8 ! TE/EE/BB lowl determinant flag
    Integer, Parameter :: tblike = 9 ! master tbtb chisq flag
    Integer, Parameter :: tbdet = 10 ! master tbtb determinant flag

    !---------------------------------------------------
    ! l range to be used in the likelihood code
    ! change these to consider a more limited l range in TTTT and TETE
    !---------------------------------------------------
    Integer :: ttmax = 1200 ! must be l.le.1200
    Integer :: ttmin = 2 ! must be l.ge.2
    Integer :: temax = 800 ! must be l.le.800
    Integer :: temin = 2 ! must be l.ge.2

    !---------------------------------------------------
    ! various likelihood options
    ! change these to include/ exclude various likelihood aspects
    !---------------------------------------------------
    Logical :: use_lowl_TT = .True. ! include TT pixel likelihood, for l<=lowl_max
    Logical :: use_lowl_pol = .True. ! include TE,EE,BB pixel likelihood for l<24
    Logical :: use_TT = .True. ! include MASTER TT in likelihood
    Logical :: use_TT_beam_ptsrc = .True. ! include beam/ptsrc errors
    Logical :: use_TE = .True. ! include MASTER TE in likelihood

    !---------------------------------------------------
    ! *** AN IMPORTANT CHANGE WITH REGARD TO THE TT LIKELIHOOD ***
    !---------------------------------------------------
    ! There are two options to choose from for evaluating the low-l temperature
    ! likelihood. Both options produce the same results.
    !
    ! (1) The direct evaluation of likelihood in pixel space using a resolution 4 temperature map.
    ! (2) The Gibbs sampling.
    !
    ! The option(2) is much faster to evaluate than the option(1).
    !
    ! To use(1), set "use_gibbs = .false." and "lowl_max = 30".
    ! To use(2), set "use_gibbs = .true." and "lowl_max = 32".
    !
    ! Note that the resolution 3 option for(1) has been disabled.
    ! 
    Logical :: use_gibbs = .True.

    !---------------------------------------------------
    ! (1) Pixel likelihood
    !---------------------------------------------------
    Integer :: lowl_tt_res = 4 ! TT map resolution
    Integer :: lowl_max = 32 ! use low l TT code 2<l<lowl_max

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
    ! for one parameter run(to use every third value from the first half
    ! (approximately) of the file), and
    ! gibbs_first_iteration = 25100
    ! gibbs_last_iteration = 50000
    ! gibbs_skip = 3
    ! for another parameter run, to use the second half of the file(every third value).
    !
    ! To get really fast(possibly inaccurate) likelihoods,
    ! set gibbs_skip to be ~ 0.01 * (gibbs_last_iteration - gibbs_first_iteration)
    !
    ! gibbs_first_iteration must be >= 1
    
    Character(Len=256) :: gibbs_sigma_filename = &
        'lowlT/gibbs/sigmaEllsHkeChu_test16_ilc_9yr_5deg_r5_2uK_corrected_kq85y9_June_r5_all.fits'
    Integer :: gibbs_first_iteration = 10
    Integer :: gibbs_last_iteration = 120000
    Integer :: gibbs_skip = 2

    ! The sum in the BR estimator goes up to this value:
    Integer :: gibbs_ell_max = 32

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
#ifdef FASTERTT
    Double Precision, Parameter :: tt_pixlike_lndet_offset(4:4) = (/ 5024.741512d0 /)
#else
    Double Precision, Parameter :: tt_pixlike_lndet_offset(4:4) = (/ - 29677.056620d0 /)
#endif
    Double Precision, Parameter :: teeebb_pixlike_lndet_offset = 16078.083180d0
    Double Precision, Parameter :: te_lndet_offset = 3584.277805d0
    Double Precision, Parameter :: tb_lndet_offset = 3598.152208d0

Contains

    Subroutine wmap_print_options()
        Print *, "-----------------------------------------------------"
        Print *, "WMAP_data_dir = ", Trim(WMAP_data_dir)
        Print *, ""
        Print *, "ttfilename =    ", Trim(ttfilename)
        Print *, "tefilename =    ", Trim(tefilename)
        Print *, "tbfilename =    ", Trim(tbfilename)
        Print *, "ttofffilename = ", Trim(ttofffilename)
        Print *, "teofffilename = ", Trim(teofffilename)
        Print *, ""
        Print *, "ttmax = ", ttmax
        Print *, "ttmin = ", ttmin
        Print *, "temax = ", temax
        Print *, "temin = ", temin
        Print *, ""
        Print *, "use_lowl_TT =       ", use_lowl_TT
        Print *, "use_lowl_pol =      ", use_lowl_pol
        Print *, "use_TT =            ", use_TT
        Print *, "use_TT_beam_ptsrc = ", use_TT_beam_ptsrc
        Print *, "use_TE =            ", use_TE
        Print *, ""
        Print *, "lowl_tt_res = ", lowl_tt_res
        Print *, "lowl_max =    ", lowl_max
        Print *, ""
        Print *, "tt_pixlike_lndet_offset =     ", tt_pixlike_lndet_offset
        Print *, "teeebb_pixlike_lndet_offset = ", teeebb_pixlike_lndet_offset
        Print *, "te_lndet_offset =             ", te_lndet_offset
#ifdef USE_HIGHELL_TB
        Print *, "tb_lndet_offset =             ", tb_lndet_offset
#endif
        Print *, ""
        Print *, "use_gibbs = ", use_gibbs
        Print *, "gibbs_sigma_filename = ", Trim(gibbs_sigma_filename)
        Print *, "gibbs_file = ", Trim(WMAP_data_dir) // Trim(gibbs_sigma_filename)
        Print *, "gibbs_first_iteration = ", gibbs_first_iteration
        Print *, "gibbs_last_iteration =  ", gibbs_last_iteration
        Print *, "gibbs_skip =            ", gibbs_skip
        Print *, "gibbs_ell_max =         ", gibbs_ell_max
        Print *, "-----------------------------------------------------"
    End Subroutine wmap_print_options


    Subroutine wmap_set_options(option)
        Character(Len=*), Intent(In) :: option

        Select Case(Trim(option))
        Case('wmap9_newgibbs_kq85cinv_v3')
            gibbs_sigma_filename = 'lowlT/gibbs/sigmaEllsHkeChu_test16_ilc_9yr_5deg_r5_2uK_corrected_kq85y9_June_r5_all.fits'
            gibbs_first_iteration = 10
            gibbs_last_iteration = 15000
            gibbs_skip = 2

            ttfilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_tt.p4v6.wmap9.kq85.cinv_v3.dat'
            tefilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_te.p5_final.dat'
            tbfilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_tb.p5_final.dat'
            ttofffilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_tt_offdiag.p4v4.wmap9.kq85.cinv_v3.dat'
            teofffilename = Trim(WMAP_data_dir) // 'highl/wmap_likelihood_inputs_te_offdiag.p5_final.dat'

            ifn_ptsrc_mode = "highl/wmap_likelihood_inputs_ptsrc.p5_final.dat"
            ifn_beam_modes = "highl/top_ten_modes.beam_covariance_VW_combined.dat"
            ifn_fiducial_cltt = "test_cls_v4.dat"

            eebbdir = Trim(WMAP_data_dir) // 'lowlP/std/'
            teeebb_filename(0) = Trim(eebbdir) // 'masked_ee_ninvplninv_qu_r3_corrected_9yr.KaQV.fits'
            teeebb_filename(1) = Trim(eebbdir) // 'masked_bb_ninvplninv_qu_r3_corrected_9yr.KaQV.fits'
            teeebb_filename(2) = Trim(eebbdir) // 'masked_ninv_qu_r3_corrected_9yr.KaQV.fits'
            teeebb_filename(3) = Trim(eebbdir) // 'wt_r3_9yr.KaQV.map_q'
            teeebb_filename(4) = Trim(eebbdir) // 'wt_r3_9yr.KaQV.map_u'
            teeebb_filename(6) = Trim(eebbdir) // 'masked_ninvy_e_qu_r3_corrected_9yr.KaQV.fits'

            teeebb_filename(5) = Trim(WMAP_data_dir) // 'lowlP/alm_tt_fs_r9_ilc_nopixwin_9yr.dat'
            teeebb_filename(9) = Trim(WMAP_data_dir) // 'healpix_data/pixel_window_n0008.txt'
            teeebb_maskfile = Trim(WMAP_data_dir) // 'lowlP/mask_r3_p06_jarosik.fits'

        Case Default
            Print *, 'Unable to interpret option:>' // Trim(option) // '<'
            Stop
        End Select
        !Print *, 'Interpreted option:>' // Trim(option) // '<'
        !Print *, 'Full option:>' // option // '<'

    End Subroutine wmap_set_options

    Subroutine wmap_init_options()
      INTEGER :: length, status
      length = 0
      status = 0

      if (len(trim(WMAP_data_dir)) .ne. 0) then
            write(*,*) "Using precompiled WMAP_DATA_DIR: " // trim(wmap_data_dir)
        return
      endif
      CALL get_environment_variable("WMAP9_DATA_DIR", WMAP_data_dir, length, status)
      IF (status .EQ. 1) then
        if (trim(wmap_data_dir)=="") then
            STOP "Please define (using export for bash or setenv for [t]csh) the environment variable WMAP9_DATA_DIR pointing to the WMAP 9 year data directory"
        else
            write(*,*) "Using predefined WMAP_DATA_DIR: " // trim(wmap_data_dir)
        endif
      ENDIF 
      WMAP_data_dir = trim(WMAP_data_dir) // '/'
    End Subroutine wmap_init_options

End Module WMAP_OPTIONS
