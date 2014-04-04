!=========================================
Module wmap_tt_beam_ptsrc_chisq

    ! Module to calculated the correction to the TTTT chisq
    ! from beam and point source corrections
    !
    ! Written by Mike Nolta
    !=========================================
    Implicit None

    Public :: init_tt_beam_and_ptsrc_chisq
    Public :: quit_tt_beam_and_ptsrc_chisq
    Public :: compute_tt_beam_and_ptsrc_chisq

    Private

    Integer, Parameter :: n_beam_modes = 9
    Integer :: nmodes, ptsrc_mode_index
    !Character(Len=128) :: ifn_ptsrc_mode = "highl/clps.p5v1.dat",
    !Character(Len=128) :: ifn_beam_modes = "highl/top_ten_modes.beam_covariance_VW_combined_7yr.dat"
    !Character(Len=128) :: ifn_fiducial_cltt = "test_cls_v4.dat"
    Real(Kind=8), Parameter :: ptsrc_err = 0.1 !! 10%

    Integer :: lmin0, lmax0
    Real(Kind=8), Allocatable :: mode(:, :), F_mode(:, :), beam_mode(:, :)
    Real(Kind=8), Allocatable :: fiducial_cltt(:), ptsrc_mode(:)
    Real(Kind=8), Allocatable :: a(:), a2(:), b(:, :), c(:)
    Integer, Allocatable :: iwork(:)

    ! beam options See Appendix of Hinshaw, et.al. (2006) for a description.
    Logical :: beam_diagonal_sigma = .True.
    Logical :: beam_gaussian_likelihood = .True.
    Logical :: beam_fixed_fiducial_spectrum = .False.
    Logical :: beam_include_beam_modes = .True.
    Logical :: beam_include_ptsrc_mode = .True.

Contains

    Subroutine init_tt_beam_and_ptsrc_chisq(lmin, lmax)

        Use wmap_util
        Use wmap_options, Only: WMAP_data_dir, ifn_ptsrc_mode, ifn_beam_modes, ifn_fiducial_cltt
        Implicit None

        Integer, Intent(In) :: lmin, lmax

        Integer :: lun, i, l, stat
        Real(Kind=8) :: x
        Character(Len=256) :: ifn

        lmin0 = lmin
        lmax0 = lmax

        nmodes = 0
        If (beam_include_beam_modes) Then
            nmodes = nmodes + n_beam_modes
        End If
        If (beam_include_ptsrc_mode) Then
            nmodes = nmodes + 1
            ptsrc_mode_index = nmodes
        End If

        If (nmodes == 0) Then
            Return
        End If

        Allocate(a(nmodes))
        Allocate(b(nmodes, nmodes))
        Allocate(c(nmodes))
        Allocate(a2(nmodes))
        Allocate(iwork(nmodes))
        Allocate(fiducial_cltt(lmin:lmax))
        Allocate(ptsrc_mode(lmin:lmax))
        Allocate(beam_mode(lmin:lmax, n_beam_modes))
        Allocate(mode(lmin:lmax, nmodes))
        Allocate(F_mode(lmin:lmax, nmodes))

        If (beam_include_beam_modes) Then
            ifn = Trim(WMAP_data_dir) // Trim(ifn_beam_modes)
            Call get_free_lun(lun)
            Open(lun, File=ifn, Action='read', Status='old', Form='formatted')
            Do
                Read(lun,*, IoStat=stat) i, l, x
                If (stat /= 0) Exit
                If (i <= n_beam_modes) Then
                    beam_mode(l, i) = x
                End If
            End Do
            Close(lun)
        End If

        If (beam_fixed_fiducial_spectrum) Then
            ifn = Trim(WMAP_data_dir) // Trim(ifn_fiducial_cltt)
            Call get_free_lun(lun)
            Open(lun, File=ifn, Action='read', Status='old', Form='formatted')
            Do
                Read(lun,*, IoStat=stat) l, x
                If (stat /= 0) Exit
                If (l >= lmin .And. l <= lmax) Then
                    fiducial_cltt(l) = x
                End If
            End Do
            Close(lun)
        End If

        If (beam_include_ptsrc_mode) Then
            ifn = Trim(WMAP_data_dir) // Trim(ifn_ptsrc_mode)
            Call get_free_lun(lun)
            Open(lun, File=ifn, Action='read', Status='old', Form='formatted')
            Do
                Read(lun,*, IoStat=stat) l, x
                If (stat /= 0) Exit
                If (l >= lmin .And. l <= lmax) Then
                    ptsrc_mode(l) = x
                End If
            End Do
            Close(lun)
        End If

    End Subroutine init_tt_beam_and_ptsrc_chisq

    Subroutine quit_tt_beam_and_ptsrc_chisq()

        Implicit None

        If (nmodes > 0) Then
            Deallocate(beam_mode, mode, F_mode)
            Deallocate(ptsrc_mode, fiducial_cltt)
            Deallocate(a, b, c, a2, iwork)
        End If

    End Subroutine quit_tt_beam_and_ptsrc_chisq

    Function compute_tt_beam_and_ptsrc_chisq(lmin, lmax, cltt, cltt_dat, neff, fisher, z, zbar)

        Use wmap_util
        Implicit None

        Integer, Intent(In) :: lmin, lmax
        Real(Kind=8), Dimension(lmin0:lmax0), Intent(In) :: cltt, cltt_dat, neff, z, zbar
        Real(Kind=8), Dimension(lmin0:lmax0, lmin0:lmax0), Intent(In) :: fisher
        Real(Kind=8) :: compute_tt_beam_and_ptsrc_chisq

        Real(Kind=8) :: dgauss, dlnnorm, dlndet
        Integer :: i, j, l, l1, l2, stat

        If (nmodes == 0) Then
            compute_tt_beam_and_ptsrc_chisq = 0d0
            Return
        End If

        mode = 0d0

	!! beam modes
        If (beam_include_beam_modes) Then
            If (beam_fixed_fiducial_spectrum) Then
                Do i = 1, n_beam_modes
                    Do l = lmin, lmax
                        mode(l, i) = beam_mode(l, i) * fiducial_cltt(l)
                    End Do
                End Do
            Else
                Do i = 1, n_beam_modes
                    Do l = lmin, lmax
                        mode(l, i) = beam_mode(l, i) * cltt(l)
                    End Do
                End Do
            End If
        End If

	!! ptsrc mode
        If (beam_include_ptsrc_mode) Then
            !print *, 'including beam mode', ptsrc_mode_index
            mode(lmin:lmax, ptsrc_mode_index) = ptsrc_err * ptsrc_mode(lmin:lmax)
            !print *, 'ptsrc_mode(l=1000) = ', mode(1000,ptsrc_mode_index)
        End If

        F_mode = 0d0
        If (beam_diagonal_sigma) Then
            Do i = 1, nmodes
                Do l = lmin, lmax
                    F_mode(l, i) = fisher(l, l) * mode(l, i)
                End Do
            End Do
        Else
            Do i = 1, nmodes
                Do l1 = lmin, lmax
                    Do l2 = lmin, lmax
                        !do l2 = l1-50,l1+50
                        If (l2 < lmin .Or. l2 > lmax) Cycle
                        If (l2 < l1) Then
                            F_mode(l1, i) = F_mode(l1, i) + fisher(l2, l1) * mode(l2, i)
                        Else
                            F_mode(l1, i) = F_mode(l1, i) + fisher(l1, l2) * mode(l2, i)
                        End If
                    End Do
                End Do
            End Do
        End If

        a = 0d0
        b = 0d0
        Do i = 1, nmodes
            Do l = lmin, lmax
                a(i) = a(i) + (cltt_dat(l)-cltt(l)) * F_mode(l, i)
            End Do
            b(i, i) = b(i, i) + 1d0
            Do j = i, nmodes
                Do l = lmin, lmax
                    b(i, j) = b(i, j) + mode(l, i) * F_mode(l, j)
                End Do
                If (i /= j) b(j, i) = b(i, j)
            End Do
        End Do

        !	print *, 'nmodes = ', nmodes
        !	do i = 1,nmodes
        !		print '("a(",I2,") = ",E)', i, a(i)
        !	end do

        Call dpotrf('L', nmodes, b, nmodes, stat)
        If (stat /= 0) Then
            Call wmap_likelihood_error('beam/ptsrc: bad dpotrf', stat)
            compute_tt_beam_and_ptsrc_chisq = 0d0
            !		print *, 'bad dpotrf'
            Return
        End If

        c(:) = a(:)
        Call dpotrs('L', nmodes, 1, b, nmodes, c, nmodes, stat)
        If (stat /= 0) Then
            Call wmap_likelihood_error('beam/ptsrc: bad dpotrs', stat)
            compute_tt_beam_and_ptsrc_chisq = 0d0
            !         print *, 'bad dpotrs'
            Return
        End If
        dgauss = 0d0
        Do i = 1, nmodes
            dgauss = dgauss + a(i) * c(i)
        End Do

        If (beam_gaussian_likelihood) Then
            dlndet = 1d0
            Do i = 1, nmodes
                dlndet = dlndet * b(i, i) ** 2
            End Do
            dlndet = dlog(dlndet)

            !print *, 'beam chisq, lndet = ', dgauss, dlndet
            compute_tt_beam_and_ptsrc_chisq = - dgauss + dlndet
        Else
            a2 = 0d0
            Do i = 1, nmodes
                Do l = lmin, lmax
                    a2(i) = a2(i) + (z(l)-zbar(l)) * (cltt(l)+neff(l)) * F_mode(l, i)
                End Do
            End Do
            c(:) = a2(:)
            Call dpotrs('L', nmodes, 1, b, nmodes, c, nmodes, stat)
            If (stat /= 0) Then
                Call wmap_likelihood_error('beam/ptsrc: bad dpotrs', stat)
                compute_tt_beam_and_ptsrc_chisq = 0d0
                !			print *, 'bad dpotrs'
                Return
            End If
            dlnnorm = 0d0
            Do i = 1, nmodes
                dlnnorm = dlnnorm + a2(i) * c(i)
            End Do

            !print *, 'beam chisq, lndet = ', dgauss, dlnnorm, -(dgauss+2d0*dlnnorm)/3d0
            compute_tt_beam_and_ptsrc_chisq = - (dgauss+2d0*dlnnorm) / 3d0
        End If

    End Function compute_tt_beam_and_ptsrc_chisq

End Module wmap_tt_beam_ptsrc_chisq

