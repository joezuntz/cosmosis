! ===========================================================================
Module wmap_tlike
    !
    ! This code calculates the likelihood function for TT data at low l
    !
    !  1uK per pixel noise at nside=16 is added to regularize the matrix inversion.
    !
    !  It uses a smoothed ILC map as the temperature map and the Kp2 sky mask.
    !
    ! O. Dore
    ! D. Spergel, December 2005
    !
    ! << MODIFICATION HISTORY AFTER RELEASE on March 16, 2006 >>
    !
    ! E. Komatsu, June 20, 2006
    ! -- Use Nside=16.
    ! -- Use the pre-computed smoothed and degraded ILC and V-band maps.
    !   HEALPIX ROUTINES ARE NO LONGER NEEDED.
    ! -- Smoothing scale has been increased to 9.1831 degrees.
    !   I.e., the input map is band limited at l=40.
    ! -- White noise(1uK per pixel at nside=16) is added to the degraded maps.
    ! -- Fore_Marg is set to 2 explicitly.
    !
    ! E. Komatsu, March 5, 2009
    ! -- Changed the orders of do-loops for a better performance
    !    [thanks to Raphael Flauger]
    !
    ! E. Komatsu, December 20, 2009
    ! -- 7-yr version
    !===========================================================================

    Implicit None
    Private
    Public :: setup_for_tt_exact, compute_tt_pixlike, tt_pixlike_dof

    Integer :: ngood, nsmax, nzzz
    Real, Allocatable :: wl(:)
#ifdef FASTERTT
    Double Precision, Allocatable :: t_map(:)! ILC
#else
    Real, Allocatable :: t_map(:)! ILC
    Real, Allocatable :: f_map(:)! difference between ILC and V
#endif
    Double Precision, Allocatable, Dimension(:, :) :: C0, C ! C0 = monopole + dipole marginalized
    Double Precision, Allocatable, Dimension(:, :), Target :: zzz
    Double Precision, Allocatable :: vecs(:, :), yyy(:)
    Integer, Dimension(0:1023) :: pix2x = 0, pix2y = 0 ! for pix2vec
    Integer, Parameter :: ifore_marg = 2
    !
    ! ifore_marg = 0(no marginalization)
    ! ifore_marg = 1  (assume that the uncertainty is the difference between
    !			V and ILC)
    ! ifore_marg = 2  (no prior on amplitude of foreground unc.)
    !
Contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Subroutine setup_for_tt_exact(nlhigh, tt_ngood)
        !
        ! - READ IN DEGRADED Kp2 MASK(nside=16)
        !
        ! - ILC AND V-BAND MAPS SMOOTHED WITH A BEAM,
        !         exp(-0.5*l*(l+1)*theta_fwhm**2./(8.*alog(2.)))
        !   where
        !         theta_fwhm = 9.1285 degrees for ILC
        !         theta_fwhm = 9.1831 degrees for V-band
        !                      [ sqrt(9.1285^2+1^2)=9.1831 ]
        !
        ! - COMPUTES C0 for fiducial
        !
        Use wmap_util
        Use wmap_options, Only: WMAP_data_dir, lowl_tt_res

        Implicit None
        Include 'read_archive_map.fh'
        Include 'read_fits.fh'
        Integer, Intent(In) :: nlhigh ! vary cl's from 2 : nlhigh
        Integer, Intent(Out), Optional :: tt_ngood
        Integer :: ires != 4           ! resolution for low l TT code
        Integer :: npix, np, status, nlmax, m, lun
        Integer :: ip, jp, ipix, l, ll, jpix
        Character(Len=120) :: filename(0:6), ifn, ofn
        Logical :: exist
        Integer, Allocatable, Dimension(:) :: good
        Real, Allocatable, Dimension(:) :: dummy, ilc, vband, mask
        !    REAL, ALLOCATABLE, dimension(:) :: vband_clean
        Real, Allocatable, Dimension(:) :: cl_fiducial
        Real, Allocatable, Dimension(:, :, :) :: Plin
        Double Precision, Allocatable, Dimension(:) :: cl
        Real :: noise_rms, window_th, lc, theta_fwhm !=9.1831
        Integer :: iseed, k, stat
        Double Precision :: vec_ip(3), x, p(0:64), one_over_l
        Double Precision, Pointer, Dimension(:, :) :: Dptr2

#ifdef TIMING
        Call wmap_timing_start('setup_for_tt_exact')
#endif

        If (nlhigh /= 30) Then
            Print *, "error: set lowl_max=30 when use_gibbs=.false."
            Stop
        End If

#ifdef FASTERTT

        If (ifore_marg /= 2) Then
            Print *, '*** FASTERTT currently requires ifore_marg=2'
            Stop
        End If

        ngood = 957

        Allocate(C(0:ngood-1, 0:ngood-1))
        Allocate(t_map(0:ngood-1))

        ifn = Trim(WMAP_data_dir) // "/lowlT/faster/compressed_t_map_f2_9yr.txt"
        Call get_free_lun(lun)
        Open(lun, File=ifn, Status='old', Action='read')
        Do ip = 0, ngood - 1
            Read(lun,*) t_map(ip)
        End Do
        Close(lun)

        nzzz = (ngood*(ngood+1)) / 2
        Allocate(zzz(nzzz, nlhigh-1))
        Allocate(yyy(nzzz))

        !	do l = 2,nlhigh
        !
        !		write(ifn,'("/cita/d/raid-sievers2/sievers/WMAP3_cosmomc/faster_TT/wmap_compressed_fproj_",I0.2,".unf")') l
        !		open(lun,file=ifn,status='old',action='read',form='unformatted')
        !		read(lun) C
        !		close(lun)
        !
        !		k = 1
        !		do ip = 0,ngood-1
        !		do jp = ip,ngood-1
        !			zzz(k,l-1) = C(ip,jp)
        !			k = k + 1
        !		end do
        !		end do
        !	end do
        !	C = 0d0
        !
        !	ofn = "compressed_tt_f2.unf"
        !	open(lun,file=ofn,action="write",status="unknown",form="unformatted")
        !	write(lun) zzz
        !	close(lun)

        Dptr2 => zzz
        ifn = Trim(WMAP_data_dir) // "lowlT/faster/compressed_tt_matrices_f2_9yr.fits"
        Call Read_FITS_Double_2D(ifn, Dptr2, stat)
        If (stat .Ne. 0) Then
            Print *, 'Error ', stat, ' while reading ', Trim(ifn)
            Stop
        End If

#else

        ires = lowl_tt_res
	!filename(0)=trim(WMAP_data_dir)//'wmap_lcdm_pl_model_yr1_v1.dat'
        filename(0) = Trim(WMAP_data_dir) // 'test_cls_v5.dat'

        Select Case(lowl_tt_res)
            !	case(3)
            !		theta_fwhm=18.3
            !		filename(1)=trim(WMAP_data_dir)//'mask/mask_r3_kp2.fits'
            !		filename(2)=trim(WMAP_data_dir)//'maps/low_resolution_map_fs_r3_ilc_smooth_18.3deg.fits' ! ILC map
            !		filename(3)=trim(WMAP_data_dir)//'maps/low_resolution_map_fs_r3_vband_smooth_18.3deg.fits' ! raw V-band map for foreground marginalization
            !		filename(4)=trim(WMAP_data_dir)//'healpix_data/pixel_window_n0008.txt'
            !		!filename(5)=trim(WMAP_data_dir)//'pl_res3_sp.fits'
            !		!filename(6)=trim(WMAP_data_dir)//'maps/low_resolution_map_fs_r3_vband_clean_smooth_18.3deg.fits' ! template-cleaned V-band map -- not used
        Case(4)
            theta_fwhm = 9.1831

            filename(1) = Trim(WMAP_data_dir) // 'lowlT/pixlike/mask_r4_kq85.fits'
            filename(2) = Trim(WMAP_data_dir) // 'lowlT/pixlike/wmap_9yr_r4_ilc_smooth_9.1285deg.fits'
            filename(3) = Trim(WMAP_data_dir) // 'lowlT/pixlike/wmap_9yr_r4_vband_raw_smooth_9.1831deg.fits'
            filename(4) = Trim(WMAP_data_dir) // 'healpix_data/pixel_window_n0016.txt'
            !filename(5)=trim(WMAP_data_dir)//'' ! used only when ires=3
            !filename(6)=trim(WMAP_data_dir)//'maps/wmap_fs_r4_vband_clean_smooth_9.1831deg.fits'! smoothed cleaned V-band -- not used
        Case Default
            Call wmap_likelihood_error('bad lowl_tt_res', lowl_tt_res)
            Stop
        End Select

        lc = Sqrt(8.*alog(2.)) / theta_fwhm / 3.14159 * 180.

        !
        !       READ IN cl's for fiducial model(used for high l)
        !
        Allocate(cl_fiducial(2:512))
        Inquire(File=filename(0), Exist=Exist)
        If ( .Not. exist) Then
            Write(*,*) ' unable to open ' // Trim(filename(0))
            Stop
        End If
        Call get_free_lun(lun)
        Open(lun, File=filename(0), Status='old')
        Do l = 2, 512
            Read(lun,*) ll, cl_fiducial(l)
            cl_fiducial(ll) = cl_fiducial(ll) * 1.e-6 * 2. * 3.1415927 / (ll*(ll+1))
        End Do
        Close(lun)
        !
        ! READ IN LOW-RESOLUTION Kp2 MASK
        !
        nsmax = 2 ** ires
        nlmax = 4 * nsmax
        npix = 12 * nsmax * nsmax
        Allocate(dummy(0:npix-1), mask(0:npix-1))
        Call read_archive_map(filename(1), dummy, mask, np, status)
        If (status .Ne. 0 .Or. np .Ne. npix) Then
            Write(*,*) ' error reading mask'
            Stop
        End If
        ngood = Sum(mask)
        Allocate(good(0:ngood-1))
        ip = 0
        Do ipix = 0, 12 * nsmax * nsmax - 1
            If (mask(ipix) .Eq. 1) Then
                good(ip) = ipix
                ip = ip + 1
            End If
        End Do
        If (ip .Ne. ngood) Stop
        !
        !  READ IN LOW-RESOLUTION ILC AND V-BAND MAPS
        !  - smoothing scale is FWHM = 9.1 degrees
        !
        Allocate(ilc(0:npix-1), vband(0:npix-1))
        !    ALLOCATE(vband_clean(0:npix-1))
        Call read_archive_map(filename(2), ilc, dummy, np, status)
        If (status .Ne. 0 .Or. np .Ne. npix) Then
            Write(*,*) ' error reading ILC map'
            Stop
        End If
        Call read_archive_map(filename(3), vband, dummy, np, status)
        If (status .Ne. 0 .Or. np .Ne. npix) Then
            Write(*,*) ' error reading V-band map'
            Stop
        End If
        !    call read_archive_map(filename(6),vband_clean,dummy,np,status)
        !    if(status.ne.0.or.np.ne.npix) then
        !       write(*,*) ' error reading cleaned V-band map'
        !       stop
        !    endif
        Deallocate(dummy)

        !
        !  CALCULATE COMBINED BEAM
        !
        Allocate(wl(2:1024))
        Call get_free_lun(lun)
        Open(lun, File=filename(4), Status='old')
        Read(lun,*) window_th
        Read(lun,*) window_th
        Do l = 2, nlmax
            Read(lun,*) window_th
            wl(l) = window_th * Exp(-0.5*l*(l+1)/lc**2.)
        End Do
        Close(lun)
        Allocate(cl(2:nlmax))
        cl(2:nlmax) = cl_fiducial(2:nlmax) * wl(2:nlmax) ** 2.
        Deallocate(cl_fiducial)

        Allocate(C0(0:ngood-1, 0:ngood-1))
        Allocate(C(0:ngood-1, 0:ngood-1))
        Allocate(t_map(0:ngood-1))
        Allocate(f_map(0:ngood-1))
        Allocate(vecs(3, 0:ngood-1))

        C0 = 0d0

        Do ip = 0, ngood - 1
            t_map(ip) = ilc(good(ip))
!!$ t_map(ip) = vband_clean(good(ip)) !!$ don't use cleaned V. use ILC
            f_map(ip) = vband(good(ip)) - ilc(good(ip))! foreground template = V-band - ilc

            !! MRN
            Call pix2vec_nest(nsmax, good(ip), vec_ip)
            vecs(:, ip) = vec_ip(:)

        End Do

        !! MRN
        ! --- original
        !       do ip = 0,ngood-1
        !          do jp = ip,ngood-1
        ! --- RF
        Do jp = 0, ngood - 1
            Do ip = 0, jp
                ! ---
                x = Sum(vecs(:, ip)*vecs(:, jp))
                p(0) = 1d0
                p(1) = x
                Do l = 2, nlmax
                    one_over_l = 1d0 / l
                    p(l) = (2d0-one_over_l) * x * p(l-1) - (1d0-one_over_l) * p(l-2)
                End Do
                Do l = 0, nlmax
                    p(l) = p(l) * (2d0*l+1d0) / (4d0*3.1415927d0)
                End Do
                C0(ip, jp) = Sum(cl(nlhigh+1:nlmax)*p(nlhigh+1:nlmax)) &
                    + 100. * cl(2) * (p(0)+p(1))
            End Do
        End Do

        Deallocate(good, ilc, vband, cl)
        !DEALLOCATE(vband_clean)

        !
        ! Add random noise(1uK per pixel) to regularize the covariance matrix
        ! The map will be noise-dominated at l>40(l>20 at res3).
        !
        iseed = - 562378 ! arbitrary
        noise_rms = 1.e-3 ! mK per pixel

        Do ip = 0, ngood - 1
            C0(ip, ip) = C0(ip, ip) + noise_rms ** 2.
            t_map(ip) = t_map(ip) + noise_rms * randgauss_boxmuller(iseed)
        End Do

#endif

        If (Present(tt_ngood)) Then
            tt_ngood = ngood
        End If

#ifdef TIMING
        Call wmap_timing_end()
#endif

    End Subroutine setup_for_tt_exact

    Function tt_pixlike_dof()

        Integer :: tt_pixlike_dof
        tt_pixlike_dof = ngood

    End Function tt_pixlike_dof

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Subroutine compute_tt_pixlike(nlhigh, cl_in, chisq, lndet)
        !
        !  This subroutine returns the likelihood
        !
        Use wmap_util
        Use wmap_options, Only: tt_pixlike_lndet_offset, lowl_tt_res

        Implicit None
        Integer, Intent(In) :: nlhigh
        Real, Intent(In), Dimension(2:nlhigh) :: cl_in
        Double Precision, Intent(Out) :: chisq, lndet
        Double Precision, Allocatable, Dimension(:) :: c_inv_t_map, c_inv_f_map, cl
        Double Precision :: like
        Real :: fore_marg, fCinvf
        Double Precision :: DDOT
        Integer :: info, ip, jp, l, k, i, lun
        Double Precision :: vec_ip(3), x, p(0:64), one_over_l
        Double Precision :: dk, dindex, wt, tot_cl, xxx1, xxx2, max_diff, xxx
        Double Precision :: omega_pix

#ifdef TIMING
        Call wmap_timing_start('compute_tt_pixlike')
#endif

        Allocate(cl(2:nlhigh))

#ifdef FASTERTT

        cl(2:nlhigh) = cl_in(2:nlhigh)

        Call DGEMV('N', nzzz, nlhigh-1, 1d0, zzz, &
            nzzz, cl(2:nlhigh), 1, 0d0, yyy, 1)

        C = 0d0
        Do ip = 0, ngood - 1
            C(ip, ip) = 1d0
        End Do

        k = 1
        ! --- original
        Do ip = 0, ngood - 1
            Do jp = ip, ngood - 1
                ! --- RF
                ! **this part should not be modified because of the way "yyy" is ordered(EK)**
                !        do jp = 0,ngood-1
                !           do ip = 0,jp
                ! ---
                C(ip, jp) = C(ip, jp) + yyy(k)
                k = k + 1
            End Do
        End Do

#else

        Do l = 2, nlhigh
            cl(l) = cl_in(l) * wl(l) ** 2 * 0.5e-6 * (2*l+1) / (l*(l+1))
        End Do

        ! need to fill in only the upper half of C

        C(:, :) = C0(:, :)

        !! MRN
        ! --- original
        !       do ip = 0,ngood-1
        !          do jp = ip,ngood-1
        ! --- RF
        Do jp = 0, ngood - 1
            Do ip = 0, jp
                ! ---
                x = Sum(vecs(:, ip)*vecs(:, jp))
                p(0) = 1d0
                p(1) = x
                Do l = 2, nlhigh
                    one_over_l = 1d0 / l
                    p(l) = (2d0-one_over_l) * x * p(l-1) - (1d0-one_over_l) * p(l-2)
                End Do
                C(ip, jp) = C(ip, jp) + Sum(cl(2:nlhigh)*p(2:nlhigh))
            End Do
        End Do

#endif

        Deallocate(cl)

#ifdef TIMING
        Call wmap_timing_checkpoint('finished C')
#endif

        Call DPOTRF('U', ngood, C, ngood, info)
        If (info .Ne. 0) Then
            Call wmap_likelihood_error('tlike 1st dpotrf failed', info)
            chisq = 0d0
            lndet = 0d0
            Return
        End If

#ifdef TIMING
        Call wmap_timing_checkpoint('finished dpotrf')
#endif

        !    lndet = 0d0
        !    omega_pix = 4d0*3.145927d0/dble(12*nsmax**2)
        !    do ip=0,ngood-1
        !       lndet = lndet + 2.*log(C(ip,ip)*omega_pix)
        !    end do

        lndet = 0d0
        Do ip = 0, ngood - 1
            lndet = lndet + 2. * Log(C(ip, ip))
        End Do

#ifdef TIMING
        Call wmap_timing_checkpoint('finished lndet')
#endif

        Allocate(c_inv_t_map(0:ngood-1))
        c_inv_t_map = t_map
        Call DPOTRS('U', ngood, 1, C, ngood, c_inv_t_map, ngood, info)
        If (info .Ne. 0) Then
            Call wmap_likelihood_error('tlike 2nd dpotrs failed', info)
            chisq = 0d0
            lndet = 0d0
            Return
        End If
#ifndef FASTERTT
        Allocate(c_inv_f_map(0:ngood-1))
        c_inv_f_map = f_map
        Call DPOTRS('U', ngood, 1, C, ngood, c_inv_f_map, ngood, info)
        If (info .Ne. 0) Then
            Call wmap_likelihood_error('tlike 3rd spotrs failed', info)
            chisq = 0d0
            lndet = 0d0
            Return
        End If
        fCinvf = Sum(c_inv_f_map(:)*f_map(:)) / ngood
        !DEALLOCATE(C)
#endif

#ifdef TIMING
        Call wmap_timing_checkpoint('finished dpotrs')
#endif

        chisq = Sum(c_inv_t_map(:)*t_map(:))
#ifndef FASTERTT
        If (ifore_marg .Ne. 0) Then
            If (ifore_marg .Eq. 1) Then
                fore_marg = 1.
            Else
                fore_marg = 1.e6
            End If
            chisq = chisq - (Sum(c_inv_f_map(:)*t_map(:))) ** 2. / ngood &
                / (1./fore_marg+fCinvf)
            lndet = lndet + Log(1./fore_marg+fCinvf)
        End If
        Deallocate(c_inv_f_map)
#endif
        like = (chisq+lndet) / 2.d0
        chisq = chisq / 2.
        lndet = (lndet-tt_pixlike_lndet_offset(lowl_tt_res)) / 2.
        Deallocate(c_inv_t_map)

#ifdef TIMING
        Call wmap_timing_end()
#endif
    End Subroutine compute_tt_pixlike
    !
    ! The following two subroutines are taken from ran_tools in HEALPix.
    !
    !=======================================================================
    Function randgauss_boxmuller(iseed)
        Implicit None
        Integer, Intent(Inout) :: iseed !! random number state
        Real :: randgauss_boxmuller !! result
        Logical, Save :: empty = .True.
        Real :: fac, rsq, v1, v2
        Real, Save :: gset !! test

        If (empty .Or. iseed < 0) Then ! bug correction, EH, March 13, 2003
            Do
                v1 = 2. * ran_mwc(iseed) - 1.
                v2 = 2. * ran_mwc(iseed) - 1.
                rsq = v1 ** 2 + v2 ** 2
                If ((rsq < 1.) .And. (rsq > 0.)) Exit
            End Do

            fac = Sqrt(-2.*Log(rsq)/rsq)
            gset = v1 * fac
            randgauss_boxmuller = v2 * fac
            empty = .False.
        Else
            randgauss_boxmuller = gset
            empty = .True.
        End If
    End Function randgauss_boxmuller
    !=======================================================================
    Function ran_mwc(iseed)
        Implicit None
        Integer, Intent(Inout) :: iseed !! random number state
        Real :: ran_mwc !! result

        Integer :: i, iseedl, iseedu, mwc, combined
        Integer, Save :: upper, lower, shifter
        Integer, Parameter :: mask16 = 65535, mask30 = 2147483647
        Real, Save :: small
        Logical, Save :: first = .True.

        If (first .Or. (iseed <= 0)) Then
            If (iseed == 0) iseed = - 1
            iseed = Abs(iseed)
            small = Nearest(1.,-1.) / mask30

            ! Users often enter small seeds - I spread them out using the
            ! Marsaglia shifter a few times.
            shifter = iseed
            Do i = 1, 9
                shifter = Ieor(shifter, Ishft(shifter, 13))
                shifter = Ieor(shifter, Ishft(shifter,-17))
                shifter = Ieor(shifter, Ishft(shifter, 5))
            End Do

            iseedu = Ishft(shifter,-16)
            upper = Ishft(iseedu+8765, 16) + iseedu !This avoids the fixed points.
            iseedl = Iand(shifter, mask16)
            lower = Ishft(iseedl+4321, 16) + iseedl !This avoids the fixed points.

            first = .False.
        End If

        Do
            shifter = Ieor(shifter, Ishft(shifter, 13))
            shifter = Ieor(shifter, Ishft(shifter,-17))
            shifter = Ieor(shifter, Ishft(shifter, 5))

            upper = 36969 * Iand(upper, mask16) + Ishft(upper,-16)
            lower = 18000 * Iand(lower, mask16) + Ishft(lower,-16)

            mwc = Ishft(upper, 16) + Iand(lower, mask16)

            combined = Iand(mwc, mask30) + Iand(shifter, mask30)

            ran_mwc = small * Iand(combined, mask30)
            If (ran_mwc /= 0.) Exit
        End Do
    End Function ran_mwc
    !
    ! The following two subroutines are taken from pix_tools in HEALPix.
    !
    !=======================================================================
    Subroutine pix2vec_nest(nside, ipix, vector, vertex)
        Use wmap_util
        Implicit None
        !=======================================================================
        !     renders vector(x,y,z) coordinates of the nominal pixel center
        !     for the pixel number ipix(NESTED scheme)
        !     given the map resolution parameter nside
        !     also returns the(x,y,z) position of the 4 pixel vertices(=corners)
        !     in the order N,W,S,E
        !=======================================================================
        Integer, Intent(In) :: nside, ipix
        Double Precision, Intent(Out), Dimension(1:) :: vector
        Double Precision, Intent(Out), Dimension(1:, 1:), Optional :: vertex

        Integer :: npix, npface, &
            & ipf, ip_low, ip_trunc, ip_med, ip_hi, &
            & jrt, jr, nr, jpt, jp, kshift, nl4
        Double Precision :: z, fn, fact1, fact2, sth, phi

        Integer :: ix, iy, face_num
        !     common /xy_nest/ ix, iy, face_num ! can be useful to calling routine

        ! coordinate of the lowest corner of each face
        Integer, Dimension(1:12) :: jrll = (/ 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 /)! in unit of nside
        Integer, Dimension(1:12) :: jpll = (/ 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7 /)! in unit of nside/2

        Double Precision :: phi_nv, phi_wv, phi_sv, phi_ev, phi_up, phi_dn
        Double Precision :: z_nv, z_sv, sth_nv, sth_sv
        Double Precision :: hdelta_phi
        Integer :: iphi_mod, iphi_rat
        Logical :: do_vertex
        Double Precision :: halfpi = 1.570796326794896619231321691639751442099d0
        Double Precision :: pi = 3.141592653589793238462643383279502884197d0
        Integer :: info
        !-----------------------------------------------------------------------
        If (nside < 1 .Or. nside > 1024) Call wmap_likelihood_error('nside out of range', info)
        npix = 12 * nside ** 2
        If (ipix < 0 .Or. ipix > npix-1) Call wmap_likelihood_error('ipix out of range', info)

        !     initiates the array for the pixel number -> (x,y) mapping
        If (pix2x(1023) <= 0) Call mk_pix2xy()

        fn = Dble(nside)
        fact1 = 1.0d0 / (3.0d0*fn*fn)
        fact2 = 2.0d0 / (3.0d0*fn)
        nl4 = 4 * nside

        do_vertex = .False.
        If (Present(vertex)) Then
            If (Size(vertex, dim=1) >= 3 .And. Size(vertex, dim=2) >= 4) Then
                do_vertex = .True.
            Else
                Call wmap_likelihood_error(' pix2vec_ring : vertex array has wrong size ', info)
            End If
        End If

        !     finds the face, and the number in the face
        npface = nside ** 2

        face_num = ipix / npface ! face number in {0,11}
        ipf = Modulo(ipix, npface)! pixel number in the face {0,npface-1}

        !     finds the x,y on the face(starting from the lowest corner)
        !     from the pixel number
        ip_low = Modulo(ipf, 1024)! content of the last 10 bits
        ip_trunc = ipf / 1024 ! truncation of the last 10 bits
        ip_med = Modulo(ip_trunc, 1024)! content of the next 10 bits
        ip_hi = ip_trunc / 1024 ! content of the high weight 10 bits

        ix = 1024 * pix2x(ip_hi) + 32 * pix2x(ip_med) + pix2x(ip_low)
        iy = 1024 * pix2y(ip_hi) + 32 * pix2y(ip_med) + pix2y(ip_low)

        !     transforms this in(horizontal, vertical) coordinates
        jrt = ix + iy ! 'vertical' in {0,2*(nside-1)}
        jpt = ix - iy ! 'horizontal' in {-nside+1,nside-1}

        !     computes the z coordinate on the sphere
        jr = jrll(face_num+1) * nside - jrt - 1 ! ring number in {1,4*nside-1}

        nr = nside ! equatorial region(the most frequent)
        z = (2*nside-jr) * fact2
        kshift = Modulo(jr-nside, 2)
        If (do_vertex) Then
            z_nv = (2*nside-jr+1) * fact2
            z_sv = (2*nside-jr-1) * fact2
            If (jr == nside) Then ! northern transition
                z_nv = 1.0d0 - (nside-1) ** 2 * fact1
            Else If (jr == 3*nside) Then ! southern transition
                z_sv = - 1.0d0 + (nside-1) ** 2 * fact1
            End If
        End If
        If (jr < nside) Then ! north pole region
            nr = jr
            z = 1.0d0 - nr * nr * fact1
            kshift = 0
            If (do_vertex) Then
                z_nv = 1.0d0 - (nr-1) ** 2 * fact1
                z_sv = 1.0d0 - (nr+1) ** 2 * fact1
            End If
        Else If (jr > 3*nside) Then ! south pole region
            nr = nl4 - jr
            z = - 1.0d0 + nr * nr * fact1
            kshift = 0
            If (do_vertex) Then
                z_nv = - 1.0d0 + (nr+1) ** 2 * fact1
                z_sv = - 1.0d0 + (nr-1) ** 2 * fact1
            End If
        End If

        !     computes the phi coordinate on the sphere, in [0,2Pi]
        jp = (jpll(face_num+1)*nr+jpt+1+kshift) / 2 ! 'phi' number in the ring in {1,4*nr}
        If (jp > nl4) jp = jp - nl4
        If (jp < 1) jp = jp + nl4

        phi = (jp-(kshift+1)*0.5d0) * (halfpi/nr)

        sth = Sqrt((1.0d0-z)*(1.0d0+z))
        vector(1) = sth * Cos(phi)
        vector(2) = sth * Sin(phi)
        vector(3) = z

        If (do_vertex) Then
            phi_nv = phi
            phi_sv = phi

            phi_up = 0.0d0
            iphi_mod = Modulo(jp-1, nr)! in {0,1,... nr-1}
            iphi_rat = (jp-1) / nr ! in {0,1,2,3}
            If (nr > 1) phi_up = halfpi * (iphi_rat+iphi_mod/Dble(nr-1))
            phi_dn = halfpi * (iphi_rat+(iphi_mod+1)/Dble(nr+1))
            If (jr < nside) Then ! North polar cap
                phi_nv = phi_up
                phi_sv = phi_dn
            Else If (jr > 3*nside) Then ! South polar cap
                phi_nv = phi_dn
                phi_sv = phi_up
            Else If (jr == nside) Then ! North transition
                phi_nv = phi_up
            Else If (jr == 3*nside) Then ! South transition
                phi_sv = phi_up
            End If

            hdelta_phi = pi / (4.0d0*nr)

            ! west vertex
            phi_wv = phi - hdelta_phi
            vertex(1, 2) = sth * Cos(phi_wv)
            vertex(2, 2) = sth * Sin(phi_wv)
            vertex(3, 2) = z

            ! east vertex
            phi_ev = phi + hdelta_phi
            vertex(1, 4) = sth * Cos(phi_ev)
            vertex(2, 4) = sth * Sin(phi_ev)
            vertex(3, 4) = z

            ! north vertex
            sth_nv = Sqrt((1.0d0-z_nv)*(1.0d0+z_nv))
            vertex(1, 1) = sth_nv * Cos(phi_nv)
            vertex(2, 1) = sth_nv * Sin(phi_nv)
            vertex(3, 1) = z_nv

            ! south vertex
            sth_sv = Sqrt((1.0d0-z_sv)*(1.0d0+z_sv))
            vertex(1, 3) = sth_sv * Cos(phi_sv)
            vertex(2, 3) = sth_sv * Sin(phi_sv)
            vertex(3, 3) = z_sv
        End If

        Return
    End Subroutine pix2vec_nest
    !=======================================================================
    Subroutine mk_pix2xy()
        Implicit None
        !=======================================================================
        !     constructs the array giving x and y in the face from pixel number
        !     for the nested(quad-cube like) ordering of pixels
        !
        !     the bits corresponding to x and y are interleaved in the pixel number
        !     one breaks up the pixel number by even and odd bits
        !=======================================================================
        Integer :: kpix, jpix, ix, iy, ip, id

        !cc cf block data      data      pix2x(1023) /0/
        !-----------------------------------------------------------------------
        !      print *, 'initiate pix2xy'
        Do kpix = 0, 1023 ! pixel number
            jpix = kpix
            ix = 0
            iy = 0
            ip = 1 ! bit position(in x and y)
            !        do while(jpix/=0) ! go through all the bits
            Do
                If (jpix == 0) Exit! go through all the bits
                id = Modulo(jpix, 2)! bit value(in kpix), goes in ix
                jpix = jpix / 2
                ix = id * ip + ix

                id = Modulo(jpix, 2)! bit value(in kpix), goes in iy
                jpix = jpix / 2
                iy = id * ip + iy

                ip = 2 * ip ! next bit(in x and y)
            End Do
            pix2x(kpix) = ix ! in 0,31
            pix2y(kpix) = iy ! in 0,31
        End Do
    End Subroutine mk_pix2xy
End Module wmap_tlike
