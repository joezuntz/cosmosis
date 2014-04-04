! ===========================================================================
MODULE wmap_tlike
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
! -- White noise (1uK per pixel at nside=16) is added to the degraded maps.
! -- Fore_Marg is set to 2 explicitly.
!
! E. Komatsu, March 5, 2009
! -- Changed the orders of do-loops for a better performance 
!    [thanks to Raphael Flauger]
!
! E. Komatsu, December 20, 2009
! -- 7-yr version
!===========================================================================

  implicit none
  private
  public :: setup_for_tt_exact, compute_tt_pixlike, tt_pixlike_dof

  INTEGER :: ngood,nsmax, nzzz
  REAL, ALLOCATABLE :: wl(:)
#ifdef FASTERTT
  double precision, ALLOCATABLE :: t_map(:) ! ILC
#else
  REAL, ALLOCATABLE :: t_map(:) ! ILC
  REAL, ALLOCATABLE :: f_map(:) ! difference between ILC and V
#endif
  DOUBLE PRECISION, ALLOCATABLE, dimension(:,:) :: C0, C  ! C0 = monopole + dipole marginalized
  DOUBLE PRECISION, ALLOCATABLE, dimension(:,:), target :: zzz
  DOUBLE PRECISION, allocatable :: vecs(:,:), yyy(:)
  integer, dimension(0:1023) :: pix2x=0, pix2y=0 ! for pix2vec
  INTEGER, parameter :: ifore_marg = 2
  !
  ! ifore_marg = 0 (no marginalization)
  ! ifore_marg = 1  (assume that the uncertainty is the difference between
  !			V and ILC)
  ! ifore_marg = 2  (no prior on amplitude of foreground unc.)
  !  
CONTAINS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  SUBROUTINE SETUP_FOR_TT_EXACT(nlhigh, tt_ngood)
    !
    ! - READ IN DEGRADED Kp2 MASK (nside=16)
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
    use wmap_util
    use wmap_options, only : WMAP_data_dir, lowl_tt_res
 
   IMPLICIT NONE
    include 'read_archive_map.fh'
    include 'read_fits.fh'
    INTEGER, INTENT(IN) :: nlhigh ! vary cl's from 2 : nlhigh
    integer, intent(out), optional :: tt_ngood
    integer :: ires != 4           ! resolution for low l TT code
    INTEGER :: npix,np,status,nlmax,m, lun
    INTEGER :: ip,jp,ipix,l,ll,jpix
    CHARACTER(len=120) :: filename(0:6), ifn, ofn
    LOGICAL :: exist
    INTEGER, ALLOCATABLE, dimension(:) :: good
    REAL, ALLOCATABLE, dimension(:) :: dummy,ilc,vband,mask
!    REAL, ALLOCATABLE, dimension(:) :: vband_clean
    REAL, ALLOCATABLE, dimension(:) :: cl_fiducial
    REAL, ALLOCATABLE, dimension(:,:,:) :: Plin
    DOUBLE PRECISION, ALLOCATABLE, dimension(:) :: cl
    REAL :: noise_rms,window_th,lc,theta_fwhm!=9.1831
    INTEGER :: iseed, k, stat
    DOUBLE PRECISION :: vec_ip(3),x,p(0:64), one_over_l
    DOUBLE PRECISION, pointer, dimension(:,:) :: Dptr2

#ifdef TIMING
	call wmap_timing_start( 'setup_for_tt_exact' )
#endif

	if ( nlhigh /= 30 ) then
		print *, "error: set lowl_max=30 when use_gibbs=.false."
		stop
	end if

#ifdef FASTERTT

	if ( ifore_marg /= 2 ) then
		print *, '*** FASTERTT currently requires ifore_marg=2'
		stop
	end if

	ngood = 957

	ALLOCATE(C(0:ngood-1,0:ngood-1))
	ALLOCATE(t_map(0:ngood-1))

	ifn = trim(WMAP_data_dir)//"/lowlT/faster/compressed_t_map_f2_7yr.txt"
	call get_free_lun( lun )
	open(lun,file=ifn,status='old',action='read')
	do ip = 0,ngood-1
		read(lun,*) t_map(ip)
	end do
	close(lun)

	nzzz = (ngood*(ngood+1))/2
	allocate( zzz(nzzz,nlhigh-1) )
	allocate( yyy(nzzz) )

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
	ifn = trim(WMAP_data_dir)//"lowlT/faster/compressed_tt_matrices_f2_7yr.fits"
	Call Read_FITS_Double_2D (ifn, Dptr2, stat)
	If (stat .NE. 0) Then
		Print *, 'Error ', stat, ' while reading ', trim(ifn)
		Stop
	End If

#else

	ires = lowl_tt_res
	!filename(0)=trim(WMAP_data_dir)//'wmap_lcdm_pl_model_yr1_v1.dat'
	filename(0) = trim(WMAP_data_dir)//'test_cls_v4.dat'

	select case(lowl_tt_res)
!	case(3)
!		theta_fwhm=18.3
!		filename(1)=trim(WMAP_data_dir)//'mask/mask_r3_kp2.fits'
!		filename(2)=trim(WMAP_data_dir)//'maps/low_resolution_map_fs_r3_ilc_smooth_18.3deg.fits' ! ILC map
!		filename(3)=trim(WMAP_data_dir)//'maps/low_resolution_map_fs_r3_vband_smooth_18.3deg.fits' ! raw V-band map for foreground marginalization
!		filename(4)=trim(WMAP_data_dir)//'healpix_data/pixel_window_n0008.txt'
!		!filename(5)=trim(WMAP_data_dir)//'pl_res3_sp.fits'
!		!filename(6)=trim(WMAP_data_dir)//'maps/low_resolution_map_fs_r3_vband_clean_smooth_18.3deg.fits' ! template-cleaned V-band map -- not used
	case(4)
		theta_fwhm=9.1831
	
		filename(1)=trim(WMAP_data_dir)//'lowlT/pixlike/mask_r4_kq_85p0.fits'
		filename(2)=trim(WMAP_data_dir)//'lowlT/pixlike/wmap_7yr_r4_ilc_smooth_9.1285deg.fits'
 		filename(3)=trim(WMAP_data_dir)//'lowlT/pixlike/wmap_7yr_r4_vband_raw_smooth_9.1831deg.fits'


		filename(4)=trim(WMAP_data_dir)//'healpix_data/pixel_window_n0016.txt'
		!filename(5)=trim(WMAP_data_dir)//'' ! used only when ires=3
		!filename(6)=trim(WMAP_data_dir)//'maps/wmap_fs_r4_vband_clean_smooth_9.1831deg.fits'! smoothed cleaned V-band -- not used
	case default
		call wmap_likelihood_error( 'bad lowl_tt_res', lowl_tt_res )
		stop
	end select

	lc = sqrt(8.*alog(2.))/theta_fwhm/3.14159*180.

    !
    !       READ IN cl's for fiducial model (used for high l)
    !
    ALLOCATE(cl_fiducial(2:512))
    INQUIRE(file=filename(0),exist=exist)
    IF(.not.exist) THEN
       write(*,*) ' unable to open '//trim(filename(0))
       STOP
    ENDIF
    call get_free_lun( lun )
    OPEN(lun,file=filename(0),status='old')
    do l = 2,512
       read(lun,*) ll,cl_fiducial(l)
       cl_fiducial(ll) = cl_fiducial(ll)*1.e-6*2.*3.1415927/(ll*(ll+1))
    end do
    CLOSE(lun)
    !
    ! READ IN LOW-RESOLUTION Kp2 MASK
    !
    nsmax = 2**ires
    nlmax = 4*nsmax
    npix = 12*nsmax*nsmax
    ALLOCATE(dummy(0:npix-1),mask(0:npix-1))
    call read_archive_map(filename(1),dummy,mask,np,status)
    if(status.ne.0.or.np.ne.npix) then
       write(*,*) ' error reading mask'
       stop
    endif
    ngood = sum(mask)
    ALLOCATE(good(0:ngood-1))
    ip = 0
    do ipix = 0,12*nsmax*nsmax-1
       if(mask(ipix).eq.1) then
          good(ip) = ipix
          ip = ip+1
       endif
    end do
    if(ip.ne.ngood) STOP
    !
    !  READ IN LOW-RESOLUTION ILC AND V-BAND MAPS
    !  - smoothing scale is FWHM = 9.1 degrees
    !
    ALLOCATE(ilc(0:npix-1),vband(0:npix-1))
!    ALLOCATE(vband_clean(0:npix-1))
    call read_archive_map(filename(2),ilc,dummy,np,status)
    if(status.ne.0.or.np.ne.npix) then
       write(*,*) ' error reading ILC map'
       stop
    endif
    call read_archive_map(filename(3),vband,dummy,np,status)
    if(status.ne.0.or.np.ne.npix) then
       write(*,*) ' error reading V-band map'
       stop
    endif
!    call read_archive_map(filename(6),vband_clean,dummy,np,status)
!    if(status.ne.0.or.np.ne.npix) then
!       write(*,*) ' error reading cleaned V-band map'
!       stop
!    endif
    DEALLOCATE(dummy)
   
    !
    !  CALCULATE COMBINED BEAM
    !
    ALLOCATE(wl(2:1024))
    call get_free_lun( lun )
    OPEN(lun,file=filename(4),status='old')
    read(lun,*) window_th
    read(lun,*) window_th
    do l = 2,nlmax
       read(lun,*) window_th
       wl(l) = window_th*exp(-0.5*l*(l+1)/lc**2.)
    end do
    CLOSE(lun)
    ALLOCATE(cl(2:nlmax))
    cl(2:nlmax) = cl_fiducial(2:nlmax)*wl(2:nlmax)**2.
    DEALLOCATE(cl_fiducial)

    ALLOCATE(C0(0:ngood-1,0:ngood-1))
    ALLOCATE(C(0:ngood-1,0:ngood-1))
    ALLOCATE(t_map(0:ngood-1))
    ALLOCATE(f_map(0:ngood-1))
    allocate( vecs(3,0:ngood-1) )

    C0 = 0d0

    do ip = 0,ngood-1
       t_map(ip) = ilc(good(ip))
!!$ t_map(ip) = vband_clean(good(ip)) !!$ don't use cleaned V. use ILC!
       f_map(ip) = vband(good(ip))-ilc(good(ip)) ! foreground template = V-band - ilc
      
	!! MRN
	call pix2vec_nest( nsmax, good(ip), vec_ip )
	vecs(:,ip) = vec_ip(:)

    end do

       !! MRN
! --- original
!       do ip = 0,ngood-1
!          do jp = ip,ngood-1
! --- RF
       do jp = 0,ngood-1
          do ip = 0,jp
! ---
		x = sum(vecs(:,ip)*vecs(:,jp))
                p(0) = 1d0
                p(1) = x
                do l = 2,nlmax
			one_over_l = 1d0/l
                        p(l) = (2d0-one_over_l)*x*p(l-1)-(1d0-one_over_l)*p(l-2)
                end do
                do l = 0,nlmax
                        p(l) = p(l)*(2d0*l+1d0)/(4d0*3.1415927d0)
                end do
          	C0(ip,jp) = sum(cl(nlhigh+1:nlmax)*p(nlhigh+1:nlmax)) &
			+ cl(2)*(p(0)+p(1))
          end do
       end do

    DEALLOCATE(good,ilc,vband,cl)
    !DEALLOCATE(vband_clean)

    !
    ! Add random noise (1uK per pixel) to regularize the covariance matrix
    ! The map will be noise-dominated at l>40 (l>20 at res3).
    !
    iseed = -562378 ! arbitrary
    noise_rms = 1.e-3 ! mK per pixel
 
    do ip = 0,ngood-1
       C0(ip,ip) = C0(ip,ip) + noise_rms**2.
       t_map(ip) = t_map(ip) + noise_rms*randgauss_boxmuller(iseed)
    enddo

#endif

	if ( present(tt_ngood) ) then
		tt_ngood = ngood
	end if
 
#ifdef TIMING
	call wmap_timing_end()
#endif

  END SUBROUTINE SETUP_FOR_TT_EXACT

  function tt_pixlike_dof()

	integer :: tt_pixlike_dof
	tt_pixlike_dof = ngood

  end function

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  SUBROUTINE COMPUTE_TT_PIXLIKE(nlhigh,cl_in,chisq,lndet)
    !
    !  This subroutine returns the likelihood 
    !
    use wmap_util
    use wmap_options, only : tt_pixlike_lndet_offset, lowl_tt_res

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nlhigh
    REAL, INTENT(IN), DIMENSION(2:nlhigh) :: cl_in
    DOUBLE PRECISION,INTENT(OUT) :: chisq,lndet
    DOUBLE PRECISION, ALLOCATABLE, dimension(:) :: c_inv_t_map,c_inv_f_map,cl
    DOUBLE PRECISION:: like
    REAL :: fore_marg,fCinvf
    DOUBLE PRECISION :: DDOT
    INTEGER :: info,ip,jp, l, k, i, lun
    DOUBLE PRECISION :: vec_ip(3),x,p(0:64), one_over_l
    double precision :: dk, dindex, wt, tot_cl, xxx1, xxx2, max_diff, xxx
    double precision :: omega_pix
   
#ifdef TIMING
	call wmap_timing_start( 'compute_tt_pixlike' )
#endif

    ALLOCATE(cl(2:nlhigh))

#ifdef FASTERTT

	cl(2:nlhigh) = cl_in(2:nlhigh)

	call DGEMV( 'N', nzzz, nlhigh-1, 1d0, zzz, &
		nzzz, cl(2:nlhigh), 1, 0d0, yyy, 1 )

	C = 0d0
	do ip = 0,ngood-1
		C(ip,ip) = 1d0
	end do

	k = 1
! --- original
	do ip = 0,ngood-1
	   do jp = ip,ngood-1
! --- RF 
! **this part should not be modified because of the way "yyy" is ordered (EK)**
!        do jp = 0,ngood-1
!           do ip = 0,jp
! ---
		C(ip,jp) = C(ip,jp) + yyy(k)
		k = k + 1
	end do
	end do

#else

    do l =2,nlhigh
       cl(l) = cl_in(l)*wl(l)**2*0.5e-6*(2*l+1)/(l*(l+1))
    end do

    ! need to fill in only the upper half of C

	C(:,:) = C0(:,:)

       !! MRN
! --- original
!       do ip = 0,ngood-1
!          do jp = ip,ngood-1
! --- RF
       do jp = 0,ngood-1
          do ip = 0,jp
! ---
		x = sum(vecs(:,ip)*vecs(:,jp))
                p(0) = 1d0
                p(1) = x
                do l = 2,nlhigh
			one_over_l = 1d0/l
                        p(l) = (2d0-one_over_l)*x*p(l-1)-(1d0-one_over_l)*p(l-2)
                end do
          	C(ip,jp) = C(ip,jp) + sum(cl(2:nlhigh)*p(2:nlhigh))
          end do
       end do

#endif

    DEALLOCATE(cl)

#ifdef TIMING
	call wmap_timing_checkpoint( 'finished C' )
#endif

    CALL DPOTRF('U',ngood,C,ngood,info) 
    IF(info.NE.0)then
       call wmap_likelihood_error( 'tlike 1st dpotrf failed', info )
       chisq = 0d0
       lndet = 0d0
       return
    endif

#ifdef TIMING
	call wmap_timing_checkpoint( 'finished dpotrf' )
#endif

!    lndet = 0d0
!    omega_pix = 4d0*3.145927d0/dble(12*nsmax**2)
!    do ip=0,ngood-1
!       lndet = lndet + 2.*log(C(ip,ip)*omega_pix)
!    end do

    lndet = 0d0
    do ip=0,ngood-1
       lndet = lndet + 2.*log(C(ip,ip))
    end do

#ifdef TIMING
	call wmap_timing_checkpoint( 'finished lndet' )
#endif

    ALLOCATE(c_inv_t_map(0:ngood-1))
    c_inv_t_map = t_map
    call DPOTRS('U', ngood, 1, C, ngood, c_inv_t_map, ngood, INFO)
    IF(info.NE.0)then
       call wmap_likelihood_error( 'tlike 2nd dpotrs failed', info )
       chisq = 0d0
       lndet = 0d0
       return
    endif
#ifndef FASTERTT
    ALLOCATE(c_inv_f_map(0:ngood-1))
    c_inv_f_map = f_map
    call DPOTRS('U', ngood, 1, C, ngood, c_inv_f_map, ngood, INFO)
    IF(info.NE.0)then
       call wmap_likelihood_error( 'tlike 3rd spotrs failed', info )
       chisq = 0d0
       lndet = 0d0
       return
    endif
    fCinvf = sum(c_inv_f_map(:)*f_map(:))/ngood
    !DEALLOCATE(C)
#endif

#ifdef TIMING
	call wmap_timing_checkpoint( 'finished dpotrs' )
#endif

    chisq = sum(c_inv_t_map(:)*t_map(:)) 
#ifndef FASTERTT
    if(ifore_marg.ne.0) then
       if(ifore_marg.eq.1)then
          fore_marg = 1.
       else
          fore_marg = 1.e6
       endif
       chisq = chisq - (sum(c_inv_f_map(:)*t_map(:)))**2./ngood &
            /( 1./fore_marg + fCinvf )
       lndet = lndet + log(1./fore_marg+fCinvf)
    endif
    DEALLOCATE(c_inv_f_map)
#endif
    like = (chisq+lndet)/2.d0
    chisq=chisq/2.
    lndet=(lndet-tt_pixlike_lndet_offset(lowl_tt_res))/2.
    DEALLOCATE(c_inv_t_map)

#ifdef TIMING
	call wmap_timing_end()
#endif
  END subroutine COMPUTE_TT_PIXLIKE
!
! The following two subroutines are taken from ran_tools in HEALPix.
!
  !=======================================================================
  function randgauss_boxmuller(iseed)
    implicit none
    integer, intent(inout) :: iseed  !! random number state
    real :: randgauss_boxmuller       !! result
    logical, save :: empty=.true.
    real :: fac,rsq,v1,v2
    real,save :: gset !! test
    
    if (empty .or. iseed < 0) then ! bug correction, EH, March 13, 2003
       do
          v1 = 2.*ran_mwc(iseed) - 1.
          v2 = 2.*ran_mwc(iseed) - 1.
          rsq = v1**2+v2**2
          if((rsq<1.) .and. (rsq>0.)) exit
       end do
       
       fac = sqrt(-2.*log(rsq)/rsq)
       gset = v1*fac
       randgauss_boxmuller = v2*fac
       empty = .false.
    else
       randgauss_boxmuller = gset
       empty = .true.
    endif
  end function randgauss_boxmuller
  !=======================================================================
  function ran_mwc(iseed)
    implicit none
    integer, intent(inout):: iseed !! random number state
    real :: ran_mwc                 !! result
    
    integer :: i,iseedl,iseedu,mwc,combined
    integer,save :: upper,lower,shifter
    integer,parameter :: mask16=65535,mask30=2147483647
    real,save :: small
    logical, save :: first=.true.
    
    if (first .or. (iseed<=0)) then
       if (iseed==0) iseed=-1
       iseed = abs(iseed)
       small = nearest (1.,-1.)/mask30
       
       ! Users often enter small seeds - I spread them out using the
       ! Marsaglia shifter a few times.
       shifter=iseed
       do i=1,9
          shifter=ieor(shifter,ishft(shifter,13))
          shifter=ieor(shifter,ishft(shifter,-17))
          shifter=ieor(shifter,ishft(shifter,5))
       enddo
       
       iseedu=ishft(shifter,-16)
       upper=ishft(iseedu+8765,16)+iseedu !This avoids the fixed points.
       iseedl=iand(shifter,mask16)
       lower=ishft(iseedl+4321,16)+iseedl !This avoids the fixed points.
       
       first=.false.
    endif

    do
       shifter=ieor(shifter,ishft(shifter,13))
       shifter=ieor(shifter,ishft(shifter,-17))
       shifter=ieor(shifter,ishft(shifter,5))
       
       upper=36969*iand(upper,mask16)+ishft(upper,-16)
       lower=18000*iand(lower,mask16)+ishft(lower,-16)
       
       mwc=ishft(upper,16)+iand(lower,mask16)
       
       combined=iand(mwc,mask30)+iand(shifter,mask30)
       
       ran_mwc=small*iand(combined,mask30)
       if(ran_mwc/=0.) exit
    end do
  end function ran_mwc
!
! The following two subroutines are taken from pix_tools in HEALPix.
!
  !=======================================================================
  subroutine pix2vec_nest(nside, ipix, vector, vertex)
    use wmap_util
    implicit none
    !=======================================================================
    !     renders vector (x,y,z) coordinates of the nominal pixel center
    !     for the pixel number ipix (NESTED scheme)
    !     given the map resolution parameter nside
    !     also returns the (x,y,z) position of the 4 pixel vertices (=corners)
    !     in the order N,W,S,E
    !=======================================================================
    INTEGER, INTENT(IN) :: nside, ipix
    DOUBLE PRECISION, INTENT(OUT), dimension(1:) :: vector
    DOUBLE PRECISION,     INTENT(OUT),dimension(1:,1:), optional :: vertex

    INTEGER :: npix, npface, &
         &     ipf, ip_low, ip_trunc, ip_med, ip_hi, &
         &     jrt, jr, nr, jpt, jp, kshift, nl4
    DOUBLE PRECISION :: z, fn, fact1, fact2, sth, phi

    INTEGER ::  ix, iy, face_num
!     common /xy_nest/ ix, iy, face_num ! can be useful to calling routine

    ! coordinate of the lowest corner of each face
    INTEGER, dimension(1:12) :: jrll = (/ 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 /) ! in unit of nside
    INTEGER, dimension(1:12) :: jpll = (/ 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7 /) ! in unit of nside/2

    double precision :: phi_nv, phi_wv, phi_sv, phi_ev, phi_up, phi_dn
    double precision :: z_nv, z_sv, sth_nv, sth_sv
    double precision :: hdelta_phi
    integer :: iphi_mod, iphi_rat
    logical :: do_vertex
    double precision :: halfpi = 1.570796326794896619231321691639751442099d0
    double precision :: pi = 3.141592653589793238462643383279502884197d0
    integer :: info
    !-----------------------------------------------------------------------
    if (nside<1 .or. nside>1024) call wmap_likelihood_error('nside out of range',info)
    npix = 12 * nside**2
    if (ipix <0 .or. ipix>npix-1) call wmap_likelihood_error('ipix out of range',info)

    !     initiates the array for the pixel number -> (x,y) mapping
    if (pix2x(1023) <= 0) call mk_pix2xy()

    fn = dble(nside)
    fact1 = 1.0d0/(3.0d0*fn*fn)
    fact2 = 2.0d0/(3.0d0*fn)
    nl4   = 4*nside

    do_vertex = .false.
    if (present(vertex)) then
       if (size(vertex,dim=1) >= 3 .and. size(vertex,dim=2) >= 4) then
          do_vertex = .true.
       else
          call wmap_likelihood_error(' pix2vec_ring : vertex array has wrong size ',info)
       endif
    endif

    !     finds the face, and the number in the face
    npface = nside**2

    face_num = ipix/npface  ! face number in {0,11}
    ipf = MODULO(ipix,npface)  ! pixel number in the face {0,npface-1}

    !     finds the x,y on the face (starting from the lowest corner)
    !     from the pixel number
    ip_low = MODULO(ipf,1024)       ! content of the last 10 bits
    ip_trunc =   ipf/1024        ! truncation of the last 10 bits
    ip_med = MODULO(ip_trunc,1024)  ! content of the next 10 bits
    ip_hi  =     ip_trunc/1024   ! content of the high weight 10 bits

    ix = 1024*pix2x(ip_hi) + 32*pix2x(ip_med) + pix2x(ip_low)
    iy = 1024*pix2y(ip_hi) + 32*pix2y(ip_med) + pix2y(ip_low)

    !     transforms this in (horizontal, vertical) coordinates
    jrt = ix + iy  ! 'vertical' in {0,2*(nside-1)}
    jpt = ix - iy  ! 'horizontal' in {-nside+1,nside-1}

    !     computes the z coordinate on the sphere
    jr =  jrll(face_num+1)*nside - jrt - 1   ! ring number in {1,4*nside-1}

    nr = nside                  ! equatorial region (the most frequent)
    z  = (2*nside-jr)*fact2
    kshift = MODULO(jr - nside, 2)
    if (do_vertex) then
       z_nv = (2*nside-jr+1)*fact2
       z_sv = (2*nside-jr-1)*fact2
       if (jr == nside) then ! northern transition
          z_nv =  1.0d0 - (nside-1)**2 * fact1
       elseif (jr == 3*nside) then  ! southern transition
          z_sv = -1.0d0 + (nside-1)**2 * fact1
       endif
    endif
    if (jr < nside) then     ! north pole region
       nr = jr
       z = 1.0d0 - nr*nr*fact1
       kshift = 0
       if (do_vertex) then
          z_nv = 1.0d0 - (nr-1)**2*fact1
          z_sv = 1.0d0 - (nr+1)**2*fact1
       endif
    else if (jr > 3*nside) then ! south pole region
       nr = nl4 - jr
       z = - 1.0d0 + nr*nr*fact1
       kshift = 0
       if (do_vertex) then
          z_nv = - 1.0d0 + (nr+1)**2*fact1
          z_sv = - 1.0d0 + (nr-1)**2*fact1
       endif
    endif

    !     computes the phi coordinate on the sphere, in [0,2Pi]
    jp = (jpll(face_num+1)*nr + jpt + 1 + kshift)/2  ! 'phi' number in the ring in {1,4*nr}
    if (jp > nl4) jp = jp - nl4
    if (jp < 1)   jp = jp + nl4

    phi = (jp - (kshift+1)*0.5d0) * (halfpi / nr)

    sth = SQRT((1.0d0-z)*(1.0d0+z))
    vector(1) = sth * COS(phi)
    vector(2) = sth * SIN(phi)
    vector(3) = z

    if (do_vertex) then
       phi_nv = phi
       phi_sv = phi

       phi_up = 0.0d0
       iphi_mod = MODULO(jp-1, nr) ! in {0,1,... nr-1}
       iphi_rat = (jp-1) / nr      ! in {0,1,2,3}
       if (nr > 1) phi_up = HALFPI * (iphi_rat +  iphi_mod   /dble(nr-1))
       phi_dn             = HALFPI * (iphi_rat + (iphi_mod+1)/dble(nr+1))
       if (jr < nside) then            ! North polar cap
          phi_nv = phi_up
          phi_sv = phi_dn
       else if (jr > 3*nside) then     ! South polar cap
          phi_nv = phi_dn
          phi_sv = phi_up
       else if (jr == nside) then      ! North transition
          phi_nv = phi_up
       else if (jr == 3*nside) then    ! South transition
          phi_sv = phi_up
       endif

       hdelta_phi = PI / (4.0d0*nr)

       ! west vertex
       phi_wv      = phi - hdelta_phi
       vertex(1,2) = sth * COS(phi_wv)
       vertex(2,2) = sth * SIN(phi_wv)
       vertex(3,2) = z

       ! east vertex
       phi_ev      = phi + hdelta_phi
       vertex(1,4) = sth * COS(phi_ev)
       vertex(2,4) = sth * SIN(phi_ev)
       vertex(3,4) = z

       ! north vertex
       sth_nv = SQRT((1.0d0-z_nv)*(1.0d0+z_nv))
       vertex(1,1) = sth_nv * COS(phi_nv)
       vertex(2,1) = sth_nv * SIN(phi_nv)
       vertex(3,1) = z_nv

       ! south vertex
       sth_sv = SQRT((1.0d0-z_sv)*(1.0d0+z_sv))
       vertex(1,3) = sth_sv * COS(phi_sv)
       vertex(2,3) = sth_sv * SIN(phi_sv)
       vertex(3,3) = z_sv
    endif

    return
  end subroutine pix2vec_nest
  !=======================================================================
  subroutine mk_pix2xy()
    implicit none
    !=======================================================================
    !     constructs the array giving x and y in the face from pixel number
    !     for the nested (quad-cube like) ordering of pixels
    !
    !     the bits corresponding to x and y are interleaved in the pixel number
    !     one breaks up the pixel number by even and odd bits
    !=======================================================================
    INTEGER ::  kpix, jpix, ix, iy, ip, id

    !cc cf block data      data      pix2x(1023) /0/
    !-----------------------------------------------------------------------
    !      print *, 'initiate pix2xy'
    do kpix=0,1023          ! pixel number
       jpix = kpix
       IX = 0
       IY = 0
       IP = 1               ! bit position (in x and y)
!        do while (jpix/=0) ! go through all the bits
       do
          if (jpix == 0) exit ! go through all the bits
          ID = MODULO(jpix,2)  ! bit value (in kpix), goes in ix
          jpix = jpix/2
          IX = ID*IP+IX

          ID = MODULO(jpix,2)  ! bit value (in kpix), goes in iy
          jpix = jpix/2
          IY = ID*IP+IY

          IP = 2*IP         ! next bit (in x and y)
       enddo
       pix2x(kpix) = IX     ! in 0,31
       pix2y(kpix) = IY     ! in 0,31
    enddo
  end subroutine mk_pix2xy
END MODULE wmap_tlike
