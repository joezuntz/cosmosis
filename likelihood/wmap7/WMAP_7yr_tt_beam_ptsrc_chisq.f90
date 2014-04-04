!=========================================
module wmap_tt_beam_ptsrc_chisq

! Module to calculated the correction to the TTTT chisq 
! from beam and point source corrections
!
! Written by Mike Nolta
!=========================================
	implicit none

	public :: init_tt_beam_and_ptsrc_chisq
	public :: quit_tt_beam_and_ptsrc_chisq
	public :: compute_tt_beam_and_ptsrc_chisq

	private

	integer, parameter :: n_beam_modes = 9
	integer :: nmodes, ptsrc_mode_index
	character(len=*), parameter :: &
		ifn_ptsrc_mode="highl/clps.p4v6.dat", &
		ifn_beam_modes="highl/top_ten_modes.beam_covariance_VW_combined_7yr.dat", &
		ifn_fiducial_cltt="test_cls_v4.dat"
	real(kind=8), parameter :: ptsrc_err = 0.1 !! 10%

	integer :: lmin0, lmax0
	real(kind=8), allocatable :: mode(:,:), F_mode(:,:), beam_mode(:,:)
	real(kind=8), allocatable :: fiducial_cltt(:), ptsrc_mode(:)
	real(kind=8), allocatable :: a(:), a2(:), b(:,:), c(:)
	integer, allocatable :: iwork(:)

	! beam options See Appendix of Hinshaw, et.al. (2006) for a description.
	logical :: beam_diagonal_sigma = .true.
	logical :: beam_gaussian_likelihood = .true.
	logical :: beam_fixed_fiducial_spectrum = .false.
	logical :: beam_include_beam_modes = .true.
	logical :: beam_include_ptsrc_mode = .true.

contains

  subroutine init_tt_beam_and_ptsrc_chisq( lmin, lmax )

	use wmap_util
	use wmap_options, only: WMAP_data_dir
	implicit none

	integer, intent(in) :: lmin, lmax

	integer :: lun, i, l, stat
	real(kind=8) :: x
	character(len=256) :: ifn

	lmin0 = lmin
	lmax0 = lmax

	nmodes = 0
	if ( beam_include_beam_modes ) then
		nmodes = nmodes + n_beam_modes
	end if
	if ( beam_include_ptsrc_mode ) then
		nmodes = nmodes + 1
		ptsrc_mode_index = nmodes
	end if

	if ( nmodes == 0 ) then
		return
	end if

	allocate( a(nmodes) )
	allocate( b(nmodes,nmodes) )
	allocate( c(nmodes) )
	allocate( a2(nmodes) )
	allocate( iwork(nmodes) )
	allocate( fiducial_cltt(lmin:lmax) )
	allocate( ptsrc_mode(lmin:lmax) )
	allocate( beam_mode(lmin:lmax,n_beam_modes) )
	allocate( mode(lmin:lmax,nmodes) )
	allocate( F_mode(lmin:lmax,nmodes) )

	if ( beam_include_beam_modes ) then
		ifn = trim(WMAP_data_dir)//trim(ifn_beam_modes)
		call get_free_lun( lun )
		open(lun,file=ifn,action='read',status='old',form='formatted')
		do
			read(lun,*,iostat=stat) i, l, x
			if ( stat /= 0 ) exit
			if ( i <= n_beam_modes ) then
				beam_mode(l,i) = x
			end if
		end do
		close(lun)
	end if

	if ( beam_fixed_fiducial_spectrum ) then
		ifn = trim(WMAP_data_dir)//trim(ifn_fiducial_cltt)
		call get_free_lun( lun )
		open(lun,file=ifn,action='read',status='old',form='formatted')
		do
			read(lun,*,iostat=stat) l, x
			if ( stat /= 0 ) exit
			if ( l >= lmin .and. l <= lmax ) then
				fiducial_cltt(l) = x
			end if
		end do
		close(lun)
	end if

	if ( beam_include_ptsrc_mode ) then
		ifn = trim(WMAP_data_dir)//trim(ifn_ptsrc_mode)
		call get_free_lun( lun )
		open(lun,file=ifn,action='read',status='old',form='formatted')
		do
			read(lun,*,iostat=stat) l, x
			if ( stat /= 0 ) exit
			if ( l >= lmin .and. l <= lmax ) then
				ptsrc_mode(l) = x
			end if
		end do
		close(lun)
	end if

  end subroutine

  subroutine quit_tt_beam_and_ptsrc_chisq( )

	implicit none

	if ( nmodes > 0 ) then
		deallocate( beam_mode, mode, F_mode )
		deallocate( ptsrc_mode, fiducial_cltt )
		deallocate( a, b, c, a2, iwork )
	end if

  end subroutine

  function compute_tt_beam_and_ptsrc_chisq( &
	lmin, lmax, cltt, cltt_dat, neff, fisher, z, zbar )

        use wmap_util
	implicit none

	integer, intent(in) :: lmin, lmax
	real(kind=8), dimension(lmin0:lmax0), intent(in) :: &
		cltt, cltt_dat, neff, z, zbar
	real(kind=8), dimension(lmin0:lmax0,lmin0:lmax0), intent(in) :: fisher
	real(kind=8) :: compute_tt_beam_and_ptsrc_chisq

	real(kind=8) :: dgauss, dlnnorm, dlndet
	integer :: i, j, l, l1, l2, stat

	if ( nmodes == 0 ) then
		compute_tt_beam_and_ptsrc_chisq = 0d0
		return
	end if

	mode = 0d0

	!! beam modes
	if ( beam_include_beam_modes ) then
	if ( beam_fixed_fiducial_spectrum ) then
		do i = 1,n_beam_modes
		do l = lmin,lmax
			mode(l,i) = beam_mode(l,i)*fiducial_cltt(l)
		end do
		end do
	else
		do i = 1,n_beam_modes
		do l = lmin,lmax
			mode(l,i) = beam_mode(l,i)*cltt(l)
		end do
		end do
	end if
	end if

	!! ptsrc mode
	if ( beam_include_ptsrc_mode ) then
		!print *, 'including beam mode', ptsrc_mode_index
		mode(lmin:lmax,ptsrc_mode_index) = ptsrc_err*ptsrc_mode(lmin:lmax)
		!print *, 'ptsrc_mode(l=1000) = ', mode(1000,ptsrc_mode_index)
	end if

	F_mode = 0d0
	if ( beam_diagonal_sigma ) then
		do i = 1,nmodes
		do l = lmin,lmax
			F_mode(l,i) = fisher(l,l)*mode(l,i)
		end do
		end do
	else
		do i = 1,nmodes
		do l1 = lmin,lmax
		do l2 = lmin,lmax
		!do l2 = l1-50,l1+50
			if ( l2 < lmin .or. l2 > lmax ) cycle
			if ( l2 < l1 ) then
				F_mode(l1,i) = F_mode(l1,i) + fisher(l2,l1)*mode(l2,i)
			else
				F_mode(l1,i) = F_mode(l1,i) + fisher(l1,l2)*mode(l2,i)
			end if
		end do
		end do
		end do
	end if

	a = 0d0
	b = 0d0
	do i = 1,nmodes
		do l = lmin,lmax
			a(i) = a(i) + (cltt_dat(l)-cltt(l))*F_mode(l,i)
		end do
		b(i,i) = b(i,i) + 1d0
		do j = i,nmodes
			do l = lmin,lmax
				b(i,j) = b(i,j) + mode(l,i)*F_mode(l,j)
			end do
			if ( i /= j ) b(j,i) = b(i,j)
		end do
	end do

!	print *, 'nmodes = ', nmodes
!	do i = 1,nmodes
!		print '("a(",I2,") = ",E)', i, a(i)
!	end do

	call dpotrf( 'L', nmodes, b, nmodes, stat )
	if ( stat /= 0 ) then
		call wmap_likelihood_error( 'beam/ptsrc: bad dpotrf', stat )
		compute_tt_beam_and_ptsrc_chisq = 0d0
!		print *, 'bad dpotrf'
		return
	end if

	c(:) = a(:)
	call dpotrs( 'L', nmodes, 1, b, nmodes, c, nmodes, stat )
	if ( stat /= 0 ) then
           call wmap_likelihood_error( 'beam/ptsrc: bad dpotrs', stat )	
           compute_tt_beam_and_ptsrc_chisq = 0d0
!         print *, 'bad dpotrs'
           return
	end if
	dgauss = 0d0
	do i = 1,nmodes
		dgauss = dgauss + a(i)*c(i)
	end do

	if ( beam_gaussian_likelihood ) then
		dlndet = 1d0
		do i = 1,nmodes
			dlndet = dlndet*b(i,i)**2
		end do
		dlndet = dlog(dlndet)

		!print *, 'beam chisq, lndet = ', dgauss, dlndet
		compute_tt_beam_and_ptsrc_chisq = -dgauss + dlndet
	else
		a2 = 0d0
		do i = 1,nmodes
		do l = lmin,lmax
			a2(i) = a2(i) + (z(l)-zbar(l))*(cltt(l)+neff(l))*F_mode(l,i)
		end do
		end do
		c(:) = a2(:)
		call dpotrs( 'L', nmodes, 1, b, nmodes, c, nmodes, stat )
		if ( stat /= 0 ) then
			call wmap_likelihood_error( 'beam/ptsrc: bad dpotrs', stat )
			compute_tt_beam_and_ptsrc_chisq = 0d0
!			print *, 'bad dpotrs'
			return
		end if
		dlnnorm = 0d0
		do i = 1,nmodes
			dlnnorm = dlnnorm + a2(i)*c(i)
		end do

		!print *, 'beam chisq, lndet = ', dgauss, dlnnorm, -(dgauss+2d0*dlnnorm)/3d0
		compute_tt_beam_and_ptsrc_chisq = -(dgauss + 2d0*dlnnorm)/3d0
	end if

  end function

end module

