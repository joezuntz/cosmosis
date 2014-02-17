module br_mod_dist
  implicit none

  ! *********************************************************************
  ! *      br_mod -- An F90 module for computing the Blackwell-Rao      *
  ! *                estimator given signal samples from the posterior  *
  ! *                                                                   *
  ! *                 Written by Hans Kristian Eriksen                  *
  ! *                                                                   *
  ! *                Copyright 2006, all rights reserved                *
  ! *                                                                   *
  ! *                                                                   *
  ! *   NB! The code is provided as is, and *no* guarantees are given   *
  ! *       as far as either accuracy or correctness goes.              *
  ! *                                                                   *
  ! *  If used for published results, please cite these papers:         *
  ! *                                                                   *
  ! *      - Eriksen et al. 2006, ApJ, submitted, astro-ph/0606088      *
  ! *      - Chu et al. 2005, Phys. Rev. D, 71, 103002                  *
  ! *                                                                   *
  ! *********************************************************************
! E. Komatsu, March 5, 2009
! -- Changed the orders of do-loops for a better performance 
!    [thanks to Raphael Flauger]

  integer,      parameter,                     private :: i4b = selected_int_kind(8)
  integer,      parameter,                     private :: sp  = selected_real_kind(5,30)
  integer,      parameter,                     private :: dp  = selected_real_kind(12,200)
  integer,      parameter,                     private :: lgt = kind(.true.)

  integer(i4b),                                private  :: lmin, lmax, numsamples, numchain
  logical(lgt),                                private  :: first_eval
  real(dp),                                    private  :: offset
  real(dp),     allocatable, dimension(:,:,:), private  :: sigmas


contains

  ! Initialization routines
  subroutine initialize_br_mod(lmin_in, sigmas_in)
    implicit none

    integer(i4b),                            intent(in)  :: lmin_in
    real(sp),     dimension(lmin_in:,1:,1:), intent(in)  :: sigmas_in

    integer(i4b) :: i, j, ell
    
    lmin       = lmin_in
    lmax       = size(sigmas_in(:,1,1)) + lmin - 1
    numchain   = size(sigmas_in(lmin,:,1))
    numsamples = size(sigmas_in(lmin,1,:))
    first_eval = .true.

    allocate(sigmas(lmin:lmax,numchain,numsamples))
    sigmas = real(sigmas_in,dp)
! --- original
!    do ell = lmin, lmax
!       do i = 1, numchain
!          do j = 1, numsamples
! ---
    do j = 1, numsamples
       do i = 1, numchain
          do ell = lmin, lmax
             if (sigmas(ell, i, j) .le. 0.0) then
                print *, "Error: sigma value <= zero."
                print *, "sigma value is ", sigmas(ell, i, j)
                print *, "at (ell,chain,sample) = ", ell, i, j
                stop
             endif
          enddo
       enddo
    enddo

  end subroutine initialize_br_mod


  subroutine clean_up_br_mod
    implicit none

    if (allocated(sigmas)) deallocate(sigmas)

  end subroutine clean_up_br_mod



  ! Base computation routine
  subroutine compute_br_estimator(cls, lnL)
    implicit none

    real(dp), dimension(lmin:), intent(in)  :: cls
    real(dp),                   intent(out) :: lnL

    integer(i4b) :: i, j, l
    real(dp)     :: subtotal, x

    if (first_eval) then
       call compute_largest_term(cls)
       first_eval = .false.
    end if

    ! Compute the Blackwell-Rao estimator
    lnL = 0.d0

! --- original
!    do i = 1, numchain
!       do j = 1, numsamples
! --- RF

    do j = 1, numsamples
       do i = 1, numchain
! ---
          subtotal = 0.d0
          do l = lmin, lmax
             x = sigmas(l,i,j)/cls(l)
             subtotal = subtotal + &
                  & 0.5d0 * real(2*l+1,dp) * (-x + log(x)) - log(real(sigmas(l,i,j),dp))
          end do
          
          lnL = lnL + exp(subtotal-offset)
          
       end do
    end do

    if (lnL > 1e-20) then
       lnL = log(lnL)
    else
       lnL = log(1e-30)
    end if

    ! print *, lnL

  end subroutine compute_br_estimator



  ! Routine for reading the Gibbs sigma samples 
  subroutine read_gibbs_chain(filename, unit, lmax, numchains, numsamples, data)
    implicit none

    character(len=*),                            intent(in)  :: filename
    integer(i4b),                                intent(in)  :: unit
    integer(i4b),                                intent(out) :: lmax, numchains, numsamples
    real(sp),         pointer, dimension(:,:,:)              :: data

    integer(i4b)         :: l, status, blocksize, readwrite, numspec, i, j, k
    integer(i4b)         :: fpixel, group, numargs
    logical(lgt)         :: anyf
    real(sp)             :: nullval
    character(len=80)    :: comment

    integer(i4b),          dimension(4)     :: naxes
    real(sp),     pointer, dimension(:,:,:,:) :: indata

    status = 0
    readwrite = 0
    nullval = 0.

    ! numargs = 1
    numargs = 0

    ! Open the result file
    call ftopen(unit,trim(filename),readwrite,blocksize,status)

    ! Read keywords
    call ftgkyj(unit,'LMAX',     lmax,       comment,status)
    call ftgkyj(unit,'NUMSAMP',  numsamples, comment,status)
    call ftgkyj(unit,'NUMCHAIN', numchains,  comment,status)
    call ftgkyj(unit,'NUMSPEC',  numspec,    comment,status)

    allocate(data(0:lmax,numchains,numsamples))
    nullify(indata)
    allocate(indata(0:lmax,1:1,1:numchains,1:numargs+numsamples))

!!$    print *, "Allocated arrays"

    ! Read the binned power spectrum array
    group  = 1
    fpixel = 1
    call ftgpve(unit,group,fpixel,size(indata),nullval,indata,anyf,status)

!!$    print *, "Read data"

    call ftclos(unit,status)

!!$    print *, "Closed file"

! --- original
!    do i = 0, lmax
!       do j = 1, numchains
!          do k = numargs+1, numargs+numsamples
! --- RF
    do k = numargs+1, numargs+numsamples
       do j = 1, numchains
          do i = 0, lmax
! ---
             data(i, j, k) = indata(i, 1, j, k)
             ! data(:,:,:) = indata(0:lmax,1:1,1:numchains,numargs+1:numargs+numsamples)
          enddo
       enddo
    enddo

!!$    print *, "Deallocating data"

    deallocate(indata)

!!$    print *, "Leaving subroutine"

  end subroutine read_gibbs_chain



  ! Utility routine for initializing the offset to be subtracted from each term
  ! to avoid overflow errors. Only called with the first power spectrum
  subroutine compute_largest_term(cls)
    implicit none

    real(dp), dimension(lmin:), intent(in)  :: cls

    integer(i4b) :: i, j, l
    real(dp)     :: subtotal, x

    ! Compute the Blackwell-Rao estimator
    offset = -1.6375e30

! --- original
!    do i = 1, numchain
!       do j = 1, numsamples
! --- RF

    do j = 1, numsamples
       do i = 1, numchain
! ---          
          subtotal = 0.d0
          do l = lmin, lmax
             x = sigmas(l,i,j)/cls(l)
             subtotal = subtotal + &
                  & 0.5d0 * real(2*l+1,dp) * (-x + log(x)) - log(real(sigmas(l,i,j),dp))
          end do
          
          offset = max(offset,subtotal)
          
       end do
    end do

    if (offset < -1.637e30) then
       print *, "Error: offset in br_mod_dist not being computed properly"
       print *, "lmin = ", lmin
       print *, "lmax = ", lmax
       print *, "numchain = ", numchain
       print *, "numsamples = ", numsamples
       print *, "offset = ", offset
       print *, "cls = ", cls(lmin:lmax)
       print *, "sigmas(lmin:lmax, 10, 10) = ", sigmas(lmin:lmax, 10, 10)
       stop
    endif

  end subroutine compute_largest_term


end module br_mod_dist
