
module wmap_util

	implicit none

	logical, public :: wmap_likelihood_ok, wmap_likelihood_good
	logical, parameter :: warnings_are_errors = .false.

	public :: get_free_lun
	public :: wmap_likelihood_error_init
	public :: wmap_likelihood_error
	public :: wmap_likelihood_warning
	public :: wmap_likelihood_error_report
	public :: wmap_timing_start
	public :: wmap_timing_checkpoint
	public :: wmap_timing_end

	private

	integer, parameter :: n_errs_max = 10
	character(len=78) :: err_msg(n_errs_max)
	integer :: n_errs, err_code(n_errs_max)

	integer, parameter :: n_warn_max = 10
	character(len=78) :: warn_msg(n_warn_max)
	integer :: n_warn, warn_code(n_warn_max)

	type Timer
		character(len=64) :: desc
		integer :: start_count, end_count, count_rate, count_max
		integer :: checkpoint_count
		logical :: stub
	end type

	integer :: tlun
	integer :: n_timers = 0
	integer, parameter :: max_n_timers = 6
	type(Timer) :: timers(max_n_timers)

contains

  subroutine get_free_lun( lun )

	implicit none

	integer, intent(out) :: lun

	integer, save :: last_lun = 19
	logical :: used

	lun = last_lun
	do
		inquire( unit=lun, opened=used )
		if ( .not. used ) exit
		lun = lun + 1
	end do

	last_lun = lun

  end subroutine

  subroutine wmap_likelihood_error_init( )

	wmap_likelihood_ok = .true.
	wmap_likelihood_good = .true.
	n_errs = 0
	n_warn = 0

  end subroutine

  subroutine wmap_likelihood_error( msg, code )

	character(len=*), intent(in) :: msg
	integer, intent(in) :: code

	wmap_likelihood_ok = .false.
	wmap_likelihood_good = .false.

	if ( n_errs < n_errs_max ) then
		n_errs = n_errs + 1
		err_msg(n_errs) = msg
		err_code(n_errs) = code
	else
		print *, '*** error log full'
		stop
	end if

  end subroutine

  subroutine wmap_likelihood_warning( msg, code )

	character(len=*), intent(in) :: msg
	integer, intent(in) :: code

	if ( warnings_are_errors) then
		wmap_likelihood_ok = .false.
	end if
	wmap_likelihood_good = .false.

	if ( n_warn < n_warn_max ) then
		n_warn = n_warn + 1
		warn_msg(n_warn) = msg
		warn_code(n_warn) = code
	else
		print *, '*** warning log full'
		stop
	end if

  end subroutine

  subroutine wmap_likelihood_error_report( )

	integer :: i

	print *, '------------------------------------------------------------'
	print *, 'WMAP likelihood evaluation report:'

	if ( wmap_likelihood_ok ) then
		print *, 'no errors'
	else
		print *, 'number of errors = ', n_errs
		do i = 1,n_errs
			print *, ''
			print *, 'error #', i, '::'
			print *, err_msg(i)
			print *, 'error code = ', err_code(i)
		end do
	end if

	if ( n_warn > 0 ) then
		print *, 'number of warnings = ', n_warn
		do i = 1,n_warn
			print *, ''
			print *, 'warning #', i, '::'
			print *, warn_msg(i)
			print *, 'warning code = ', warn_code(i)
		end do
	end if

	print *, '------------------------------------------------------------'

  end subroutine

  subroutine wmap_timing_start( desc )

	character(len=*), intent(in) :: desc

	integer :: k
	real :: elapsed_sec
	character(len=64) :: f

!	if ( n_timers == 0 ) then
!		call get_free_lun( tlun )
!		open(tlun,file=ofn_timing,action='write',status='unknown')
!	end if

	if ( n_timers >= max_n_timers ) then
		print *, '*** too many timing levels'
		stop
	end if

	if ( n_timers > 0 .and. timers(n_timers)%stub ) then
		timers(n_timers)%stub = .false.
		write(f,'(A,I1,A)') '("        .......",', n_timers, '("  "),A,":")'
		write(*,f) trim(timers(n_timers)%desc)
	end if

	n_timers = n_timers + 1
	k = n_timers

	timers(k)%desc = desc
        call system_clock( timers(k)%start_count, timers(k)%count_rate, &
		timers(k)%count_max )
	timers(k)%checkpoint_count = -1
	timers(k)%stub = .true.

  end subroutine

  subroutine wmap_timing_checkpoint( desc )

	character(len=*), intent(in) :: desc

	integer :: start_count, end_count, count_rate, count_max, k
	real :: elapsed_sec
	character(len=64) :: f

	if ( timers(n_timers)%checkpoint_count == -1 ) then
		start_count = timers(n_timers)%start_count
		if ( timers(n_timers)%stub ) then
			timers(n_timers)%stub = .false.
			write(f,'(A,I1,A)') '("        .......",', n_timers, '("  "),A,":")'
			write(*,f) trim(timers(n_timers)%desc)
		end if
	else
		start_count = timers(n_timers)%checkpoint_count
	end if
	call system_clock( end_count, count_rate, count_max )

	elapsed_sec = real(end_count-start_count)/real(count_rate)
	write(f,'(A,I1,A)') '(F15.4,', n_timers, '("  ")," - ",A)'
	write(*,f) elapsed_sec, trim(desc)

	timers(n_timers)%checkpoint_count = end_count

  end subroutine

  subroutine wmap_timing_end( )

	integer :: start_count, end_count, count_rate, count_max
	real :: elapsed_sec
	character(len=64) :: f

	if ( n_timers == 0 ) then
		print *, '*** n_timers == 0 in wmap_timing_end'
		stop
	end if

	call system_clock( end_count, count_rate, count_max )
	elapsed_sec = real(end_count-timers(n_timers)%start_count)/real(count_rate)

	write(f,'(A,I1,A)') '(F15.4,', n_timers, '("  "),A)'
	write(*,f) elapsed_sec, trim(timers(n_timers)%desc)
	n_timers = n_timers - 1

!	if ( n_timers == 0 ) then
!		close(tlun)
!	end if

  end subroutine

end module

