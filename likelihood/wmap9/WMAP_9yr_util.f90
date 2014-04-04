
Module wmap_util

    Implicit None

    Logical, Public :: wmap_likelihood_ok, wmap_likelihood_good
    Logical, Parameter :: warnings_are_errors = .False.

    Public :: get_free_lun
    Public :: wmap_likelihood_error_init
    Public :: wmap_likelihood_error
    Public :: wmap_likelihood_warning
    Public :: wmap_likelihood_error_report
    Public :: wmap_timing_start
    Public :: wmap_timing_checkpoint
    Public :: wmap_timing_end

    Private

    Integer, Parameter :: n_errs_max = 10
    Character(Len=78) :: err_msg(n_errs_max)
    Integer :: n_errs, err_code(n_errs_max)

    Integer, Parameter :: n_warn_max = 10
    Character(Len=78) :: warn_msg(n_warn_max)
    Integer :: n_warn, warn_code(n_warn_max)

    Type Timer
        Character(Len=64) :: desc
        Integer :: start_count, end_count, count_rate, count_max
        Integer :: checkpoint_count
        Logical :: stub
    End Type Timer

    Integer :: tlun
    Integer :: n_timers = 0
    Integer, Parameter :: max_n_timers = 6
    Type(Timer) :: timers(max_n_timers)

Contains

    Subroutine get_free_lun(lun)

        Implicit None

        Integer, Intent(Out) :: lun

        Integer, Save :: last_lun = 19
        Logical :: used

        lun = last_lun
        Do
            Inquire(Unit=lun, Opened=used)
            If ( .Not. used) Exit
            lun = lun + 1
        End Do

        last_lun = lun

    End Subroutine get_free_lun

    Subroutine wmap_likelihood_error_init()

        wmap_likelihood_ok = .True.
        wmap_likelihood_good = .True.
        n_errs = 0
        n_warn = 0

    End Subroutine wmap_likelihood_error_init

    Subroutine wmap_likelihood_error(msg, code)

        Character(Len=*), Intent(In) :: msg
        Integer, Intent(In) :: code

        wmap_likelihood_ok = .False.
        wmap_likelihood_good = .False.

        If (n_errs < n_errs_max) Then
            n_errs = n_errs + 1
            err_msg(n_errs) = msg
            err_code(n_errs) = code
        Else
            Print *, '*** error log full'
            Stop
        End If

    End Subroutine wmap_likelihood_error

    Subroutine wmap_likelihood_warning(msg, code)

        Character(Len=*), Intent(In) :: msg
        Integer, Intent(In) :: code

        If (warnings_are_errors) Then
            wmap_likelihood_ok = .False.
        End If
        wmap_likelihood_good = .False.

        If (n_warn < n_warn_max) Then
            n_warn = n_warn + 1
            warn_msg(n_warn) = msg
            warn_code(n_warn) = code
        Else
            Print *, '*** warning log full'
            Stop
        End If

    End Subroutine wmap_likelihood_warning

    Subroutine wmap_likelihood_error_report()

        Integer :: i

        Print *, '------------------------------------------------------------'
        Print *, 'WMAP likelihood evaluation report:'

        If (wmap_likelihood_ok) Then
            Print *, 'no errors'
        Else
            Print *, 'number of errors = ', n_errs
            Do i = 1, n_errs
                Print *, ''
                Print *, 'error #', i, '::'
                Print *, err_msg(i)
                Print *, 'error code = ', err_code(i)
            End Do
        End If

        If (n_warn > 0) Then
            Print *, 'number of warnings = ', n_warn
            Do i = 1, n_warn
                Print *, ''
                Print *, 'warning #', i, '::'
                Print *, warn_msg(i)
                Print *, 'warning code = ', warn_code(i)
            End Do
        End If

        Print *, '------------------------------------------------------------'

    End Subroutine wmap_likelihood_error_report

    Subroutine wmap_timing_start(desc)

        Character(Len=*), Intent(In) :: desc

        Integer :: k
        Real :: elapsed_sec
        Character(Len=64) :: f

        !	if ( n_timers == 0 ) then
        !		call get_free_lun( tlun )
        !		open(tlun,file=ofn_timing,action='write',status='unknown')
        !	end if

        If (n_timers >= max_n_timers) Then
            Print *, '*** too many timing levels'
            Stop
        End If

        If (n_timers > 0 .And. timers(n_timers)%stub) Then
            timers(n_timers)%stub = .False.
            Write(f, '(A,I1,A)') '("        .......",', n_timers, '("  "),A,":")'
            Write(*, f) Trim(timers(n_timers)%desc)
        End If

        n_timers = n_timers + 1
        k = n_timers

        timers(k)%desc = desc
        Call System_clock(timers(k)%start_count, timers(k)%count_rate, &
            timers(k)%count_max)
        timers(k)%checkpoint_count = - 1
        timers(k)%stub = .True.

    End Subroutine wmap_timing_start

    Subroutine wmap_timing_checkpoint(desc)

        Character(Len=*), Intent(In) :: desc

        Integer :: start_count, end_count, count_rate, count_max, k
        Real :: elapsed_sec
        Character(Len=64) :: f

        If (timers(n_timers)%checkpoint_count ==-1) Then
            start_count = timers(n_timers)%start_count
            If (timers(n_timers)%stub) Then
                timers(n_timers)%stub = .False.
                Write(f, '(A,I1,A)') '("        .......",', n_timers, '("  "),A,":")'
                Write(*, f) Trim(timers(n_timers)%desc)
            End If
        Else
            start_count = timers(n_timers)%checkpoint_count
        End If
        Call System_clock(end_count, count_rate, count_max)

        elapsed_sec = Real(end_count-start_count) / Real(count_rate)
        Write(f, '(A,I1,A)') '(F15.4,', n_timers, '("  ")," - ",A)'
        Write(*, f) elapsed_sec, Trim(desc)

        timers(n_timers)%checkpoint_count = end_count

    End Subroutine wmap_timing_checkpoint

    Subroutine wmap_timing_end()

        Integer :: start_count, end_count, count_rate, count_max
        Real :: elapsed_sec
        Character(Len=64) :: f

        If (n_timers == 0) Then
            Print *, '*** n_timers == 0 in wmap_timing_end'
            Stop
        End If

        Call System_clock(end_count, count_rate, count_max)
        elapsed_sec = Real(end_count-timers(n_timers)%start_count) / Real(count_rate)

        Write(f, '(A,I1,A)') '(F15.4,', n_timers, '("  "),A)'
        Write(*, f) elapsed_sec, Trim(timers(n_timers)%desc)
        n_timers = n_timers - 1

        !	if ( n_timers == 0 ) then
        !		close(tlun)
        !	end if

    End Subroutine wmap_timing_end

End Module wmap_util

