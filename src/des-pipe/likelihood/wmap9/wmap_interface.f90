function setup(options) result(output)
	use cosmosis_modules
	implicit none
	integer(c_size_t), value :: options
	integer(c_size_t) :: output

	! The publically released WMAP code
	! does not have any generally useful 
	! options that we could set here from the ini
	! file, though if you wanted to change some
	! of the internals you could do it here by reading
	! from options


	! We do not need to save anything so just return 0.
	! We could have returned anything we wanted here;
	! the execute function will get it back later.
	output = 0

end function


function execute(block, config) result(status)
	use wmap_options
	use wmap_likelihood_9yr
	use wmap_util
	use cosmosis_modules
	implicit none
	real(8), allocatable, dimension(:) :: tt,te,ee,bb
	integer(4), allocatable, dimension(:) :: ell
    real(8)  :: total_like, like(8)
	integer n_ell
	integer(cosmosis_status) :: status
	integer(cosmosis_block), value :: block
	integer(c_size_t) :: config

	integer ell_start
	status=0


	!Load all the columns 
	n_ell=0
	status = status + datablock_get_int_array_1d(block, cmb_cl_section, "ELL", ell, n_ell)
	status = status + datablock_get_double_array_1d(block, cmb_cl_section, "TT",  tt,  n_ell)
	status = status + datablock_get_double_array_1d(block, cmb_cl_section, "EE",  ee,  n_ell)
	status = status + datablock_get_double_array_1d(block, cmb_cl_section, "BB",  bb,  n_ell)
	status = status + datablock_get_double_array_1d(block, cmb_cl_section, "TE",  te,  n_ell)
	!Check for errors in loading the columns.  Free any allocated mem if so
	if (status .ne. 0 .or. n_ell .le. 0) then
		if (allocated(ell)) deallocate(ell)
		if (allocated(tt)) deallocate(tt)
		if (allocated(ee)) deallocate(ee)
		if (allocated(bb)) deallocate(bb)
		if (allocated(te)) deallocate(te)
		status = max(status, 1)
		return
	endif
		
	!find ell=2 - the column may have started at ell=0 or ell=1
	do ell_start=1,n_ell
		if (ell(ell_start)==2) exit
	enddo
	
	
	!If the ells are wrong and cannot find 2 then complain and free the memory
	if (ell_start .ge. n_ell) then
		write(*,*) "Could not figure out where ell=2 was in the WMAP data"
		status=2
		if (allocated(ell)) deallocate(ell)
		if (allocated(tt)) deallocate(tt)
		if (allocated(ee)) deallocate(ee)
		if (allocated(bb)) deallocate(bb)
		if (allocated(te)) deallocate(te)
		return
	endif
		
	
	!Compute likelihood
	!Set wmap max values.  If we have only computed 
	if (n_ell-ell_start+2 .lt. ttmax) ttmax = n_ell-ell_start+2
	if (n_ell-ell_start+2 .lt. temax) temax = n_ell-ell_start+2

	!Setup WMAP error routine to report errors
    call wmap_likelihood_error_init()

	
	!Call the WMAP code, making sure we start from the right place in the data
	like=0.0d0
    call wmap_likelihood_compute(tt(ell_start:),te(ell_start:),ee(ell_start:),bb(ell_start:),like)
	!WMAP code returns -log(like)
	total_like = -sum(like)
	!check for WMAP errors and report them if so
    if (.not. wmap_likelihood_ok) then
		call wmap_likelihood_error_report()
		status = 3
	endif

	!Free memory
	if (allocated(ell)) deallocate(ell)
	if (allocated(tt)) deallocate(tt)
	if (allocated(ee)) deallocate(ee)
	if (allocated(bb)) deallocate(bb)
	if (allocated(te)) deallocate(te)
	
	
	status = datablock_put_double(block, likelihoods_section, "WMAP9_LIKE", total_like)


end function execute
