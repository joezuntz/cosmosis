

function execute(handle) result(status)
	use wmap_options
	use wmap_likelihood_7yr
	use wmap_util
	use f90_desglue
	implicit none
	real(8), allocatable, dimension(:) :: tt,te,ee,bb
	integer(4), allocatable, dimension(:) :: ell
    real(8)  :: total_like, like(8)
	integer n_ell
	integer(c_size_t), value :: handle
	integer(c_size_t) :: fitsfile
	integer status
	integer ell_start
	status=0

	if (handle .eq. 0) then
		status=1
		write(*,*) "NULL handle passed to wmap7 interface"
		return
	endif
	
	!Open the fits file interface using function from f90_desglue
	fitsfile = fitsfile_from_internal(handle)
	!Check for errors in opening file
	if (fitsfile==0) then
		status=1
		return
	endif
	
	!Go to the CMB section of the file.  "cmb_cl_section" is a constant defined in f90_desglue.f90
	status = fits_goto_extension(fitsfile, cmb_cl_section)
	!Check for error in finding extension
	if (status .ne. 0) then
		return
	endif
	
	!Load all the columns 
	n_ell=0
	status = status + fits_get_column(fitsfile, "ELL", ell, n_ell)
	status = status + fits_get_column(fitsfile, "TT",  tt,  n_ell)
	status = status + fits_get_column(fitsfile, "EE",  ee,  n_ell)
	status = status + fits_get_column(fitsfile, "BB",  bb,  n_ell)
	status = status + fits_get_column(fitsfile, "TE",  te,  n_ell)
	!Check for errors in loading the columns.  Free any allocated mem if so
	if (status .ne. 0 .or. n_ell .le. 0) then
		if (allocated(ell)) deallocate(ell)
		if (allocated(tt)) deallocate(tt)
		if (allocated(ee)) deallocate(ee)
		if (allocated(bb)) deallocate(bb)
		if (allocated(te)) deallocate(te)
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
	
	
	!Save result back to the correct extension (which may not exist yet, this might be the first likelihood function)
	status = fits_goto_or_create_extension(fitsfile, likelihoods_section)
	!Save the total likelihood.  If we wanted to save the likelihood components separately we could do it here.
	status = status + fits_put_double_parameter(fitsfile, "WMAP7_LIKE", total_like, "WMAP 7 total likelihood")
	
	!Close interface
	status = close_fits_object(fitsfile)

end function execute
