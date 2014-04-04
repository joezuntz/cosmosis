! Liklehood from massively simplified WMAP data
! Just looks at the CMB shift parameter R
! Reported by WMAP9 as 
! R = 1.728 +/- 0.016

function execute(handle) result(status)
	use f90_desglue
	use iso_c_binding
	integer(c_size_t), value :: handle
	integer(c_size_t) :: package	
	integer(c_int) :: status
	real(8) :: R, like

	!Fixed parameter values for R_mu and R_sigma
	real(8), parameter ::  R_mu = 1.728
	real(8), parameter ::  R_sigma = 0.016
	
	!Error checking
	status = 0

	! Open the package
	package = fitsfile_from_internal(handle)
	!Check everything opened okay.  Print message if not.
	if (package==0) then
		write(*,*) "Could not open package in wmap_shift. Error", status
		status=1
		return
	endif

	!Extract the shift parameter and calculate the likelihood
	R = 0.0
	status = status + fits_goto_extension(package, distances_section)
	status = status + fits_get_double_parameter(package, "CMBSHIFT", R)
	like = -0.5*(R - R_mu)**2/R_sigma**2

	!Save the likelihood and close the file
	status = status + fits_goto_or_create_extension(package, likelihoods_section)
	status = status + fits_put_double_parameter(package, "R_LIKE", like, "Shift parameter likelihood")
	status = status + close_fits_object(package)

	!status is the return value - any problems will be passed along

end function