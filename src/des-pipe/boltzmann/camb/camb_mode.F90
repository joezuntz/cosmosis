function setup(options) result(config)
	use f90_desglue
	use iso_c_binding
	use camb_interface_tools
	use f90_des_options

	implicit none

	integer(c_int) :: status
	integer(c_size_t), value :: options
	integer(c_size_t) :: config
	integer :: mode

	status = camb_initial_setup(options, mode)
	if (status .ne. 0) then
		write(*,*) "CAMB setup error.  Quitting."
		write(*,*) trim(global_error_message)
		stop
	endif

	config = mode

end function setup

function execute_all(handle, config) result(status)
	use f90_desglue
	use iso_c_binding
	use camb
	use camb_interface_tools
	
	implicit none
	integer(c_size_t), value :: handle, config
	integer(c_int) :: status, ignored_status
	integer(c_size_t) :: fitsfile
	type(CAMBparams) :: params
	
	status = 0

	fitsfile = fitsfile_from_internal(handle)
	if (fitsfile==0) then
		write(*,*) "Could not get fitsfile in CAMB. Error", status
		status=1
		return
	endif
	
	status = camb_interface_set_params(fitsfile, params)
	status = status + camb_interface_setup_zrange(fitsfile, params)
	
	if (status .ne. 0) then
		write(*,*) "Failed to get parameters in CAMB. Error", status
		ignored_status = close_fits_object(fitsfile)
		status=3
		return
	endif
	
	!Run CAMB to compute the c_ell.
	call CAMB_GetResults(params, status)
	if (status .ne. 0) then
		write(*,*) "Failed to run camb_getresults.  Error", status
		write(*,*) "Message:", trim(global_error_message)
		ignored_status = close_fits_object(fitsfile)
		return
	endif
	
	status = camb_interface_save_cls(fitsfile)
	if (status .ne. 0) then
		write(*,*) "Failed to write cmb column data in fits section. Error", status
		ignored_status = close_fits_object(fitsfile)
		return
	endif

	status = camb_interface_save_transfer(fitsfile)
	if (status .ne. 0) then
		write(*,*) "Failed to write transfer column data in fits section. Error", status
		ignored_status = close_fits_object(fitsfile)
		return
	endif
	
	status = camb_interface_save_sigma8(fitsfile)
	if (status .ne. 0) then
		write(*,*) "Failed to save sigma8 parameter to fits section. Error", status
		ignored_status = close_fits_object(fitsfile)
		return
	endif
	
	
	status = camb_interface_save_da(params, fitsfile)
	if (status .ne. 0) then
		write(*,*) "Failed to write angular diameter distance data in fits section. Error", status
		ignored_status = close_fits_object(fitsfile)
		return
	endif
	
	
	status = close_fits_object(fitsfile)
	!There is probably some clean up to be done here, but not sure what.
	return


end function execute_all



function execute_cmb(handle) result(status)
	use f90_desglue
	use iso_c_binding
	use camb
	use camb_interface_tools
	implicit none
	integer(c_size_t), value :: handle
	integer(c_int) :: status, ignored_status
	integer(c_size_t) :: fitsfile
	type(CAMBparams) :: params
	
	status = 0

	fitsfile = fitsfile_from_internal(handle)
	if (fitsfile==0) then
		write(*,*) "Could not get fitsfile in CAMB"
		status=1
		return
	endif
	
	status = camb_interface_set_params(fitsfile, params)
	if (status .ne. 0) then
		write(*,*) "Failed to get parameters in CAMB"
		ignored_status = close_fits_object(fitsfile)
		status=3
		return
	endif
	
	status = status + camb_interface_setup_zrange(fitsfile, params)
	if (status .ne. 0) then
		write(*,*) "Failed to set redshift ranges in CAMB"
		ignored_status = close_fits_object(fitsfile)
		status=4
		return
	endif


	!Run CAMB to compute the c_ell.  Also get the ells.
	call CAMB_GetResults(params, status)
	if (status .ne. 0) then
		write(*,*) "Failed to run camb_getresults."
		ignored_status = close_fits_object(fitsfile)
		return
	endif


	status = camb_interface_save_cls(fitsfile)
	if (status .ne. 0) then
		write(*,*) "Failed to save camb cls."
		ignored_status = close_fits_object(fitsfile)
		return
	endif
	
	status = camb_interface_save_da(params, fitsfile, .false.)
	if (status .ne. 0) then
		write(*,*) "Failed to save camb distances."
		ignored_status = close_fits_object(fitsfile)
		return
	endif


	status = close_fits_object(fitsfile)
	!There is probably some CAMB clean up to be done here, but not sure what.
	return


end function execute_cmb




function execute_bg(handle, config) result(status)
	use f90_desglue
	use iso_c_binding
	use camb
	use camb_interface_tools
	
	implicit none
	integer(c_size_t), value :: handle, config
	integer(c_int) :: status, ignored_status
	integer(c_size_t) :: fitsfile
	type(CAMBparams) :: params
	
	status = 0

	fitsfile = fitsfile_from_internal(handle)
	if (fitsfile==0) then
		write(*,*) "Could not get fitsfile in CAMB. Error", status
		status=1
		return
	endif
	
	status = camb_interface_set_params(fitsfile, params, background_only=.true.)
	status = status + camb_interface_setup_zrange(fitsfile, params)
	

	if (status .ne. 0) then
		write(*,*) "Failed to get parameters in CAMB. Error", status
		ignored_status = close_fits_object(fitsfile)
		status=3
		return
	endif


	params%WantCls = .false.
	params%WantScalars = .false.
	params%WantTensors = .false.
	params%WantVectors = .false.
	call CAMBParams_Set(params, status, DoReion=.false.)
	
		if (status .ne. 0) then
		write(*,*) "Failed to set camb params in camb_bg. Error", status
		write(*,*) trim(global_error_message)
		ignored_status = close_fits_object(fitsfile)
		return
	endif

	
	status = camb_interface_save_da(params, fitsfile, save_density=.false., save_thermal=.false.)
	if (status .ne. 0) then
		write(*,*) "Failed to write angular diameter distance data in fits section. Error", status
		ignored_status = close_fits_object(fitsfile)
		return
	endif
	
	
	status = close_fits_object(fitsfile)
	!There is probably some clean up to be done here, but not sure what.
	return


end function execute_bg



function execute_thermal(handle, config) result(status)
	use f90_desglue
	use iso_c_binding
	use camb
	use camb_interface_tools
	
	implicit none
	integer(c_size_t), value :: handle, config
	integer(c_int) :: status, ignored_status
	integer(c_size_t) :: fitsfile
	type(CAMBparams) :: params
	
	status = 0

	fitsfile = fitsfile_from_internal(handle)
	if (fitsfile==0) then
		write(*,*) "Could not get fitsfile in CAMB. Error", status
		status=1
		return
	endif
	
	status = camb_interface_set_params(fitsfile, params, background_only=.true.)
	status = status + camb_interface_setup_zrange(fitsfile, params)

	if (status .ne. 0) then
		write(*,*) "Failed to get parameters in CAMB. Error", status
		ignored_status = close_fits_object(fitsfile)
		status=3
		return
	endif


	params%WantCls = .false.
	params%WantScalars = .false.
	params%WantTensors = .false.
	params%WantVectors = .false.
	call CAMBParams_Set(params, status, DoReion=.false.)
	
		if (status .ne. 0) then
		write(*,*) "Failed to set camb params in camb_bg. Error", status
		write(*,*) trim(global_error_message)
		ignored_status = close_fits_object(fitsfile)
		return
	endif

	!Run CAMB thermal stuff to compute the thermal history
	call CAMB_GetResults(params, status)
	if (status .ne. 0) then
		write(*,*) "Failed to run camb_getresults.  Error", status
		write(*,*) "Message:", trim(global_error_message)
		ignored_status = close_fits_object(fitsfile)
		return
	endif
	
	status = camb_interface_save_da(params, fitsfile, save_density=.false.)
	if (status .ne. 0) then
		write(*,*) "Failed to write angular diameter distance data in fits section. Error", status
		ignored_status = close_fits_object(fitsfile)
		return
	endif
	
	
	status = close_fits_object(fitsfile)
	!There is probably some clean up to be done here, but not sure what.
	return


end function execute_thermal

function execute(handle,config) result(status)
	use f90_desglue
	use iso_c_binding
	use camb_interface_tools
	implicit none

	interface
		function execute_bg(handle,config) result(status)
			use iso_c_binding
			integer(c_size_t), value :: handle, config
			integer(c_int) :: status
		end function execute_bg

		function execute_cmb(handle,config) result(status)
			use iso_c_binding
			integer(c_size_t), value :: handle, config
			integer(c_int) :: status
		end function execute_cmb

		function execute_thermal(handle,config) result(status)
			use iso_c_binding
			integer(c_size_t), value :: handle, config
			integer(c_int) :: status
		end function execute_thermal

		function execute_all(handle,config) result(status)
			use iso_c_binding
			integer(c_size_t), value :: handle, config
			integer(c_int) :: status
		end function execute_all
    end interface


	integer(c_size_t), value :: handle, config
	integer(c_int) :: status
	integer :: mode

	mode=config
	if (mode==CAMB_MODE_BG) then
		status = execute_bg(handle,config)
	else if (mode==CAMB_MODE_CMB) then
		status = execute_cmb(handle,config)
	else if (mode==CAMB_MODE_ALL) then
		status = execute_all(handle,config)
	else if (mode==CAMB_MODE_THERMAL) then
		status = execute_thermal(handle,config)
	else
		write(*,*) "Unknown camb mode.  Was expecting one of:"
		write(*,*) CAMB_MODE_BG,  " -- background_only"
		write(*,*) CAMB_MODE_CMB, " -- cmb"
		write(*,*) CAMB_MODE_ALL, " -- all"
		write(*,*) CAMB_MODE_THERMAL, " -- thermal (background w/ thermal history)"
		write(*,*) "This should be set in the pipeline ini file."
		status = 1
	endif

end function execute