function setup(options_block) result(config)
	use cosmosis_modules
	use camb_interface_tools

	implicit none

	integer(c_int) :: status
	integer(cosmosis_block), value :: options_block
	integer(cosmosis_status) :: config
	integer :: mode

	status = camb_initial_setup(options_block, mode)
	if (status .ne. 0) then
		write(*,*) "CAMB setup error.  Quitting."
		write(*,*) trim(global_error_message)
		stop
	endif

	config = mode

end function setup

function execute_all(block, config) result(status)
	use cosmosis_modules
	use camb
	use camb_interface_tools
	
	implicit none
	integer(cosmosis_status) :: status
	integer(cosmosis_block), value :: block
	integer(c_size_t), value :: config
	type(CAMBparams) :: params
	
	status = 0

	if (block==0) then
		write(*,*) "Could not get block in CAMB. Error", status
		status=1
		return
	endif
	
	status = camb_interface_set_params(block, params)
	status = status + camb_interface_setup_zrange(params)
	
	if (status .ne. 0) then
		write(*,*) "Failed to get parameters in CAMB. Error", status
		status=3
		return
	endif
	
	!Run CAMB to compute the c_ell.
	call CAMB_GetResults(params, status)
	if (status .ne. 0) then
		write(*,*) "Failed to run camb_getresults.  Error", status
		write(*,*) "Message:", trim(global_error_message)
		return
	endif
	
	status = camb_interface_save_cls(block)
	if (status .ne. 0) then
		write(*,*) "Failed to write cmb cls section. Error", status
		return
	endif

	status = camb_interface_save_transfer(block)
	if (status .ne. 0) then
		write(*,*) "Failed to write transfer section. Error", status
		return
	endif
	
	status = camb_interface_save_sigma8(block)
	if (status .ne. 0) then
		write(*,*) "Failed to save sigma8 parameter. Error", status
		return
	endif
	
	
	status = camb_interface_save_da(params, block)
	if (status .ne. 0) then
		write(*,*) "Failed to write angular diameter distance data. Error", status
		return
	endif
	
	return


end function execute_all



function execute_cmb(block, config) result(status)
	use cosmosis_modules
	use camb
	use camb_interface_tools
	implicit none
	integer(cosmosis_status) :: status
	integer(cosmosis_block), value :: block
	integer(c_size_t), value :: config
	type(CAMBparams) :: params
	
	status = 0
	
	status = camb_interface_set_params(block, params)
	if (status .ne. 0) then
		write(*,*) "Failed to get parameters in CAMB"
		status=3
		return
	endif
	
	status = status + camb_interface_setup_zrange(params)
	if (status .ne. 0) then
		write(*,*) "Failed to set redshift ranges in CAMB"
		status=4
		return
	endif


	!Run CAMB to compute the c_ell.  Also get the ells.
	call CAMB_GetResults(params, status)
	if (status .ne. 0) then
		write(*,*) "Failed to run camb_getresults."
		return
	endif


	status = camb_interface_save_cls(block)
	if (status .ne. 0) then
		write(*,*) "Failed to save camb cls."
		return
	endif
	
	status = camb_interface_save_da(params, block, .false.)
	if (status .ne. 0) then
		write(*,*) "Failed to save camb distances."
		return
	endif


	!There is probably some CAMB clean up to be done here, but not sure what.
	return


end function execute_cmb




function execute_bg(block, config) result(status)
	use cosmosis_modules
	use camb
	use camb_interface_tools
	
	implicit none
	integer(cosmosis_status) :: status
	integer(cosmosis_block), value :: block
	integer(c_size_t), value :: config
	type(CAMBparams) :: params
	
	status = 0

	status = camb_interface_set_params(block, params, background_only=.true.)
	status = status + camb_interface_setup_zrange(params)
	

	if (status .ne. 0) then
		write(*,*) "Failed to get parameters in CAMB. Error", status
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
		return
	endif

	
	status = camb_interface_save_da(params, block, save_density=.false., save_thermal=.false.)
	if (status .ne. 0) then
		write(*,*) "Failed to write angular diameter distance data in fits section. Error", status
		return
	endif
	
		!There is probably some clean up to be done here, but not sure what.
	return


end function execute_bg



function execute_thermal(block, config) result(status)
	use cosmosis_modules
	use camb
	use camb_interface_tools
	
	implicit none
	integer(cosmosis_status) :: status
	integer(cosmosis_block), value :: block
	integer(c_size_t), value :: config
	type(CAMBparams) :: params
	
	status = 0
	
	status = camb_interface_set_params(block, params, background_only=.true.)
	status = status + camb_interface_setup_zrange(params)

	if (status .ne. 0) then
		write(*,*) "Failed to get parameters in CAMB. Error", status
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
		return
	endif

	!Run CAMB thermal stuff to compute the thermal history
	call CAMB_GetResults(params, status)
	if (status .ne. 0) then
		write(*,*) "Failed to run camb_getresults.  Error", status
		write(*,*) "Message:", trim(global_error_message)
		return
	endif
	
	status = camb_interface_save_da(params, block, save_density=.false.)
	if (status .ne. 0) then
		write(*,*) "Failed to write angular diameter distance data in fits section. Error", status
		return
	endif
	
	!There is probably some clean up to be done here, but not sure what.
	return


end function execute_thermal

function execute(block,config) result(status)
	use cosmosis_modules
	use camb_interface_tools
	implicit none

	interface
		function execute_bg(block,config) result(status)
			use cosmosis_modules
			integer(cosmosis_status) :: status
			integer(cosmosis_block), value :: block
			integer(c_size_t), value :: config
		end function execute_bg

		function execute_cmb(block,config) result(status)
			use cosmosis_modules
			integer(cosmosis_status) :: status
			integer(cosmosis_block), value :: block
			integer(c_size_t), value :: config
		end function execute_cmb

		function execute_thermal(block,config) result(status)
			use cosmosis_modules
			integer(cosmosis_status) :: status
			integer(cosmosis_block), value :: block
			integer(c_size_t), value :: config
		end function execute_thermal

		function execute_all(block,config) result(status)
			use cosmosis_modules
			integer(cosmosis_status) :: status
			integer(cosmosis_block), value :: block
			integer(c_size_t), value :: config
		end function execute_all
    end interface


	integer(cosmosis_status) :: status
	integer(cosmosis_block), value :: block
	integer(c_size_t), value :: config
	integer :: mode

	mode=config
	if (mode==CAMB_MODE_BG) then
		status = execute_bg(block,config)
	else if (mode==CAMB_MODE_CMB) then
		status = execute_cmb(block,config)
	else if (mode==CAMB_MODE_ALL) then
		status = execute_all(block,config)
	else if (mode==CAMB_MODE_THERMAL) then
		status = execute_thermal(block,config)
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