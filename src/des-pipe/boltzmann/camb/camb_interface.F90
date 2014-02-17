module camb_interface_tools
	use camb
	use f90_desglue
	use iso_c_binding
	use f90_des_options
	implicit none

	integer :: standard_lmax = 1200
	integer, parameter :: CAMB_MODE_ALL = 1
	integer, parameter :: CAMB_MODE_CMB = 2
	integer, parameter :: CAMB_MODE_BG  = 3
	integer, parameter :: CAMB_MODE_THERMAL  = 4



	contains

	function camb_comoving_sound_horizon() result(rsdrag)
		use ModelParams
		use Precision
		use ThermoData, only : z_drag
		implicit none
		real(dl) ::  adrag, atol, rsdrag
		real(dl), external :: rombint
		integer error

		adrag = 1.0d0/(1.0d0+z_drag)
		atol = 1e-6
		rsdrag = rombint(dsound_da,1d-8,adrag,atol)
	end function camb_comoving_sound_horizon


	function camb_shift_parameter(params) result(shift_parameter)
		type(cambparams) :: params
		real(dl) :: omega_m, ombh2, omdmh2, zstar, shift_parameter
		real(dl), parameter :: c_km_per_s = 299792.458

		omega_m = params%omegac + params%omegab

         ombh2 = CP%omegab*(CP%h0/100.0d0)**2
         omdmh2 = (CP%omegac+CP%omegan)*(CP%h0/100.0d0)**2

    !From Hu & Sugiyama (via modules.f90)
		zstar =  1048*(1+0.00124*ombh2**(-0.738))*(1+ &
			(0.0783*ombh2**(-0.238)/(1+39.5*ombh2**0.763)) * &
			(omdmh2+ombh2)**(0.560/(1+21.1*ombh2**1.81)))

		shift_parameter = sqrt(omega_m) * params%H0 / c_km_per_s * &
		&   (1+zstar)*AngularDiameterDistance(zstar)

	end function

	function camb_initial_setup(options, mode, fixed_mode) result(status)
		integer default_lmax
		integer(c_size_t) :: options
		integer status
		character(des_opt_len) :: mode_name
		integer :: mode
		integer, optional :: fixed_mode
		integer::  use_tabulated_w_int
		default_lmax = standard_lmax
		status=0
		! There are currently three camb modes - "background", "cmb", and "all"
		! This code may get called with a fixed mode, or with not in which case
		! we read from file
		! First in the fixed mode case, we just use that as the output mode
		if (present(fixed_mode)) then
			mode=fixed_mode
		else
			!Otherwise read from ini file
			mode_name = des_optionset_get(options, default_option_section, "mode")
			if (trim(mode_name) == "background") then
				mode=CAMB_MODE_BG
			else if (trim(mode_name) == "cmb") then
				mode=CAMB_MODE_CMB
			else if (trim(mode_name) == "all") then
				mode=CAMB_MODE_ALL				
			else if (trim(mode_name) == "thermal") then
				mode=CAMB_MODE_THERMAL				
			else
				write(*,*) "You need to specify a mode to use the camb module you chose."
				write(*,*) "In the camb section of your ini file, please specify one of:"
				write(*,*) "mode=background  ; For background quantities like D_A(z) only"
				write(*,*) "mode=cmb         ; For background + cmb power spectra"
				write(*,*) "mode=all         ; For background + cmb + linear matter power spectra"
				write(*,*) "mode=thermal     ; For background + thermal history params"
				write(*,*) ""
				write(*,*) "Quitting now."
				stop 1
			endif
		endif

		!We do not use the CMB lmax if only using the background mode
		if (mode .ne. CAMB_MODE_BG) then
			status = status + des_optionset_get_int_default(options, default_option_section, "cmb_lmax", standard_lmax, default_lmax)
		endif
		!We can always set an optional feedback level,
		!which defaults to zero (silent)
		status = status + des_optionset_get_int_default(options, default_option_section, "feedback", FeedbackLevel, 0)
		
		use_tabulated_w = .false.
		use_tabulated_w_int = 0
		status = status + des_optionset_get_int_default(options, default_option_section, "use_tabulated_w", use_tabulated_w_int, 0)
		if (use_tabulated_w_int .ne. 0) use_tabulated_w = .true.
		!Error check
		if (status .ne. 0) then
			write(*,*) "Problem setting some options for camb. Status code =  ", status
		endif

		!If noisy, report relevant params
		if (FeedbackLevel .gt. 0) then
			write(*,*) "camb mode  = ", mode
			if (mode .ne. CAMB_MODE_BG) write(*,*) "camb cmb_lmax = ", standard_lmax
			write(*,*) "camb FeedbackLevel = ", FeedbackLevel
		endif
	end function camb_initial_setup

	function camb_interface_set_params(fitsfile, params, background_only) result(status)
		integer (c_int) :: status
		integer (c_size_t) :: fitsfile
		logical, optional :: background_only
		logical :: perturbations
		type(CambParams) :: params
		real(8) :: omegam
		perturbations = .true.
		if (present(background_only)) perturbations = .not. background_only

		status = fits_goto_extension(fitsfile, cosmological_parameters_section)
	    if (status .ne. 0) then
			write(*,*) "No parameters section present in CAMB"
			status=2
			return
		endif
	
		call CAMB_SetDefParams(params)
		status = 0
		status = status + fits_get_double_parameter(fitsfile, "OMEGA_B", params%omegab)
		status = status + fits_get_double_parameter(fitsfile, "OMEGA_M", omegam)
		status = status + fits_get_double_parameter(fitsfile, "H0",      params%H0)
		if (perturbations) status = status + fits_get_double_parameter(fitsfile, "N_S",     params%initpower%an(1))
		if (perturbations) status = status + fits_get_double_parameter(fitsfile, "A_S",     params%initpower%ScalarPowerAmp(1))
		if (perturbations) status = status + fits_get_double_parameter(fitsfile, "OPT_TAU", params%Reion%optical_depth)
		status = status + fits_get_double_parameter_default(fitsfile, "OMEGA_K", params%omegak, 0.0D0)
		status = status + fits_get_double_parameter_default(fitsfile, "CS2_DE", cs2_lam, 1.0D0)
		call setcgammappf()


		! tabulated dark energy EoS
		if (use_tabulated_w) then
			status = fits_goto_extension(fitsfile, de_equation_of_state_section)
			status = status + fits_count_column_rows(fitsfile, "W", nw_ppf)
			if (nw_ppf .gt. nwmax) then
				write(*,*) "The size of the w(a) table was too large ", nw_ppf, nwmax
				status=nw_ppf
			endif
			w_ppf = -1.0
			a_ppf = 1.0
			status = status + fits_get_column_double_preallocated(fitsfile, "W", w_ppf, nw_ppf, nw_ppf)
			status = status + fits_get_column_double_preallocated(fitsfile, "A", a_ppf, nw_ppf, nw_ppf)
			a_ppf=dlog(a_ppf)  !a is stored as log(a)
			call setddwa()
			call interpolrde()
		else
			status = status + fits_get_double_parameter_default(fitsfile, "W", w_lam, -1.0D0)
			status = status + fits_get_double_parameter_default(fitsfile, "WA", wa_ppf, 0.0D0)
			if (w_lam+wa_ppf .gt. 0) then
				write(*,*) "Unphysical w_0 + w_a = ", w_lam, " + ", wa_ppf, " = ", w_lam+wa_ppf, " > 0"
				status = 1
			endif


		endif	

		params%wantTransfer = .true.
		params%transfer%kmax = 50.0

        params%Max_l=standard_lmax
        params%Max_eta_k=2*standard_lmax

	
	
		!Some extras and modifications - assume flatness.
		params%omegac = omegam-params%omegab
		params%omegav = 1-omegam-params%omegak
		params%H0 = params%H0*100
		params%want_zdrag = .true.
		params%want_zstar = .true.
		params%reion%use_optical_depth = .true.
		use_spline_template=.false.
		params%AccurateReionization = .true.
        params%Transfer%PK_num_redshifts = 1
        params%Transfer%PK_redshifts = 0



	end function
	
	function camb_interface_setup_zrange(fitsfile, params) result(status)
		integer(c_int) :: status
		integer(c_size_t) :: fitsfile
		type(CambParams) :: params
		real(8) :: zmin, zmax, dz
		integer nz, i
		status = 0 
		status = fits_goto_extension(fitsfile, cosmological_parameters_section)
		status = status + fits_get_double_parameter_default(fitsfile, "LIN_ZMIN", zmin, 0.0D0)
		status = status + fits_get_double_parameter_default(fitsfile, "LIN_ZMAX", zmax, 4.0D0)
		status = status + fits_get_int_parameter_default(fitsfile, "LIN_NZ",   nz,   401)
		
! 		nz=ceiling((zmax-zmin)/dz) + 1
		dz=(zmax-zmin)/(nz-1.0)
		params%transfer%num_redshifts =  nz
        params%Transfer%PK_num_redshifts = nz

		if (nz .gt. max_transfer_redshifts) then
			write(*,*) "Requested too many redshifts for CAMB to handle: ", nz, " = (", zmax, " - ", zmin, ") / ", dz, " + 1"
			status = 1
		endif
		
        do i=1,params%transfer%num_redshifts
			params%transfer%redshifts(nz-i+1)  = zmin + dz*(i-1)
	        params%transfer%pk_redshifts(nz-i+1)  = zmin + dz*(i-1)
    	enddo

    	call Transfer_SortAndIndexRedshifts(params%transfer)
		return
	end function



	function camb_interface_save_cls(fitsfile) result(status)
	
		integer (c_size_t) :: fitsfile
		integer (c_int) :: status
	
		integer, parameter :: input_set = 1
		real  :: cls(2:standard_lmax,1:4)
		real(8)  :: cls_double(2:standard_lmax,1:4)
		integer  :: ell(2:standard_lmax), l
		logical, parameter :: switch_polarization_convention = .false.	
		character(*), dimension(5), parameter :: column_names = (/ "ELL", "TT ","EE ","BB ","TE " /)
		character(*), dimension(5), parameter :: column_units = (/ "N/A","uK2","uK2","uK2","uK2" /)
		character(*), dimension(5), parameter :: column_fmts  = (/ "J","D","D","D","D" /)
	
		status = 0
		call CAMB_GetCls(cls, standard_lmax, input_set, switch_polarization_convention)
		cls_double = cls * 7.4311e12
	    do l=2,standard_lmax
			ell(l) = l
		enddo
	
		!Save the data.
		status = status + fits_create_new_table(fitsfile, cmb_cl_section, column_names, column_fmts, column_units)
		if (status .ne. 0) then
			write(*,*) "Failed to create fits section in camb interface."
			return
		endif

! 		write(*,*) "Max TT = ", maxval(cls_double(:,1))
		status = fits_write_column(fitsfile, "ELL", ell)
		status = status + fits_write_column(fitsfile, "TT", cls_double(:,1))
		status = status + fits_write_column(fitsfile, "EE", cls_double(:,2))
		status = status + fits_write_column(fitsfile, "BB", cls_double(:,3))
		status = status + fits_write_column(fitsfile, "TE", cls_double(:,4))
	
		if (status .ne. 0) then
			write(*,*) "Failed to write column data in fits section."
			return
		endif
	end function

	function camb_interface_save_sigma8(fitsfile) result(status)
		!Save sigma8 at z=0 to the cosmological parameters section of the file
		integer (c_size_t) :: fitsfile
		integer (c_int) :: status
		real(8) :: sigma8
		real(8), parameter :: radius8 = 8.0_8
		integer nz
		
		!Tell CAMB to compute sigma8. 
		status = 0
		sigma8=0.0
		call Transfer_Get_sigma8(MT,radius8)
		
		!Get sigma8 at z=0 from camb, which computes it for all redshifts
		!so we take the last one out.
		nz = CP%Transfer%num_redshifts
		sigma8 = MT%sigma_8(nz,1)

		!Go to the right section of the file to save it in.
		status = fits_goto_extension(fitsfile, cosmological_parameters_section)
	    if (status .ne. 0) then
			write(*,*) "No parameters section present in CAMB"
			status=2
			return
		endif
		!Save the parameter
		status = status + fits_put_double_parameter(fitsfile, "SIGMA_8", sigma8, "Sigma 8 value from CAMB")
		if (status .ne. 0) then
			write(*,*) "Unable to save parameter sigma_8 from camb"
			status = 3
			return
		endif
		!And we are done!
		return
	end function
	
	

	

	function camb_interface_save_transfer(fitsfile) result(status)
		integer (c_size_t) :: fitsfile
		integer (c_int) :: status
		integer nz, nk, nt, iz, ik, idx
		real(8), allocatable, dimension(:) :: k, z, t, P
		character(*), dimension(3), parameter :: column_names = (/ "K_H      ", "Z        ","DELTA_CDM" /)
		character(*), dimension(3), parameter :: column_units = (/ "Mpc","N/A","N/A" /)
		character(*), dimension(3), parameter :: column_fmts  = (/ "D","D","D" /)
    
		character(*), dimension(3), parameter :: pk_column_names = (/ "K_H", "Z  ","P_K" /)
		character(*), dimension(3), parameter :: pk_column_units = (/ "Mpc","N/A","N/A" /)
		character(*), dimension(3), parameter :: pk_column_fmts  = (/ "D","D","D" /)
		Type(MatterPowerData) :: PK
	    
		

		status = 0

		call Transfer_GetMatterPowerData(MT, PK, 1)
		!Extract the transfer data from CAMB internal structures
		nz = CP%Transfer%num_redshifts
		nk = MT%num_q_trans
		nt = nk*nz
		allocate(k(nt))
		allocate(z(nt))
		allocate(T(nt))
		allocate(P(nt))
		T=0.0
		do ik=1,nk
			do iz=1,nz
				idx = ((ik-1)*nz+iz)
				k(idx) = MT%TransferData(Transfer_kh,ik,nz-iz+1)
				z(idx) = CP%Transfer%Redshifts(nz-iz+1)
				T(idx) = MT%TransferData(Transfer_cdm,ik,nz-iz+1)
				P(idx) = MatterPowerData_k(PK,  k(idx), nz-iz+1)
			enddo
		enddo
	
		call MatterPowerdata_Free(PK)
	
		!Save the data.
		status = status + fits_create_new_table(fitsfile, linear_cdm_transfer_section, column_names, column_fmts, column_units)
		if (status .ne. 0) then
			write(*,*) "Failed to create fits section in camb interface."
			deallocate(k, z, T, P)
			return
		endif

		status = status + fits_write_column(fitsfile, "K_H", k)
		status = status + fits_write_column(fitsfile, "Z", z)
		status = status + fits_write_column(fitsfile, "DELTA_CDM", T)
		status = status + fits_put_int_parameter(fitsfile, "NZ", nz, "Number of redshift values for transfers")
		status = status + fits_put_int_parameter(fitsfile, "NK", nk, "Number of k/h-values for transfers")
		status = status + fits_put_int_parameter(fitsfile, "NT", nt, "NZ*NK - total number of transfer samples")
	
		if (status .ne. 0) then
			write(*,*) "Failed to write column data in fits section."
			deallocate(k, z, T, P)
			return
		endif
		
		status = status + fits_create_new_table(fitsfile, matter_power_lin_section,&
		 pk_column_names, pk_column_fmts, pk_column_units)
		
		if (status .ne. 0) then
			write(*,*) "Failed to create P(K) section in camb interface."
			deallocate(k, z, T, P)
			return
		endif
		
		status = status + fits_write_column(fitsfile, "K_H", k)
		status = status + fits_write_column(fitsfile, "Z", z)
		status = status + fits_write_column(fitsfile, "P_K", P)
		status = status + fits_put_int_parameter(fitsfile, "NZ", nz, "Number of redshift values for transfers")
		status = status + fits_put_int_parameter(fitsfile, "NK", nk, "Number of k/h-values for transfers")
		status = status + fits_put_int_parameter(fitsfile, "NT", nt, "NZ*NK - total number of transfer samples")
		
		
		if (status .ne. 0) then
			write(*,*) "Failed to create fits section in camb interface."
		endif
		
		
		deallocate(k, z, T, P)
		return
		
	end function
	
	function camb_interface_save_da(params, fitsfile, save_density, save_thermal) result(status)
		integer (c_size_t) :: fitsfile
		type(CambParams) :: params
		integer (c_int) :: status
		logical, optional :: save_density, save_thermal
		logical :: density, thermal
		real(8), dimension(:), allocatable :: distance, z, rho
		character(*), dimension(7), parameter :: column_names_rho = (/ "Z  ", "D_A", "D_L", "D_M", "RHO", "MU ", "H  " /)
		character(*), dimension(7), parameter :: column_units_rho = (/ "N/A     ", "MPC     ", "MPC     ", "MPC     ", "KG/M3   ", "N/A     ", "KM/S/MPC" /)
		character(*), dimension(7), parameter :: column_fmts_rho  = (/ "D"  , "D", "D", "D", "D", "D", "D" /)

		character(*), dimension(6), parameter :: column_names = (/ "Z  ", "D_A", "D_L", "D_M", "MU ", "H  " /)
		character(*), dimension(6), parameter :: column_units = (/ "N/A     ", "MPC     ", "MPC     ", "MPC     ",  "N/A     ", "KM/S/MPC" /)
		character(*), dimension(6), parameter :: column_fmts  = (/ "D"  , "D", "D", "D", "D", "D" /)

		integer nz, i
		


		! Rho as given by the code is actually 8 * pi * G * rho / c**2 , and it is measured in (Mpc)**-2
		! There really isn't a sensible set of units to do this in, so let's just use kg/m**3
		! c**2 / (8 pi G) = 5.35895884e24 kg/m
		! 1 Mpc = 3.08568025e24 m
		real(8), parameter :: mpc_in_m = 3.08568025e22
		real(8), parameter :: c2_8piG_kgm = 5.35895884e25
		real(8), parameter :: rho_units = c2_8piG_kgm / (mpc_in_m**2)
		real(8) :: shift, rs_zdrag

		density = .true.
		if (present(save_density)) density = save_density
		thermal = .true.
		if (present(save_thermal)) thermal = save_thermal

		status = 0


		nz = params%transfer%num_redshifts
		allocate(distance(nz))
		allocate(z(nz))


		if (density) allocate(rho(nz))

		

		do i=1,nz
			z(i) = params%transfer%redshifts(i)
			distance(i) = AngularDiameterDistance(z(i))
			if (density) rho(i) = MT%TransferData(Transfer_rho_tot,1,i) * rho_units
			
			!JAZ Need to check the differential one works first.
			!			distance(i) = DeltaAngularDiameterDistance(z(i), z(i-1))+DeltaAngularDiameterDistance(i-1)
		enddo
		
		if (density) then
			status = status + fits_create_new_table(fitsfile, distances_section, column_names_rho, column_fmts_rho, column_units_rho)
		else
			status = status + fits_create_new_table(fitsfile, distances_section, column_names, column_fmts, column_units)
		endif

		if (status .ne. 0) then
			write(*,*) "Failed to create fits section in camb interface."
			deallocate(distance)
			deallocate(z)
			if (density) deallocate(rho)
			return
		endif

		shift = camb_shift_parameter(params)
		status = status + fits_put_double_parameter(fitsfile, "CMBSHIFT", shift, "CMB shift parameter R")


		if (thermal) then
			status = status + fits_put_double_parameter(fitsfile, &
				"AGE", ThermoDerivedParams( derived_Age ), &
				"Age of universe")

			status = status + fits_put_double_parameter(fitsfile, &
				"RS_ZDRAG", ThermoDerivedParams( derived_rdrag ), &
				"Comov sound horzn at z_drag BAO")

			status = status + fits_put_double_parameter(fitsfile, &
				"THETA", ThermoDerivedParams( derived_thetastar ), &
				"Angle of first peak")

			status = status + fits_put_double_parameter(fitsfile, &
				"ZDRAG", ThermoDerivedParams( derived_zdrag ), &
				"Drag redshift")

			status = status + fits_put_double_parameter(fitsfile, &
				"ZSTAR", ThermoDerivedParams( derived_zstar ), &
				"Redshift of reionization")

			status = status + fits_put_double_parameter(fitsfile, &
				"CHISTAR", ComovingRadialDistance(ThermoDerivedParams( derived_zstar )), &
				"Comoving distance to CMB")
		else
			status = status + fits_put_double_parameter(fitsfile, &
				"AGE", DeltaPhysicalTimeGyr(0.0_dl,1.0_dl), &
				"Age of universe")
		endif


		status = status + fits_write_column(fitsfile, "Z", z)
		status = status + fits_write_column(fitsfile, "D_A", distance)

		distance = distance * (1+z) !Convert to D_M
		status = status + fits_write_column(fitsfile, "D_M", distance)

		distance = distance * (1+z) !Convert to D_L
		status = status + fits_write_column(fitsfile, "D_L", distance)

		distance = 5*log10(distance)+25 !Convert to distance modulus
		! The distance is already the dimensionful one, so we do not
		! multiply be c/H0
		status = status + fits_write_column(fitsfile, "MU ", distance)

		! Save H(z)
		do i=1,nz
			distance(i) = HofZ(z(i))
		enddo
		status = status + fits_write_column(fitsfile, "H", distance)



		if (density)  status = status + fits_write_column(fitsfile, "RHO", rho)

		status = status + fits_put_int_parameter(fitsfile, "NZ", nz, "Number of redshift values for distance measurement")
		if (status .ne. 0) then
			write(*,*) "Failed to write redshift-distance column data in fits section."
		endif
		
		deallocate(distance)
		deallocate(z)
		if (density) deallocate(rho)
		
	end function
	
end module camb_interface_tools



