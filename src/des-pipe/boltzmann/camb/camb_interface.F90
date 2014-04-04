module camb_interface_tools
	use camb
	use cosmosis_modules
	implicit none

	integer :: standard_lmax = 1200
	integer, parameter :: CAMB_MODE_ALL = 1
	integer, parameter :: CAMB_MODE_CMB = 2
	integer, parameter :: CAMB_MODE_BG  = 3
	integer, parameter :: CAMB_MODE_THERMAL  = 4

	real(8) :: linear_zmin=0.0, linear_zmax=4.0
	integer :: linear_nz = 401



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

	function camb_initial_setup(block, mode, fixed_mode) result(status)
		integer default_lmax
		integer(c_size_t) :: block
		integer status
		character(64) :: mode_name=""
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

			status = datablock_get_string(block, default_option_section, "mode", mode_name)
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
				write(*,*) "We found error status: ", status
				write(*,*) "And mode=", mode_name
				write(*,*) "Quitting now."
				stop 1
			endif
		endif

		!We do not use the CMB lmax if only using the background mode
		if (mode .ne. CAMB_MODE_BG) then
			status = status + datablock_get_int_default(block, default_option_section, "cmb_lmax", default_lmax, standard_lmax)
		endif
		!We can always set an optional feedback level,
		!which defaults to zero (silent)
		status = status + datablock_get_int_default(block, default_option_section, "feedback", 0, FeedbackLevel)
		use_tabulated_w = .false.
		status = status + datablock_get_logical_default(block, default_option_section, "use_tabulated_w", .false., use_tabulated_w)

		if (mode == CAMB_MODE_ALL) then
			status = status + datablock_get_double_default(block,default_option_section,"zmin", linear_zmin, linear_zmin)
			status = status + datablock_get_double_default(block,default_option_section,"zmax", linear_zmax, linear_zmax)
			status = status + datablock_get_int_default(block,default_option_section,"nz", linear_nz, linear_nz)
		endif

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

	function camb_interface_set_params(block, params, background_only) result(status)
		integer (c_int) :: status
		integer (c_size_t) :: block
		logical, optional :: background_only
		logical :: perturbations
		type(CambParams) :: params
		real(8) :: omegam
		real(8), dimension(:), allocatable :: w_array, a_array
		character(*), parameter :: cosmo = cosmological_parameters_section
		perturbations = .true.
		if (present(background_only)) perturbations = .not. background_only

	
		call CAMB_SetDefParams(params)
		status = 0
		status = status + datablock_get_double(block, cosmo, "Omega_b", params%omegab)
		status = status + datablock_get_double(block, cosmo, "Omega_m", omegam)
		status = status + datablock_get_double(block, cosmo, "h0", params%h0)
		write(*,*) status
		if (perturbations) then
			status = status + datablock_get_double(block, cosmo, "n_s",     params%initpower%an(1))
			status = status + datablock_get_double(block, cosmo, "A_s",     params%initpower%ScalarPowerAmp(1))
			status = status + datablock_get_double(block, cosmo, "tau", params%Reion%optical_depth)
		endif
		write(*,*) status
		status = status + datablock_get_double_default(block, cosmo, "Omega_K", 0.0D0, params%omegak)
		status = status + datablock_get_double_default(block, cosmo, "cs2_de", 1.0D0, cs2_lam)
		call setcgammappf()

		! tabulated dark energy EoS
		if (use_tabulated_w) then
			status = status + datablock_get_double_array_1d(block, de_equation_of_state_section, "W", w_array, nw_ppf)
			status = status + datablock_get_double_array_1d(block, de_equation_of_state_section, "A", a_array, nw_ppf)
			if (nw_ppf .gt. nwmax) then
				write(*,*) "The size of the w(a) table was too large ", nw_ppf, nwmax
				status=nw_ppf
				return
			endif
			w_ppf(1:nw_ppf) = w_array(1:nw_ppf)
			a_ppf(1:nw_ppf) = dlog(a_array(1:nw_ppf))  !a is stored as log(a)
			call setddwa()
			call interpolrde()
		else
			status = status + datablock_get_double_default(block, cosmo, "W", -1.0D0, w_lam)
			status = status + datablock_get_double_default(block, cosmo, "WA",  0.0D0, wa_ppf)
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
	
	function camb_interface_setup_zrange(params) result(status)
		integer(cosmosis_status) :: status
		type(CambParams) :: params
		real(8) :: zmin, zmax, dz
		integer nz, i

		zmin = linear_zmin
		zmax = linear_zmax
		nz = linear_nz

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
		status = 0
	end function



	function camb_interface_save_cls(block) result(status)
	
		integer (cosmosis_block) :: block
		integer (cosmosis_status) :: status
	
		integer, parameter :: input_set = 1
		real  :: cls(2:standard_lmax,1:4)
		real(8)  :: cls_double(2:standard_lmax,1:4)
		integer  :: ell(2:standard_lmax), l
		logical, parameter :: switch_polarization_convention = .false.	
	
		status = 0
		call CAMB_GetCls(cls, standard_lmax, input_set, switch_polarization_convention)
		cls_double = cls * 7.4311e12  !cmb output scale
	    do l=2,standard_lmax
			ell(l) = l
		enddo
	
		status = status + datablock_put_int_array_1d(block, cmb_cl_section, "ELL", ell)
		status = status + datablock_put_double_array_1d(block, cmb_cl_section, "TT", cls_double(:,1))
		status = status + datablock_put_double_array_1d(block, cmb_cl_section, "EE", cls_double(:,2))
		status = status + datablock_put_double_array_1d(block, cmb_cl_section, "BB", cls_double(:,3))
		status = status + datablock_put_double_array_1d(block, cmb_cl_section, "TE", cls_double(:,4))

	
		if (status .ne. 0) then
			write(*,*) "Failed to save cmb!."
			return
		endif
	end function

	function camb_interface_save_sigma8(block) result(status)
		!Save sigma8 at z=0 to the cosmological parameters section of the file
		integer (cosmosis_block) :: block
		integer (cosmosis_status) :: status
		real(8) :: sigma8
		real(8), parameter :: radius8 = 8.0_8
		integer nz
		
		!Ask camb for sigma8
		status = 0
		sigma8=0.0
		call Transfer_Get_sigma8(MT,radius8)
		
		!It gives us the array sigma8(z).
		!We want the entry for z=0
		nz = CP%Transfer%num_redshifts
		sigma8 = MT%sigma_8(nz,1)

		!Save sigma8
		status = status + datablock_put_double(block, cosmological_parameters_section, "SIGMA_8", sigma8)
		return
	end function
	
	

	

	function camb_interface_save_transfer(block) result(status)
		integer (cosmosis_block) :: block
		integer (cosmosis_status) :: status
		integer nz, nk, nt, iz, ik, idx
		real(8), allocatable, dimension(:) :: k, z, t, P
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
		status = status + datablock_put_double_array_1d(block, linear_cdm_transfer_section, "K_H", k);
		status = status + datablock_put_double_array_1d(block, linear_cdm_transfer_section, "Z", z);
		status = status + datablock_put_double_array_1d(block, linear_cdm_transfer_section, "DELTA_CDM", T);
		status = status + datablock_put_int(block, linear_cdm_transfer_section, "NK", nk);
		status = status + datablock_put_int(block, linear_cdm_transfer_section, "NZ", nz);
		status = status + datablock_put_int(block, linear_cdm_transfer_section, "NT", nt);

		status = status + datablock_put_double_array_1d(block, matter_power_lin_section, "K_H", k);
		status = status + datablock_put_double_array_1d(block, matter_power_lin_section, "Z", z);
		status = status + datablock_put_double_array_1d(block, matter_power_lin_section, "P_K", P);
		status = status + datablock_put_int(block, matter_power_lin_section, "NK", nk);
		status = status + datablock_put_int(block, matter_power_lin_section, "NZ", nz);
		status = status + datablock_put_int(block, matter_power_lin_section, "NT", nt);

		if (status .ne. 0) then
			write(*,*) "Failed to save transfer/matter power sections."
		endif
		
		
		deallocate(k, z, T, P)
		return
		
	end function
	
	function camb_interface_save_da(params, block, save_density, save_thermal) result(status)
		integer (cosmosis_block) :: block
		type(CambParams) :: params
		integer (c_int) :: status
		logical, optional :: save_density, save_thermal
		logical :: density, thermal
		real(8), dimension(:), allocatable :: distance, z, rho
		character(*), parameter :: dist = distances_section
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
		enddo
		

		shift = camb_shift_parameter(params)
		status = status + datablock_put_double(block, dist, "CMBSHIFT", shift)


		if (thermal) then
			status = status + datablock_put_double(block, dist, &
				"AGE", ThermoDerivedParams( derived_Age ))

			status = status + datablock_put_double(block, dist, &
				"RS_ZDRAG", ThermoDerivedParams( derived_rdrag ))

			status = status + datablock_put_double(block, dist, &
				"THETA", ThermoDerivedParams( derived_thetastar ))

			status = status + datablock_put_double(block, dist, &
				"ZDRAG", ThermoDerivedParams( derived_zdrag ))

			status = status + datablock_put_double(block, dist, &
				"ZSTAR", ThermoDerivedParams( derived_zstar ))

			status = status + datablock_put_double(block, dist, &
				"CHISTAR", ComovingRadialDistance(ThermoDerivedParams( derived_zstar )))
		else
			status = status + datablock_put_double(block, dist, &
				"AGE", DeltaPhysicalTimeGyr(0.0_dl,1.0_dl))
		endif


		status = status + datablock_put_double_array_1d(block, dist, "Z", z)
		status = status + datablock_put_double_array_1d(block, dist, "D_A", distance)

		distance = distance * (1+z) !Convert to D_M
		status = status + datablock_put_double_array_1d(block, dist, "D_M", distance)

		distance = distance * (1+z) !Convert to D_L
		status = status + datablock_put_double_array_1d(block, dist, "D_L", distance)

		distance = 5*log10(distance)+25 !Convert to distance modulus
		! The distance is already the dimensionful one, so we do not
		! multiply be c/H0
		status = status + datablock_put_double_array_1d(block, dist, "MU", distance)

		! Save H(z)
		do i=1,nz
			distance(i) = HofZ(z(i))
		enddo
		status = status + datablock_put_double_array_1d(block, dist, "H", distance)

		if (density) status = status + datablock_put_double_array_1d(block, dist, "RHO", rho)


		status = status + datablock_put_int(block, dist, "NZ", nz)

		if (status .ne. 0) then
			write(*,*) "Failed to write redshift-distance column data in block section."
		endif
		
		deallocate(distance)
		deallocate(z)
		if (density) deallocate(rho)
		
	end function
	
end module camb_interface_tools



