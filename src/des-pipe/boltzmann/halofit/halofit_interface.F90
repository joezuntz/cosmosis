module halofit_interface_tools
use cosmosis_modules
implicit none

	type halofit_settings
		real(8) :: kmin, kmax
		integer :: nk
	end type


contains

function save_matter_power(block, nk, k, nz, z, p) result(status)
	use halofit1
	integer(cosmosis_block) :: block
	integer(cosmosis_status) :: status
	integer nk, nz, i
	real(dl), dimension(nk) :: k
	real(dl), dimension(nz) :: z
	real(dl), dimension(nk,nz) :: p
	real(dl), allocatable, dimension(:) :: col
	integer nt
	
	nt = nk*nz
	
	status = 0
	status = status + datablock_put_int(block, matter_power_nl_section ,'NK', nk)
	status = status + datablock_put_int(block, matter_power_nl_section, 'NZ', nz)
	status = status + datablock_put_int(block, matter_power_nl_section, 'NT', nt)
	if (status .ne. 0) return


	allocate(col(nt))

	do i=1,nk
		col( (i-1)*nz+1:i*nz) = k(i)
	enddo
	status = status + datablock_put_double_array_1d(block, matter_power_nl_section, "K_H", col)
	
	do i=1,nk
		col( (i-1)*nz+1:i*nz) = z(:)
	enddo
	status = status + datablock_put_double_array_1d(block, matter_power_nl_section, "Z", col)
	
	do i=1,nk
		col( (i-1)*nz+1:i*nz) = p(i,:)
	enddo
	status = status + datablock_put_double_array_1d(block, matter_power_nl_section, "P_K", col)
	
	deallocate(col)
	
end function 


function load_matter_power(block, PK) result(status)
	use cosmosis_modules
	use halofit1
	integer(cosmosis_block) :: block
	integer(cosmosis_status) :: status
	type(MatterPowerData) :: PK
	real(dl), allocatable, dimension(:) :: k_col, z_col, p_col
	logical :: k_changes_fastest
	integer nkz
	integer i,j
	
	!Get the data columns from the fits data
	status = 0
	status = status + datablock_get_int(block, matter_power_lin_section, "NK", PK%num_k)
	status = status + datablock_get_int(block, matter_power_lin_section, "NZ", PK%num_z)
	status = status + datablock_get_double_array_1d(block, matter_power_lin_section, "Z", z_col,   nkz)
	status = status + datablock_get_double_array_1d(block, matter_power_lin_section, "K_H", k_col, nkz)
	status = status + datablock_get_double_array_1d(block, matter_power_lin_section, "P_K", p_col, nkz)
	if (PK%num_k * PK%num_z .ne. nkz) status = status + 1
	if (status .ne. 0) then
		write(*,*) "Failed to read data in properly from fits file"
		return
	endif
	
	call allocate_matterpower(PK)
	k_changes_fastest = (z_col(2)==z_col(1))
	PK%matpower = 0.0d0
	!Read from the long-columns we have loaded in into the shorter ones.
	if (k_changes_fastest) then
		!Since k changes fastest we just need the first nk elements of the array.  It should repeat after that.
		!Take logs since that is what this code wants.
		do i=1,PK%num_k
			PK%log_kh(i) = log(k_col(i))
		enddo
		!Every nk'th element should be the next redshift, since it is changing slower so will be in blocks
		do i=1,PK%num_z
			PK%redshifts(i) = z_col( PK%num_k*(i-1)+1 )
		enddo
		
		!Then finally we change the matter power from 1D to 2D
		do j=1,PK%num_z
			do i=1,PK%num_k
				PK%matpower(i,j) = log(p_col(PK%num_k*(j-1)+i))
			enddo
		enddo
		

	else
		!Now z changes fastest so the uniq k values are every num_z elements
		!We want log of k.
		do i=1,PK%num_k
			PK%log_kh(i) = log(k_col(PK%num_z*(i-1)+1))
		enddo
		!And simularly the z elements are in blocks, so we take the one at the start.
		do i=1,PK%num_z
			PK%redshifts(i) = z_col(i)
		enddo

		!Then finally we change the matter power from 1D to 2D
		do j=1,PK%num_z
			do i=1,PK%num_k
				PK%matpower(i,j) = log(p_col(PK%num_z*(i-1)+j))
			enddo
		enddo
		
	endif
	
	!Now set up the splines
	call MatterPowerdata_getsplines(PK)
	
	!Free memory
	deallocate(k_col)
	deallocate(p_col)
	deallocate(z_col)
end function

end module halofit_interface_tools

function setup(options) result(settings)
	use cosmosis_modules
	use halofit_interface_tools
	implicit none
	integer(cosmosis_block) :: options
	integer(cosmosis_status) :: status
	type(halofit_settings), pointer :: settings
	allocate(settings)
	status = 0
	status = status + datablock_get_double_default(options, option_section, "kmin", 1.0D-04, settings%kmin)
	status = status + datablock_get_double_default(options, option_section, "kmax", 1.0D+02, settings%kmax)
	status = status + datablock_get_int_default(options, option_section, "nk", 200, settings%nk)

end function setup

function execute(block, settings) result(status)
	use halofit1
	use halofit_interface_tools
	use cosmosis_modules
	implicit none
	integer(cosmosis_block), value :: block
	integer(cosmosis_status) :: status
	type(halofit_settings) :: settings	
	type(MatterPowerdata) :: PK
	type(MatterPowerdata) :: PK_NL
	real(dl), dimension(:,:), allocatable :: nonlin_ratio, p
	integer ik, nk
	real(dl) :: kmin, kmax, zmin, zmax, log_kmin, log_kmax
	real(dl) :: omega_baryon
	real(dl), allocatable, dimension(:) :: k
	integer iz

	status = 0
	
	
	!Set Halofit internal numbers
	status = status + datablock_get_double(block, cosmological_parameters_section, "OMEGA_B", omega_baryon)
	status = status + datablock_get_double(block, cosmological_parameters_section, "OMEGA_M", omega_matter)
	omegav = 1 - omega_matter

    if (status .ne. 0) then
		write(*,*) "Required parameters not found in halofit."
		return
	endif

	!Load suggested output numbers or just use defaults
	kmin = settings%kmin
	kmin = settings%kmax
	nk = settings%nk
	
	
	!Run halofit
	status = load_matter_power(block,PK)
	if (status .ne. 0) then
		write(*,*) "Could not load matter power"
		status=3
		return
	endif
	
	allocate(nonlin_ratio(PK%num_k,PK%num_z))
!	PK%redshifts = PK%redshifts + 0.01
	call NonLinear_GetNonLinRatios(PK,nonlin_ratio)
	
	PK_NL%num_k = PK%num_k
	PK_NL%num_z = PK%num_z
	call allocate_matterpower(PK_NL)
	PK_NL%log_kh = PK%log_kh
	PK_NL%redshifts = PK%redshifts
	PK_NL%matpower = PK%matpower + 2*log(nonlin_ratio)  !Since logarithmic

	!log spaced results
	log_kmin = log(kmin)
	log_kmax = log(kmax)
	call MatterPowerdata_getsplines(PK_NL)
	allocate(k(nk))
	do ik=1,nk
		k(ik) =  exp( log_kmin + (log_kmax-log_kmin)/(nk-1)*(ik-1)  )
	enddo
	allocate(p(nk,PK%num_z))
	do iz=1,PK%num_z
		do ik=1,nk
				p(ik,iz) = MatterPowerData_k(PK_NL,  k(ik), iz) 
				!This uses "dodgy linear interpolation" at high k.
		enddo
	enddo

	status = save_matter_power(block, nk, k, PK%num_z, PK%redshifts, p)

	deallocate(k)
	deallocate(nonlin_ratio)
	deallocate(p)
end function
