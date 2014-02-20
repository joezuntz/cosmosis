module halofit_interface_tools
implicit none
contains
!Function to load matter power structure from a FITS file.

function save_matter_power(fitsfile, nk, k, nz, z, p) result(status)
	use iso_c_binding
	use halofit1
	use f90_desglue
	integer(c_size_t) :: fitsfile
	integer(c_int) :: status
	integer nk, nz, i
	real(dl), dimension(nk) :: k
	real(dl), dimension(nz) :: z
	real(dl), dimension(nk,nz) :: p
	real(dl), allocatable, dimension(:) :: col
	integer nt
	character(*), dimension(3), parameter :: column_names = (/ "K_H", "Z  ","P_K" /)
	character(*), dimension(3), parameter :: column_units = (/ "Mpc","N/A","N/A" /)
	character(*), dimension(3), parameter :: column_fmts  = (/ "D","D","D" /)
	
	nt = nk*nz
	
	
	status = 0
	status = status + fits_create_new_table(fitsfile, matter_power_nl_section, column_names, column_fmts, column_units)
	status = status + fits_put_int_parameter(fitsfile, 'NK', nk, "Number of k values")
	status = status + fits_put_int_parameter(fitsfile, 'NZ', nz, "Number of z values")
	status = status + fits_put_int_parameter(fitsfile, 'NT', nt, "Number of k*z values")
	if (status .ne. 0) return


	allocate(col(nt))

	do i=1,nk
		col( (i-1)*nz+1:i*nz) = k(i)
	enddo
	status = status + fits_write_column(fitsfile, "K_H", col)
	
	do i=1,nk
		col( (i-1)*nz+1:i*nz) = z(:)
	enddo
	status = status + fits_write_column(fitsfile, "Z", col)
	
	do i=1,nk
		col( (i-1)*nz+1:i*nz) = p(i,:)
	enddo
	status = status + fits_write_column(fitsfile, "P_K", col)
	
	deallocate(col)
	
end function 


function load_matter_power(fitsfile, PK) result(status)
	use iso_c_binding
	use f90_desglue
	use halofit1
	integer(c_size_t) :: fitsfile
	integer(c_int) :: status
	type(MatterPowerData) :: PK
	real(dl), allocatable, dimension(:) :: k_col, z_col, p_col
	logical :: k_changes_fastest
	integer nkz
	integer i,j
	
	!Get the data columns from the fits data
	status = 0
	status = fits_goto_extension(fitsfile, matter_power_lin_section)
	status = status + fits_get_int_parameter(fitsfile, "NK", PK%num_k)
	status = status + fits_get_int_parameter(fitsfile, "NZ", PK%num_z)
	status = status + fits_get_column_double(fitsfile,"Z", z_col,nkz)
	status = status + fits_get_column_double(fitsfile,"K_H", k_col,nkz)
	status = status + fits_get_column_double(fitsfile,"P_K", p_col,nkz)
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


function execute(handle) result(status)
	use halofit1
	use halofit_interface_tools
	use iso_c_binding
	use f90_desglue
	implicit none
	integer(c_size_t), value :: handle
	integer(c_int) :: status
	integer(c_size_t) :: fitsfile
	type(MatterPowerdata) :: PK
	type(MatterPowerdata) :: PK_NL
	real(dl), dimension(:,:), allocatable :: nonlin_ratio, p
	integer ik, nk
	real(dl) :: kmin, kmax, zmin, zmax, log_kmin, log_kmax
	real(dl) :: omega_baryon
	real(dl), allocatable, dimension(:) :: k
	integer iz

	status = 0
	!Check for zero handle
	if (handle==0) then
		write(*,*) "Null handle in halofit"
		status=1
		return
	endif
	
	!Open the fitsfile and check it has worked
	fitsfile = fitsfile_from_internal(handle)
	if (fitsfile==0) then
		write(*,*) "Could not open fitsfile"
		status=2
		return
	endif

	!Get some important parameters 
	status = fits_goto_extension(fitsfile, cosmological_parameters_section)
	if (status .ne. 0) then
		write(*,*) "No parameters section present in halofit"
		return
	endif
	
	!Set Halofit internal numbers
	status = status + fits_get_double_parameter(fitsfile, "OMEGA_B", omega_baryon)
	status = status + fits_get_double_parameter(fitsfile, "OMEGA_M", omega_matter)
	omegav = 1 - omega_matter

	!Load suggested output numbers or just use defaults
	status = status + fits_get_double_parameter_default(fitsfile, "NL_KMIN", kmin, 1.0D-04)
	status = status + fits_get_double_parameter_default(fitsfile, "NL_KMAX", kmax, 1.0D+02)
	status = status + fits_get_int_parameter_default(fitsfile, "NL_NK",   nk,   200)
	
    if (status .ne. 0) then
		write(*,*) "Required parameters not found in halofit."
		return
	endif
	

	
	!Run halofit
	status = load_matter_power(fitsfile,PK)
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

	status = save_matter_power(fitsfile, nk, k, PK%num_z, PK%redshifts, p)

	deallocate(k)

	status = close_fits_object(fitsfile)
	deallocate(nonlin_ratio)
	deallocate(p)
end function
