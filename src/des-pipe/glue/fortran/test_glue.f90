program test
	use f90_desglue
	implicit none
	integer(c_size_t) handle
	integer(c_int) status, status2
	integer(c_size_t) fitsfile
	integer i, read_nrow
	integer(4), dimension(30) :: col1_data
	real(8), dimension(30) :: col2_data
	integer(4), allocatable, dimension(:) :: col1_data_read
	real(8), allocatable, dimension(:) :: col2_data_read
	
	real(8) :: effective_beans, bean_size, bean_age
	status=0
	handle=alloc_internal_fits()
	write(*,*) "Handle = ", handle
	status=make_fits_object(handle)
	write(*,*) "Status after make = ", status
	fitsfile = fitsfile_from_internal(handle)
	write(*,*) "fitsfile = ", fitsfile
	status = fits_goto_or_create_extension(fitsfile, "MAGIC_BEANS")
	write(*,*) "Status after create = ", status
	status = fits_put_double_parameter(fitsfile, "BEANS", 3.14d0, "Effective number of magic beans")
	write(*,*) "Status after put = ", status
	status = fits_get_double_parameter(fitsfile, "BEANS", effective_beans)
	write(*,*) "Status after put = ", status
	write(*,*) "Beans after put = ", effective_beans
	status = fits_get_double_parameter_default(fitsfile, "SIZE", bean_size, 4.0d0)
	write(*,*) "Bean size (hopefully default value) = ", bean_size
! 	status = fits_get_double_parameter(fitsfile, "AGE", bean_age)
! 	write(*,*) "Should have just failed: status = ", status

	!function fits_create_new_table(fitsfile, name, column_names, formats, units) result(status)
	
	status = fits_create_new_table(fitsfile, "MY_EPIC_TABLE", (/ "COL1", "COL2" /), (/ "J", "D" /),  (/ "UNIT1", "UNIT2" /)  )
	write(*,*) ""
	write(*,*) "Status after create table = ", status


	do i=1,30
		col1_data(i) = i
		col2_data(i) = 1.0*i
	enddo

	status = fits_write_column(fitsfile, "COL1", col1_data)
	write(*,*) ""
	write(*,*) "Status after write col1 = ", status

	status = fits_write_column(fitsfile, "COL2", col2_data)
	write(*,*) ""
	write(*,*) "Status after write col2 = ", status
	
	
	
	
	
	status = fits_get_column(fitsfile, "COL1", col1_data_read, read_nrow)
	write(*,*) ""
	write(*,*) "Status after read col1 = ", status
	write(*,*) "Size = ", read_nrow
	write(*,*) col1_data_read

	status = fits_get_column(fitsfile, "COL2", col2_data_read, read_nrow)
	write(*,*) ""
	write(*,*) "Status after read col2 = ", status
	write(*,*) "Size = ", read_nrow
	write(*,*) col2_data_read

	status2 = f90_fits_close_file(fitsfile)
	write(*,*) "Tried to close just now - ", status, status2



	status = fits_dump_to_disc(handle, "my_test.fits")
	write(*,*) "Tried to save ", status
	
	
end program test