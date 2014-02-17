program test_options
	use f90_des_options
	implicit none

	integer(c_size_t) :: options
	character(des_opt_len) :: my_value
	character(*), parameter :: section = "my_section"
	character(*), parameter :: param = "my_option"
	character(*), parameter :: int_param = "my_int"
	character(*), parameter :: double_param = "my_double"
	character(*), parameter :: filename = "f90_test_options.ini"
	integer, parameter :: my_int_default = 2505
	real(8), parameter :: my_double_default = 3.1415
	integer :: my_int
	real(8) :: my_double
	integer :: status

	options = des_optionset_read(filename)

	! Text option
	my_value = des_optionset_get(options, section, param)
	write(*,*) "Found text option = ", trim(my_value)

	!Int option
	status = des_optionset_get_int(options, section, int_param, my_int)
	write(*,*) "Found int, status", my_int, status
	status = des_optionset_get_int_default(options, section, int_param, my_int, my_int_default)
	write(*,*) "Found (default) int, status", my_int, status

	!Double option
	status = des_optionset_get_double(options, section, double_param, my_double)
	write(*,*) "Found double, status", my_int, status
	status = des_optionset_get_double_default(options, section, double_param, my_double, my_double_default)
	write(*,*) "Found (default) double, status", my_double, status

end program

