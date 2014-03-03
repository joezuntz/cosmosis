module f90_des_options
	USE ISO_C_BINDING
	USE ISO_C_UTILITIES
	implicit none
		integer, parameter :: des_opt_len = 256
		character(*), parameter :: default_option_section = "config"
	interface
		function wrap_des_optionset_get(options, section, param)  bind(C, name='des_optionset_get')
			use iso_c_binding
			implicit none
			integer(c_size_t), value :: options
			character(kind=c_char) :: section(*), param(*)
			type(c_ptr) :: wrap_des_optionset_get
		end function wrap_des_optionset_get
! des_optionset * des_optionset_read(const char * filename);

		function wrap_des_optionset_read(filename)  bind(C, name='des_optionset_read')
			use iso_c_binding
			implicit none
			character(kind=c_char) :: filename(*)
			integer(c_size_t) :: wrap_des_optionset_read
		end function wrap_des_optionset_read
! 
		function wrap_strlen(str) bind(C, name='strlen')
			use iso_c_binding
			implicit none
			type(c_ptr), value :: str
			integer(c_size_t) :: wrap_strlen
		end function wrap_strlen

		function wrap_des_optionset_get_int(options, section, param, value)  bind(C, name='des_optionset_get_int')
			use iso_c_binding
			implicit none
			integer(c_size_t), value :: options
			character(kind=c_char) :: section(*), param(*)
			integer(c_int) :: value
			integer(c_int) :: wrap_des_optionset_get_int
		end function wrap_des_optionset_get_int


		function wrap_des_optionset_get_int_default(options, section, param, value, default)  bind(C, name='des_optionset_get_int_default')
			use iso_c_binding
			implicit none
			integer(c_size_t), value :: options
			character(kind=c_char) :: section(*), param(*)
			integer(c_int) :: value
			integer(c_int), value :: default
			integer(c_int) :: wrap_des_optionset_get_int_default
		end function wrap_des_optionset_get_int_default


		function wrap_des_optionset_get_double(options, section, param, value)  bind(C, name='des_optionset_get_double')
			use iso_c_binding
			implicit none
			integer(c_size_t), value :: options
			character(kind=c_char) :: section(*), param(*)
			real(c_double) :: value
			integer(c_int) :: wrap_des_optionset_get_double
		end function wrap_des_optionset_get_double


		function wrap_des_optionset_get_double_default(options, section, param, value, default)  bind(C, name='des_optionset_get_double_default')
			use iso_c_binding
			implicit none
			integer(c_size_t), value :: options
			character(kind=c_char) :: section(*), param(*)
			real(c_double) :: value
			real(c_double), value :: default
			integer(c_int) :: wrap_des_optionset_get_double_default
		end function wrap_des_optionset_get_double_default


	end interface

	contains

	function c_string_to_fortran(c_str) result(f_str)
		character(des_opt_len) :: f_str
	    character, pointer, dimension(:) :: p_str
		type(c_ptr) :: c_str
		integer(c_size_t) :: n, shpe(1)
		integer i

		!Initialize an empty string
		do i=1,des_opt_len
			f_str(i:i+1) = " "
		enddo

		!Check for NULL pointer.  If so translate as blank
		if(.not. c_associated(c_str)) return

		!Otherwise, get string length and copy that many chars
		n = wrap_strlen(c_str)
		shpe(1) = n
		call c_f_pointer(c_str, p_str, shpe)
		do i=1,n
			f_str(i:i+1) = p_str(i)
		enddo

	end function

	function des_optionset_get(options, section, param) result(option)
		character(des_opt_len) :: option
		integer(c_size_t) :: options
		type(c_ptr) :: c_option
		integer(c_size_t) :: c_len
		character(*) :: section, param
		c_option = wrap_des_optionset_get(options, trim(section)//C_NULL_CHAR, trim(param)//C_NULL_CHAR)
		option = c_string_to_fortran(c_option)

	end function des_optionset_get

	function des_optionset_read(filename) result(options)
		integer(c_size_t) :: options
		character(*) :: filename
		options = wrap_des_optionset_read(trim(filename)//C_NULL_CHAR)
	end function

	function des_optionset_get_int(options, section, param, value) result(status)
		integer(c_size_t) :: options
		character(*) :: section, param
		integer :: value
		integer(c_int) c_value
		integer status

		status = wrap_des_optionset_get_int(options, trim(section)//C_NULL_CHAR, trim(param)//C_NULL_CHAR, c_value)
		value = c_value

	end function des_optionset_get_int

	function des_optionset_get_int_default(options, section, param, value, default) result(status)
		integer(c_size_t) :: options
		character(*) :: section, param
		integer :: value, default
		integer(c_int) c_value, c_default
		integer status

		c_default = default
		status = wrap_des_optionset_get_int_default(options, trim(section)//C_NULL_CHAR, trim(param)//C_NULL_CHAR, c_value, c_default)
		value = c_value

	end function des_optionset_get_int_default


	function des_optionset_get_double(options, section, param, value) result(status)
		integer(c_size_t) :: options
		character(*) :: section, param
		real(8) :: value
		real(c_double) c_value
		integer status

		status = wrap_des_optionset_get_double(options, trim(section)//C_NULL_CHAR, trim(param)//C_NULL_CHAR, c_value)
		value = c_value

	end function des_optionset_get_double

	function des_optionset_get_double_default(options, section, param, value, default) result(status)
		integer(c_size_t) :: options
		character(*) :: section, param
		real(8) :: value, default
		real(c_double) c_value, c_default
		integer status

		c_default = default
		status = wrap_des_optionset_get_double_default(options, trim(section)//C_NULL_CHAR, trim(param)//C_NULL_CHAR, c_value, c_default)
		value = c_value

	end function des_optionset_get_double_default


end module

! const char * des_optionset_get(des_optionset * options, const char * section, const char * param);
! int des_optionset_get_int(des_optionset * options, const char * section, const char * param, int * value);
! int des_optionset_get_int_default(des_optionset * options, const char * section, const char * param, int * value, int default_value);
! int des_optionset_get_double(des_optionset * options, const char * section, const char * param, double * value);
! int des_optionset_get_double_default(des_optionset * options, const char * section, const char * param, double * value, double default_value);
