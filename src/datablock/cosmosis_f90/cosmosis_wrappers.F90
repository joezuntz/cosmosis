module cosmosis_wrappers
	use iso_c_binding
	use cosmosis_types
	implicit none

	interface
		function make_c_datablock() bind(C, name="make_c_datablock")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer(kind=cosmosis_block) :: make_c_datablock
		end function make_c_datablock

		function c_datablock_has_section_wrapper(s, name) bind(C, name="c_datablock_has_section")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_has_section_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: name
		end function c_datablock_has_section_wrapper

		function c_datablock_put_int_wrapper(s, section, name, value) bind(C, name="c_datablock_put_int")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_put_int_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			integer(kind=c_int), value :: value
		end function c_datablock_put_int_wrapper

		function c_datablock_put_double_wrapper(s, section, name, value) bind(C, name="c_datablock_put_double")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_put_double_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			real(kind=c_double), value :: value
		end function c_datablock_put_double_wrapper

		function c_datablock_put_complex_wrapper(s, section, name, value) bind(C, name="c_datablock_put_complex")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_put_complex_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			complex(kind=c_double_complex), value :: value
		end function c_datablock_put_complex_wrapper

		function c_datablock_put_int_array_1d_wrapper(s, section, name, value, sz) &
		bind(C, name="c_datablock_put_int_array_1d")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_put_int_array_1d_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			integer(kind=c_int), dimension(:) :: value
			integer(kind=c_int) :: sz
		end function c_datablock_put_int_array_1d_wrapper


		!DATABLOCK_STATUS c_datablock_put_int(c_datablock* s, const char* section, const char* name, int val);
		function c_datablock_replace_int_wrapper(s, section, name, value) bind(C, name="c_datablock_replace_int")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_replace_int_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			integer(kind=c_int), value :: value
		end function c_datablock_replace_int_wrapper

		function c_datablock_replace_double_wrapper(s, section, name, value) bind(C, name="c_datablock_replace_double")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_replace_double_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			real(kind=c_double), value :: value
		end function c_datablock_replace_double_wrapper

		function c_datablock_replace_complex_wrapper(s, section, name, value) bind(C, name="c_datablock_replace_complex")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_replace_complex_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			complex(kind=c_double_complex), value :: value
		end function c_datablock_replace_complex_wrapper

		function c_datablock_replace_int_array_1d_wrapper(s, section, name, value, sz) &
		bind(C, name="c_datablock_replace_int_array_1d")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_replace_int_array_1d_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			integer(kind=c_int), dimension(:) :: value
			integer(kind=c_int) :: sz
		end function c_datablock_replace_int_array_1d_wrapper


		!DATABLOCK_STATUS c_datablock_get_int(c_datablock* s, const char* section, const char* name, int *val);
		function c_datablock_get_int_wrapper(s, section, name, value) bind(C, name="c_datablock_get_int")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_get_int_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			integer(kind=c_int) :: value
		end function c_datablock_get_int_wrapper

		function c_datablock_get_double_wrapper(s, section, name, value) bind(C, name="c_datablock_get_double")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_get_double_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			real(kind=c_double) :: value
		end function c_datablock_get_double_wrapper

		function c_datablock_get_complex_wrapper(s, section, name, value) bind(C, name="c_datablock_get_complex")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_get_complex_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			complex(kind=c_double_complex) :: value
		end function c_datablock_get_complex_wrapper

		function c_datablock_get_int_array_1d_preallocated_wrapper(s, section, name, value, size, maxsize) &
		bind(C, name="c_datablock_get_int_array_1d_preallocated")
			use iso_c_binding
			use cosmosis_types
			implicit none
			integer (cosmosis_status) :: c_datablock_get_int_array_1d_preallocated_wrapper
			integer(kind=cosmosis_block), value :: s
			character(kind=c_char), dimension(*) :: section
			character(kind=c_char), dimension(*) :: name
			integer(kind=c_int), dimension(:) :: value
			integer(kind=c_int) :: size
			integer(kind=c_int), value :: maxsize
		end function c_datablock_get_int_array_1d_preallocated_wrapper


	end interface

end module cosmosis_wrappers