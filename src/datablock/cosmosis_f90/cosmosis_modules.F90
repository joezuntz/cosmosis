!Main module for module writers to actually use.
!Could just rename this cosmosis.F90 and the module cosmosis?
!That might be easiest for users
module cosmosis_modules
	use cosmosis_types
	use cosmosis_wrappers
	implicit none

	contains


	!Check whether the block contains a given section
	function datablock_has_section(block, section) result(found)
		logical :: found
		integer(cosmosis_block) :: block
		character(*) :: section
		integer(cosmosis_status) :: found_status

		found_status = c_datablock_has_section_wrapper(block, trim(section)//C_NULL_CHAR)
		found = (found_status .ne. 0)
	end function datablock_has_section

	function datablock_get_array_length(block, section, name) result(n)
		integer(c_int) :: n
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name

		n = c_datablock_get_array_length_wrapper(block, trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR)
	end function datablock_get_array_length		

	!Save an integer with the given name to the given section
	function datablock_put_int(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		integer(c_int) :: value

		status = c_datablock_put_int_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_put_int

	!Save a double with the given name to the given section
	function datablock_put_double(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		real(c_double) :: value

		status = c_datablock_put_double_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_put_double



	!Save a complex double with the given name to the given section
	function datablock_put_complex(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		complex(c_double_complex) :: value

		status = c_datablock_put_complex_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_put_complex

!		function c_datablock_put_int_array_1d_wrapper(s, section, name, value, size)

	!Save an integer array with the given name to the given section
	function datablock_put_int_array_1d(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		integer(c_int), dimension(:) :: value
		integer(c_int) :: sz

		sz=size(value)

		status = c_datablock_put_int_array_1d_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, sz)

	end function datablock_put_int_array_1d


	function datablock_put_double_array_1d(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		real(c_double), dimension(:) :: value
		integer(c_int) :: sz

		sz=size(value)

		status = c_datablock_put_double_array_1d_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, sz)

	end function datablock_put_double_array_1d





	!Replace the named integer in the given section with the new value
	function datablock_replace_int(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		integer(c_int) :: value

		status = c_datablock_replace_int_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_replace_int

	!Replace the named double in the given section with the new value
	function datablock_replace_double(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		real(c_double) :: value

		status = c_datablock_replace_double_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_replace_double

	!Replace the named double complex in the given section with the new value
	function datablock_replace_complex(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		complex(c_double_complex) :: value

		status = c_datablock_replace_complex_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_replace_complex

	function datablock_replace_int_array_1d(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		integer(c_int), dimension(:) :: value
		integer(c_int) :: sz

		sz=size(value)

		status = c_datablock_replace_int_array_1d_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, sz)

	end function datablock_replace_int_array_1d


	function datablock_replace_double_array_1d(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		real(c_double), dimension(:) :: value
		integer(c_int) :: sz

		sz=size(value)

		status = c_datablock_replace_double_array_1d_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, sz)

	end function datablock_replace_double_array_1d


	!Load the named integer from the given section
	function datablock_get_int(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(*) :: section
		character(*) :: name
		integer :: value

		status = c_datablock_get_int_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_get_int

	!Load the named double from the given section
	function datablock_get_double(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(*) :: section
		character(*) :: name
		real(c_double) :: value

		status = c_datablock_get_double_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_get_double

	!Load the named complex double from the given section
	function datablock_get_complex(block, section, name, value) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(*) :: section
		character(*) :: name
		complex(c_double_complex) :: value

		status = c_datablock_get_complex_wrapper(block, &
			trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value)

	end function datablock_get_complex

	function datablock_get_int_array_1d(block, section, name, value, size) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		integer(c_int), dimension(:), allocatable :: value
		integer(c_int) :: size
		integer(c_int) :: maxsize

		maxsize = datablock_get_array_length(block, section, name)
		! We don't actually know which failure we have here
		! So we just return 1
		if (maxsize<0) then
			status = 1
		else
			allocate(value(maxsize))
			status = c_datablock_get_int_array_1d_preallocated_wrapper(block, &
				trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, size, maxsize)
		endif

	end function datablock_get_int_array_1d

	function datablock_get_double_array_1d(block, section, name, value, size) result(status)
		integer(cosmosis_status) :: status
		integer(cosmosis_block) :: block
		character(len=*) :: section
		character(len=*) :: name
		real(c_double), dimension(:), allocatable :: value
		integer(c_int) :: size
		integer(c_int) :: maxsize

		maxsize = datablock_get_array_length(block, section, name)
		! We don't actually know which failure we have here
		! So we just return 1
		if (maxsize<0) then
			status = 1
		else
			allocate(value(maxsize))
			status = c_datablock_get_double_array_1d_preallocated_wrapper(block, &
				trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, size, maxsize)
		endif

	end function datablock_get_double_array_1d

	!Create a datablock.
	!Unless you are writing a sampler you should not
	!have to use this function - you will be given the 
	!datablock you need.
	function make_datablock()
		integer(kind=cosmosis_block) :: make_datablock
		make_datablock =  make_c_datablock()
	end function make_datablock

end module cosmosis_modules

