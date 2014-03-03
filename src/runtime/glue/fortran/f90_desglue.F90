MODULE F90_DESGLUE
	USE ISO_C_BINDING
	USE ISO_C_UTILITIES
	USE dynamic_loading
	USE des_section_names
	implicit none

	!This declares the interface to DES pipeline interface functions
	ABSTRACT INTERFACE
	   FUNCTION DES_INTERFACE_FUNCTION(x) BIND(C)
	      USE ISO_C_BINDING
	      integer(C_SIZE_T), VALUE :: x
		  integer(C_INT) :: DES_INTERFACE_FUNCTION
	   END FUNCTION
	END INTERFACE


	!Bindings to make it easy to use the C library functions.
	interface
		
		!These functions ar all simply wrapped.
		!The "value" attribute says that the item is NOT a pointer in the 
		!C header declaration.
		!
		!We are pretending that the pointer arguments to the internal_fits handle
		!and the fitsfile object are just integers of size_t bytes, since these 
		!are completely compatible
		function alloc_internal_fits()  BIND(C, NAME='alloc_internal_fits')
			use iso_c_binding
			implicit none
			integer(c_size_t) alloc_internal_fits
		end function alloc_internal_fits

			
		
		subroutine delete_fits_object(internal_fits)  BIND(C, NAME='delete_fits_object')
			use iso_c_binding
			implicit none
			integer (c_size_t), value :: internal_fits
		end subroutine delete_fits_object
		
		function make_fits_object(internal_fits) bind(C, name='make_fits_object')
			use iso_c_binding
			implicit none
			integer (c_int) make_fits_object
			integer (c_size_t), value :: internal_fits
		end function make_fits_object

		function fitsfile_from_internal(internal_fits) bind(C, name='fitsfile_from_internal')
			use iso_c_binding
			implicit none
			integer (c_size_t) fitsfile_from_internal 
			integer (c_size_t), value  :: internal_fits
		end function fitsfile_from_internal
		
		function close_fits_object(fitsfile) bind(C, name='close_fits_object')
			use iso_c_binding
			implicit none
			integer(c_size_t), value :: fitsfile
			integer(c_int) :: close_fits_object
		end function close_fits_object
		
		
		
		! These functions take character string arguments, which we have to deal with a
		! bit more carefully.  Fortran handles strings by passing around an array for the 
		! characters and the string length (exactly how it handles other arrays)
		! whereas C passes around a simple pointer to the start of the string and 
		! signals the end of the string with a null character (i.e. a 1-byte zero).
		! 
		! To convert between the two we need to write wrappers that are below
		! which add the null characfter to the end of the string before passing
		! it to the bound C function.
		!
		! The function interfaces here are therefore used only internally by 
		! the proper wrapper functions below, which add the character.
		
		!int fits_dump_to_disc(internal_fits *F, char * filename)
		function wrap_fits_dump_to_disc(handle, filename) bind(C, name='fits_dump_to_disc')
			use iso_c_binding
			implicit none
			integer (c_int) :: wrap_fits_dump_to_disc
			integer (c_size_t), value :: handle
			character(c_char) :: filename		
		end function wrap_fits_dump_to_disc
		
		function wrap_fits_goto_extension(fitsfile, name) bind(C, name='fits_goto_extension')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_goto_extension
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
		end function wrap_fits_goto_extension

		function wrap_fits_goto_or_create_extension(fitsfile, name) &
		bind(C, name='fits_goto_or_create_extension')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_goto_or_create_extension
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
		end function wrap_fits_goto_or_create_extension
		
		
		function wrap_fits_create_new_table(fitsfile, name, ncol, cols, formats, units) &
		bind(C, name='fits_create_new_table')
			use iso_c_binding
			implicit none
			integer(c_int) wrap_fits_create_new_table
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			integer(c_int), value :: ncol
			type(c_ptr), dimension(ncol) :: cols, formats, units
		end function wrap_fits_create_new_table
		
		
		function wrap_fits_count_column_rows(fitsfile, name, number_rows) &
		 bind(C, name='fits_count_column_rows')
			use iso_c_binding
			implicit none
			integer (c_int) :: wrap_fits_count_column_rows
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			integer (c_int) :: number_rows
		end function wrap_fits_count_column_rows
		
		function wrap_fits_put_double_parameter(fitsfile, name, value, comment) &
		bind(C, name='fits_put_double_parameter')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_put_double_parameter
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			real(c_double), value :: value
			character(c_char) :: comment
		end function wrap_fits_put_double_parameter

		function wrap_fits_get_double_parameter(fitsfile, name, value) &
		bind(C, name='fits_get_double_parameter')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_get_double_parameter
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			real(c_double)  :: value
		end function wrap_fits_get_double_parameter

		function wrap_fits_get_double_parameter_default(fitsfile, name, value, default) &
		bind(C, name='fits_get_double_parameter_default')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_get_double_parameter_default
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			real(c_double)  :: value
			real(c_double), value  :: default
		end function wrap_fits_get_double_parameter_default

		function wrap_fits_put_int_parameter(fitsfile, name, value, comment) &
		bind(C, name='fits_put_int_parameter')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_put_int_parameter
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			integer(c_int), value :: value
			character(c_char) :: comment
		end function wrap_fits_put_int_parameter


		function wrap_fits_get_int_parameter(fitsfile, name, value) &
		bind(C, name='fits_get_int_parameter')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_get_int_parameter
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			integer(c_int)  :: value
		end function wrap_fits_get_int_parameter


		function wrap_fits_get_int_parameter_default(fitsfile, name, value, default) &
		bind(C, name='fits_get_int_parameter_default')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_get_int_parameter_default
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			integer(c_int)  :: value
			integer(c_int), value  :: default
		end function wrap_fits_get_int_parameter_default

		
		function wrap_fits_get_int_column_preallocated(fitsfile, name, data, max_rows, number_rows) &
		 bind(C, name='fits_get_int_column_preallocated')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_get_int_column_preallocated
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			integer (c_int) :: data   !Pointer to the start of the array
			integer (c_int), value :: max_rows  !input value - size of array
			integer (c_int) :: number_rows  !output value
		end function wrap_fits_get_int_column_preallocated

		function wrap_fits_get_double_column_preallocated(fitsfile, name, data, max_rows, number_rows) &
		 bind(C, name='fits_get_double_column_preallocated')
			use iso_c_binding
			implicit none
			integer (c_int) wrap_fits_get_double_column_preallocated
			integer (c_size_t), value :: fitsfile
			character(c_char) :: name
			real (c_double) :: data   !Pointer to the start of the array
			integer (c_int), value :: max_rows  !input value - size of array
			integer (c_int) :: number_rows  !output value
		end function wrap_fits_get_double_column_preallocated


		
		function wrap_fits_write_int_column(fitsfile, name, data, number_rows) &
		bind(C, name='fits_write_int_column')
			use iso_c_binding
			integer (c_int) :: wrap_fits_write_int_column
			integer (c_size_t), value :: fitsfile
			character(c_char)  :: name
			integer (c_int) :: data   !Pointer to the start of the array
			integer (c_int), value :: number_rows
		end function wrap_fits_write_int_column
		
		function wrap_fits_write_double_column(fitsfile, name, data, number_rows) &
		bind(C, name='fits_write_double_column')
			use iso_c_binding
			integer (c_int) :: wrap_fits_write_double_column
			integer (c_size_t), value :: fitsfile
			character(c_char)  :: name
			real (c_double) :: data   !Pointer to the start of the array
			integer (c_int), value :: number_rows
		end function wrap_fits_write_double_column

	end interface
	

 	interface fits_get_column
 		module procedure fits_get_column_double, fits_get_column_int
 	end interface

	interface fits_write_column
		module procedure fits_write_double_column, fits_write_int_column
	end interface



contains 

	!These are the proper wrappers for the functions above which take string arguments.
	
	function fits_dump_to_disc(handle, filename) result(status)
		integer(c_size_t) :: handle
		character(*) :: filename
		integer status
		status = wrap_fits_dump_to_disc(handle, trim(filename)//C_NULL_CHAR)
	end function fits_dump_to_disc
	
	function fits_goto_extension(fitsfile, name) result(status)
		integer (c_size_t) :: fitsfile
		character(*) :: name
		integer(c_int) :: status
		status = wrap_fits_goto_extension(fitsfile, trim(name)//C_NULL_CHAR)
	end function fits_goto_extension
	
	function fits_goto_or_create_extension(fitsfile, name) result(status)
		integer (c_size_t) :: fitsfile
		character(*) :: name
		integer(c_int) :: status
		status = wrap_fits_goto_or_create_extension(fitsfile, trim(name)//C_NULL_CHAR)
	end function fits_goto_or_create_extension
	
	function fits_count_column_rows(fitsfile, name, number_rows) result(status)
		integer(c_int) status
		integer(c_size_t) fitsfile
		character(*) :: name
		integer(c_int) :: number_rows
		status = wrap_fits_count_column_rows(fitsfile, trim(name)//C_NULL_CHAR, number_rows)
	end function fits_count_column_rows
	
	function fits_put_double_parameter(fitsfile, name, value, comment) result(status)
		use iso_c_binding
		implicit none
		integer (c_int) status
		integer (c_size_t) :: fitsfile
		character(*) :: name
		real(c_double) :: value
		character(*) :: comment
		status = wrap_fits_put_double_parameter(fitsfile, trim(name)//C_NULL_CHAR, value, trim(comment)//C_NULL_CHAR)
	end function fits_put_double_parameter



	function fits_get_double_parameter(fitsfile, name, value) result(status)
		use iso_c_binding
		implicit none
		integer (c_int) status
		integer (c_size_t) :: fitsfile
		character(*) :: name
		real(c_double)  :: value
		status = wrap_fits_get_double_parameter(fitsfile, trim(name)//C_NULL_CHAR, value)
	end function fits_get_double_parameter



	function fits_get_double_parameter_default(fitsfile, name, value, default) result(status)
		use iso_c_binding
		implicit none
		integer (c_int) status
		integer (c_size_t) :: fitsfile
		character(*) :: name
		real(c_double)  :: value
		real(c_double)  :: default
		status = wrap_fits_get_double_parameter_default(fitsfile, trim(name)//C_NULL_CHAR, value, default)
	end function fits_get_double_parameter_default


	function fits_get_int_parameter(fitsfile, name, value) result(status)
		use iso_c_binding
		implicit none
		integer (c_int) status
		integer (c_size_t) :: fitsfile
		character(*) :: name
		integer(c_int)  :: value
		status = wrap_fits_get_int_parameter(fitsfile, trim(name)//C_NULL_CHAR, value)
	end function fits_get_int_parameter

	function fits_put_int_parameter(fitsfile, name, value, comment) result(status)
		use iso_c_binding
		implicit none
		integer (c_int) status
		integer (c_size_t) :: fitsfile
		character(*) :: name
		integer(c_int) :: value
		character(*) :: comment
		status = wrap_fits_put_int_parameter(fitsfile, trim(name)//C_NULL_CHAR, value, trim(comment)//C_NULL_CHAR)
	end function fits_put_int_parameter

	function fits_get_int_parameter_default(fitsfile, name, value, default) result(status)
		use iso_c_binding
		implicit none
		integer (c_int) status
		integer (c_size_t) :: fitsfile
		character(*) :: name
		integer(c_int)  :: value
		integer(c_int)  :: default
		status = wrap_fits_get_int_parameter_default(fitsfile, trim(name)//C_NULL_CHAR, value, default)
	end function fits_get_int_parameter_default

	
	function f90_fits_close_file(fits_file)
		use iso_c_binding
		integer(c_int) :: f90_fits_close_file, status2
		integer(c_size_t) :: fits_file
		f90_fits_close_file = close_fits_object(fits_file)
	end function f90_fits_close_file
	
	
	
	

	function fits_get_column_double(fits_file, name, data, number_rows) result(status)
		use iso_c_binding
		integer status
		integer(c_size_t) :: fits_file
		character(*) :: name
		real(8), allocatable, dimension(:) :: data
		integer(c_int) :: number_rows
		integer(c_int) :: max_rows

		if (allocated(data)) then
			write(*,*) "Passed array already allocated into fits_get_column_double.  Failing."
			status=1
			return
		endif
		
		!Get number of rows and allocate space
		status = wrap_fits_count_column_rows(fits_file, trim(name) // C_NULL_CHAR, number_rows)
		if (status .ne. 0) then
			return
		endif

		allocate(data(number_rows))

		!Load data into array
		max_rows=number_rows
		status = wrap_fits_get_double_column_preallocated(fits_file, trim(name) // C_NULL_CHAR, data(1), max_rows, number_rows) 
		return
	end function fits_get_column_double


	function fits_get_column_double_preallocated(fits_file, name, data, max_rows, number_rows) result(status)
		use iso_c_binding
		integer status
		integer(c_size_t) :: fits_file
		character(*) :: name
		real(8), dimension(:) :: data
		real(8), dimension(:), allocatable :: col
	
		integer(c_int) :: number_rows
		integer(c_int) :: max_rows
	
		!Get number of rows and allocate space
		status = wrap_fits_count_column_rows(fits_file, trim(name) // C_NULL_CHAR, number_rows)
		if (status .ne. 0) then
			return
		endif

		allocate(col(number_rows))
		!Load data into array
		status = wrap_fits_get_double_column_preallocated(fits_file, trim(name) // C_NULL_CHAR, col(1), max_rows, number_rows)
		if (number_rows>max_rows) number_rows=max_rows
		data(1:number_rows) = col(1:number_rows)
		deallocate(col)
	
		return
	end function fits_get_column_double_preallocated


	function fits_get_column_int_preallocated(fits_file, name, data, max_rows, number_rows) result(status)
		use iso_c_binding
		integer status
		integer(c_size_t) :: fits_file
		character(*) :: name
		integer(c_int), dimension(:) :: data
		integer(c_int), dimension(:), allocatable :: col
	
		integer(c_int) :: number_rows
		integer(c_int) :: max_rows
	
		!Get number of rows and allocate space
		status = wrap_fits_count_column_rows(fits_file, trim(name) // C_NULL_CHAR, number_rows)
		if (status .ne. 0) then
			return
		endif

		allocate(col(number_rows))
		!Load data into array
		status = wrap_fits_get_int_column_preallocated(fits_file, trim(name) // C_NULL_CHAR, col(1), max_rows, number_rows)
		if (number_rows>max_rows) number_rows=max_rows
		data(1:number_rows) = col(1:number_rows)
		deallocate(col)
	
		return
	end function fits_get_column_int_preallocated



	function fits_get_column_int(fits_file, name, data, number_rows) result(status)
		use iso_c_binding
		integer status
		integer(c_size_t) :: fits_file
		character(*) :: name
		integer, allocatable, dimension(:) :: data
		integer(c_int) :: number_rows
		integer(c_int) :: max_rows

		if (allocated(data)) then
			write(*,*) "Passed array already allocated into fits_get_column_double.  Failing."
			status=1
			return
		endif
	
		!Get number of rows and allocate space
		status = wrap_fits_count_column_rows(fits_file, trim(name) // C_NULL_CHAR, number_rows)
		if (status .ne. 0) then
			return
		endif
	
		allocate(data(number_rows))

		!Load data into array
		max_rows=number_rows
		status = wrap_fits_get_int_column_preallocated(fits_file, trim(name) // C_NULL_CHAR, data(1), max_rows, number_rows) 
		return
	end function fits_get_column_int

	function fits_write_int_column(fitsfile, name, data) result(status)
		use iso_c_binding
		integer(c_size_t) :: fitsfile
		integer status
		character(*) :: name
		integer(c_int), dimension(:) :: data
		integer (c_int) number_rows
		number_rows=size(data)

		status = wrap_fits_write_int_column(fitsfile, trim(name)//C_NULL_CHAR, data(1), number_rows)
	end function fits_write_int_column

	function fits_write_double_column(fitsfile, name, data) result(status)
		use iso_c_binding
		integer(c_size_t) :: fitsfile
		integer status
		character(*) :: name
		real(c_double), dimension(:) :: data
		integer (c_int) number_rows
		number_rows=size(data)
		status = wrap_fits_write_double_column(fitsfile, trim(name)//C_NULL_CHAR, data(1), number_rows)
	end function fits_write_double_column
		


	function fits_create_new_table(fitsfile, name, column_names, formats, units) result(status)
		use iso_c_binding
		integer(c_size_t) :: fitsfile
		character(*) :: name
		integer status
		character(*), dimension(1:) :: column_names, formats, units
		integer(c_int) :: ncol
		integer i
		
		TYPE(C_PTR), DIMENSION(size(column_names)) :: column_name_ptrs, format_ptrs, unit_ptrs
		CHARACTER(LEN=80), DIMENSION(size(column_names)), TARGET :: column_names_c, formats_c, units_c
		
		
		ncol=size(column_names)
		if ((size(formats) .ne. ncol)   .or.  (size(units) .ne. ncol)) then
			write(*,*) "Tried to use fits_create_new_table but arrays sizes were different among column_names, formats, units"
			status=1
			return
		endif
		
		do i=1,ncol
			column_names_c(i) = trim(column_names(i))//C_NULL_CHAR
			column_name_ptrs(i) = C_LOC(column_names_c(i))
			formats_c(i) = trim(formats(i))//C_NULL_CHAR
			format_ptrs(i) = C_LOC(formats_c(i))
			units_c(i) = trim(units(i))//C_NULL_CHAR
			unit_ptrs(i) = C_LOC(units_c(i))
		enddo

		status = wrap_fits_create_new_table(fitsfile, trim(name)//C_NULL_CHAR, ncol, column_name_ptrs, format_ptrs, unit_ptrs)
		return
	end function fits_create_new_table
	

#ifndef NO_F90_INTERFACE

	!Load a DESGLUE interface function from a dynamic library.
	!This will make it possible to load functions into a fortran sampler program or similar
	function load_interface(library_name, function_name) result(interface_function)
	   CHARACTER(KIND=C_CHAR,LEN=*) :: library_name, function_name
	   PROCEDURE(DES_INTERFACE_FUNCTION), POINTER :: interface_function ! Dynamically-linked procedure

	   TYPE(C_FUNPTR) :: funptr=C_NULL_FUNPTR
	   TYPE(C_PTR) :: handle=C_NULL_PTR
	   INTEGER(C_INT) :: status
		! Open the DL:
		handle=DLOpen(TRIM(library_name)//C_NULL_CHAR, IOR(RTLD_NOW, RTLD_LOCAL))
		!Null the interface function to start with so that
		!if we fail it not be a dangling pointer.
		nullify(interface_function)

		!Check if the function exists.
		IF(.NOT.C_ASSOCIATED(handle)) THEN
		   WRITE(*,*) "Error in dlopen: ", C_F_STRING(DLError())
		   return
		END IF

		! Find the subroutine in the DL:
		funptr=DLSym(handle,TRIM(function_name)//C_NULL_CHAR)
		IF(.NOT.C_ASSOCIATED(funptr)) THEN
			!If that did not work, try loading the function with an underscore appended.
			funptr=DLSym(handle,TRIM(function_name)// "_"//C_NULL_CHAR)

			!If it has still failed then report an error.
			IF(.NOT.C_ASSOCIATED(funptr)) THEN
		   		WRITE(*,*) "Error in dlsym: ", C_F_STRING(DLError())
			   return
			ENDIF
		END IF
	
		! Now convert the C function pointer to a Fortran procedure pointer

		CALL C_F_PROCPOINTER(CPTR=funptr, FPTR=interface_function)
		!The interface_function is now in the result
	end function
	
	
	
	function simple_total_likelihood_function(parameter_names, parameter_values, library_names, function_names) result(like)
		CHARACTER(KIND=C_CHAR,LEN=*), dimension(:) :: parameter_names, library_names, function_names
		real(8), dimension(:) :: parameter_values
		PROCEDURE(DES_INTERFACE_FUNCTION), POINTER :: interface_function
		real(8) :: like, wl_like, sn_like, pk_like, bao_like, cluster_like, cmb_like, h0_like, bbn_like  !any others?
		integer(c_size_t) :: handle
		integer(c_size_t) :: fitsfile
		integer nparam,p
		integer (c_int) status
		
		status = 0
		
		nparam = size(parameter_names)
		if (size(parameter_values) .ne. nparam) then
			write(*,*) "Parameter name and value arrays different size in simple_total_likelihood_function"
			like = -1e30
			return
		endif
		
		!Create FITS file and write data to it
		handle = alloc_internal_fits()
		status = make_fits_object(handle)
		fitsfile = fitsfile_from_internal(handle)
		status = status + fits_goto_or_create_extension(fitsfile, "COSMOPAR")
		!Write the parameters
		do p=1,nparam
			status = status + fits_put_double_parameter(fitsfile, trim(parameter_names(p)), parameter_values(p), "A cosmological parameter")
		enddo
		
		!Close the fits file so that we can open it cleanly in the other modules
		status = status + close_fits_object(fitsfile)
		
		!If anything has gone wrong up to here then we clean up and return a bad likeliood without trying to run the interfaces
		if (status .ne. 0) then
			write(*,*) "Failed to properly set up FITS file - see messages."
			call delete_fits_object(handle)
			write(*,*) "Setting like to bad value"
			like = -1e30
			return
		endif
		
		!If we are okay so far then run the interfaces
		status = run_module_sequence(handle, library_names, function_names)

		!Now we need to get out the likelihood values
		if (status .eq. 0) then
			!Re-open the file and go the the likelihoods extension.
			fitsfile = fitsfile_from_internal(handle)
			status = status + fits_goto_extension(fitsfile, "LIKELIHOODS")
			
			!Get the parameters, with default values of zero if they are not there.
			!wl_like, sn_like, pk_like, bao_like, cluster_like
			status = status + fits_get_double_parameter_default(fitsfile, "WL_LIKE", wl_like, 0.0d0)
			status = status + fits_get_double_parameter_default(fitsfile, "SN_LIKE", sn_like, 0.0d0)
			status = status + fits_get_double_parameter_default(fitsfile, "PK_LIKE", pk_like, 0.0d0)
			status = status + fits_get_double_parameter_default(fitsfile, "BAO_LIKE", bao_like, 0.0d0)
			status = status + fits_get_double_parameter_default(fitsfile, "CLUSTER_LIKE", cluster_like, 0.0d0)
			status = status + fits_get_double_parameter_default(fitsfile, "CMB_LIKE", cmb_like, 0.0d0)
			status = status + fits_get_double_parameter_default(fitsfile, "H0_LIKE", h0_like, 0.0d0)    !Maybe too simple to be here
			status = status + fits_get_double_parameter_default(fitsfile, "BBN_LIKE", bbn_like, 0.0d0)  !Ditto.
			!This bit should not fail just because the default was used. 
			!If it has then something else is wrong.
			if (status .eq. 0) then
				like = wl_like + sn_like + pk_like + bao_like + cluster_like + cmb_like + h0_like + bbn_like
			else
				write(*,*) "Failed to read results from fits file cleanly."
				write(*,*) "One possibility is that one of the modules failed to close the FITS file properly, or corrupted it somehow"
				write(*,*) "Setting like to bad value."
				like = -1e30
			endif
		else
			write(*,*) "Failed to run the module sequence cleanly."
			write(*,*) "Setting like to bad value."
			like = -1e30
		endif
		
		call delete_fits_object(handle)
		return
	end function simple_total_likelihood_function
	
	
	
	function run_module_sequence(fits_handle, library_names, function_names) result(status)
		CHARACTER(KIND=C_CHAR,LEN=*), dimension(:) :: library_names, function_names
		PROCEDURE(DES_INTERFACE_FUNCTION), POINTER :: interface_function
		integer(c_size_t) :: fits_handle
		integer status
		integer n, i
		status = 0
		n = size(library_names)
		do i=1,n
			write(*,*) "Looking for library:", library_names(i)
			write(*,*) "And function: ", function_names(i)
			interface_function = load_interface(library_names(i), function_names(i))
			if (.not. associated(interface_function)) then
				write(*,*) "Interface function not found! Library:", library_names(i), " function:", function_names(i)
				status = 1
			else
				status = interface_function(fits_handle)
			endif
			
			if (status .ne. 0) return
		enddo
		
		return
	end function 
#endif	
	
END MODULE F90_DESGLUE






