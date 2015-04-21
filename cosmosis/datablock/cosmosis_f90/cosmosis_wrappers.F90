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

        function destroy_c_datablock(block) bind(C, name="destroy_c_datablock")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer(kind=cosmosis_block), value :: block
            integer (cosmosis_status) :: destroy_c_datablock
        end function destroy_c_datablock


        function c_datablock_num_sections_wrapper(block) bind(C, name="c_datablock_num_sections")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer(kind=cosmosis_block), value :: block
            integer(c_int) :: c_datablock_num_sections_wrapper
        end function c_datablock_num_sections_wrapper




        function c_datablock_get_array_length_wrapper(s, section, name) bind(C, name="c_datablock_get_array_length")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (kind=c_int) :: c_datablock_get_array_length_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section, name
        end function c_datablock_get_array_length_wrapper


        function c_datablock_has_section_wrapper(s, name) bind(C, name="c_datablock_has_section")
            use iso_c_binding
            use cosmosis_types
            implicit none
            logical (kind=c_bool) :: c_datablock_has_section_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: name
        end function c_datablock_has_section_wrapper

        function c_datablock_get_section_name_wrapper(s, i) bind(C, name="c_datablock_get_section_name")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer(kind=cosmosis_block), value :: s
            integer(kind=c_int), value :: i
            type(c_ptr) :: c_datablock_get_section_name_wrapper
        end function c_datablock_get_section_name_wrapper


!  const char* c_datablock_get_section_name(c_datablock const* s, int i)

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

        function c_datablock_put_bool_wrapper(s, section, name, value) bind(C, name="c_datablock_put_bool")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_put_bool_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            logical(kind=c_bool), value :: value
        end function c_datablock_put_bool_wrapper

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

        function c_datablock_put_string_wrapper(s, section, name, value) &
        bind(C, name="c_datablock_put_string")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_put_string_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            character(kind=c_char), dimension(*) :: value
        end function c_datablock_put_string_wrapper



        function c_datablock_put_int_array_1d_wrapper(s, section, name, value, sz) &
        bind(C, name="c_datablock_put_int_array_1d")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_put_int_array_1d_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: sz
            integer(kind=c_int), dimension(sz) :: value
        end function c_datablock_put_int_array_1d_wrapper

        function c_datablock_put_double_array_1d_wrapper(s, section, name, value, sz) &
        bind(C, name="c_datablock_put_double_array_1d")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_put_double_array_1d_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: sz
            real(kind=c_double), dimension(sz) :: value
        end function c_datablock_put_double_array_1d_wrapper

        function c_datablock_put_double_array_wrapper(s, section, name, value, ndims, extents) &
        bind(C, name="c_datablock_put_double_array")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_put_double_array_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            real(kind=c_double), dimension(*) :: value
            integer(kind=c_int), value :: ndims
            integer(kind=c_int), dimension(ndims) :: extents

        end function c_datablock_put_double_array_wrapper

        function c_datablock_get_double_array_wrapper(s, section, name, value, ndims, extents) &
        bind(C, name="c_datablock_get_double_array")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_double_array_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            real(kind=c_double), dimension(*) :: value
            integer(kind=c_int), value :: ndims
            integer(kind=c_int), dimension(ndims) :: extents

        end function c_datablock_get_double_array_wrapper

        function c_datablock_get_int_array_wrapper(s, section, name, value, ndims, extents) &
        bind(C, name="c_datablock_get_int_array")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_int_array_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), dimension(*) :: value
            integer(kind=c_int), value :: ndims
            integer(kind=c_int), dimension(ndims) :: extents

        end function c_datablock_get_int_array_wrapper



        function c_datablock_put_int_array_wrapper(s, section, name, value, ndims, extents) &
        bind(C, name="c_datablock_put_int_array")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_put_int_array_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), dimension(*) :: value
            integer(kind=c_int), value :: ndims
            integer(kind=c_int), dimension(ndims) :: extents

        end function c_datablock_put_int_array_wrapper

        function c_datablock_put_complex_array_1d_wrapper(s, section, name, value, sz) &
        bind(C, name="c_datablock_put_complex_array_1d")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_put_complex_array_1d_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: sz
            complex(kind=c_double_complex), dimension(sz) :: value
        end function c_datablock_put_complex_array_1d_wrapper



        !DATABLOCK_STATUS c_datablock_put_int(c_datablock* s, const char* section, const char* name, int val);
        function c_datablock_replace_int_wrapper(s, section, name, value) bind(C, name="c_datablock_replace_int")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_replace_int_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int)  :: value
        end function c_datablock_replace_int_wrapper

        function c_datablock_replace_bool_wrapper(s, section, name, value) bind(C, name="c_datablock_replace_bool")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_replace_bool_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            logical(kind=c_bool)  :: value
        end function c_datablock_replace_bool_wrapper

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

        function c_datablock_replace_string_wrapper(s, section, name, value) &
        bind(C, name="c_datablock_replace_string")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_replace_string_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            character(kind=c_char), dimension(*) :: value
        end function c_datablock_replace_string_wrapper


        function c_datablock_replace_int_array_1d_wrapper(s, section, name, value, sz) &
        bind(C, name="c_datablock_replace_int_array_1d")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_replace_int_array_1d_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: sz
            integer(kind=c_int), dimension(sz) :: value
        end function c_datablock_replace_int_array_1d_wrapper

        function c_datablock_replace_double_array_1d_wrapper(s, section, name, value, sz) &
        bind(C, name="c_datablock_replace_double_array_1d")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_replace_double_array_1d_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: sz
            real(kind=c_double), dimension(sz) :: value
        end function c_datablock_replace_double_array_1d_wrapper

        function c_datablock_replace_complex_array_1d_wrapper(s, section, name, value, sz) &
        bind(C, name="c_datablock_replace_complex_array_1d")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_replace_complex_array_1d_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: sz
            complex(kind=c_double_complex), dimension(sz) :: value
        end function c_datablock_replace_complex_array_1d_wrapper

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

        !DATABLOCK_STATUS c_datablock_get_int(c_datablock* s, const char* section, const char* name, int *val);
        function c_datablock_get_bool_wrapper(s, section, name, value) bind(C, name="c_datablock_get_bool")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_bool_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            logical(kind=c_bool) :: value
        end function c_datablock_get_bool_wrapper

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

        function c_datablock_get_string_wrapper(s, section, name, value) bind(C, name="c_datablock_get_string")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_string_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            type(c_ptr) :: value
        end function c_datablock_get_string_wrapper


        !DATABLOCK_STATUS c_datablock_get_int(c_datablock* s, const char* section, const char* name, int *val);
        function c_datablock_get_int_default_wrapper(s, section, name, default, value) bind(C, name="c_datablock_get_int_default")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_int_default_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int) :: value
            integer(kind=c_int), value :: default
        end function c_datablock_get_int_default_wrapper

        function c_datablock_get_bool_default_wrapper(s, section, name, default, value) bind(C, name="c_datablock_get_bool_default")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_bool_default_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            logical(kind=c_bool) :: value
            logical(kind=c_bool), value :: default
        end function c_datablock_get_bool_default_wrapper


        function c_datablock_get_double_default_wrapper(s, section, name, default, value) bind(C, name="c_datablock_get_double_default")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_double_default_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            real(kind=c_double) :: value
            real(kind=c_double), value :: default
        end function c_datablock_get_double_default_wrapper

        function c_datablock_get_complex_default_wrapper(s, section, name, default, value) bind(C, name="c_datablock_get_complex_default")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_complex_default_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            complex(kind=c_double_complex) :: value
            complex(kind=c_double_complex), value :: default
        end function c_datablock_get_complex_default_wrapper


        function c_datablock_get_string_default_wrapper(s, section, name, default, value) bind(C, name="c_datablock_get_string_default")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_string_default_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            character(kind=c_char), dimension(*) :: default
            type(c_ptr) :: value
        end function c_datablock_get_string_default_wrapper

        function c_datablock_get_int_array_1d_preallocated_wrapper(s, section, name, value, size, maxsize) &
        bind(C, name="c_datablock_get_int_array_1d_preallocated")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_int_array_1d_preallocated_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: maxsize
            integer(kind=c_int) :: value(maxsize)
            integer(kind=c_int) :: size
        end function c_datablock_get_int_array_1d_preallocated_wrapper

        function c_datablock_get_double_array_1d_preallocated_wrapper(s, section, name, value, size, maxsize) &
        bind(C, name="c_datablock_get_double_array_1d_preallocated")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_double_array_1d_preallocated_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: maxsize
            real(kind=c_double) :: value(maxsize)
            integer(kind=c_int) :: size
        end function c_datablock_get_double_array_1d_preallocated_wrapper

        function c_datablock_get_complex_array_1d_preallocated_wrapper(s, section, name, value, size, maxsize) &
        bind(C, name="c_datablock_get_complex_array_1d_preallocated")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_complex_array_1d_preallocated_wrapper
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: maxsize
            complex(kind=c_double_complex) :: value(maxsize)
            integer(kind=c_int) :: size
        end function c_datablock_get_complex_array_1d_preallocated_wrapper


        function c_datablock_get_metadata(s, section, name, key, value) &
        bind(C, name="c_datablock_get_metadata")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_metadata
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            character(kind=c_char), dimension(*) :: key
            type(c_ptr) :: value
        end function c_datablock_get_metadata

        function c_datablock_put_metadata(s, section, name, key, value) &
        bind(C, name="c_datablock_put_metadata")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_put_metadata
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            character(kind=c_char), dimension(*) :: key
            character(kind=c_char), dimension(*) :: value
        end function c_datablock_put_metadata

        function c_datablock_replace_metadata(s, section, name, key, value) &
        bind(C, name="c_datablock_replace_metadata")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_replace_metadata
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            character(kind=c_char), dimension(*) :: key
            character(kind=c_char), dimension(*) :: value
        end function c_datablock_replace_metadata

        function c_datablock_get_int_array_shape(s, section, name, ndims, extents) &
            bind(C, name="c_datablock_get_int_array_shape")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_int_array_shape
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int), value :: ndims
            integer(kind=c_int), dimension(ndims) :: extents
        end function c_datablock_get_int_array_shape

        function c_datablock_get_double_array_shape(s, section, name, ndims, extents) &
            bind(C, name="c_datablock_get_double_array_shape")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_double_array_shape
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int),value :: ndims
            integer(kind=c_int), dimension(ndims) :: extents
        end function c_datablock_get_double_array_shape

        function c_datablock_get_complex_array_shape(s, section, name, ndims, extents) &
            bind(C, name="c_datablock_get_complex_array_shape")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_get_complex_array_shape
            integer(kind=cosmosis_block), value :: s
            character(kind=c_char), dimension(*) :: section
            character(kind=c_char), dimension(*) :: name
            integer(kind=c_int),value :: ndims
            integer(kind=c_int), dimension(ndims) :: extents
        end function c_datablock_get_complex_array_shape

        function c_datablock_print_log(s) bind(C, name="c_datablock_print_log")
            use iso_c_binding
            use cosmosis_types
            implicit none
            integer (cosmosis_status) :: c_datablock_print_log
            integer(kind=cosmosis_block), value :: s
        end function c_datablock_print_log


    function wrap_strlen(str) bind(C, name='strlen')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: str
        integer(c_size_t) :: wrap_strlen
    end function wrap_strlen

    subroutine wrap_free(p) bind(C, name='free')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: p
    end subroutine wrap_free



    end interface

    contains



    function c_string_to_fortran(c_str, max_len) result(f_str)
        use iso_c_binding
        character(max_len) :: f_str
        character, pointer, dimension(:) :: p_str
        type(c_ptr) :: c_str
        integer :: max_len
        integer(c_size_t) :: n, shpe(1)
        integer(c_size_t) :: i

        !Initialize an empty string
        do i=1,max_len-1
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


end module cosmosis_wrappers