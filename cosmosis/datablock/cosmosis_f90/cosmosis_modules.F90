!Main module for module writers to actually use.
!Could just rename this cosmosis.F90 and the module cosmosis?
!That might be easiest for users
module cosmosis_modules
    use cosmosis_types
    use cosmosis_wrappers
    use cosmosis_section_names
    implicit none

    integer, parameter :: DATABLOCK_MAX_STRING_LENGTH=256
    character(*), parameter :: option_section = "module_options"

    interface datablock_put
        module procedure datablock_put_int, datablock_put_double, &
            datablock_put_logical, datablock_put_complex, datablock_put_double_array_1d, &
            datablock_put_int_array_1d, datablock_put_string
    end interface

    interface datablock_get
        module procedure datablock_get_int, datablock_get_double, &
            datablock_get_logical, datablock_get_complex, datablock_get_double_array_1d, &
            datablock_get_int_array_1d, datablock_get_string
    end interface

    interface datablock_replace
        module procedure datablock_replace_int, datablock_replace_double, &
            datablock_replace_logical, datablock_replace_complex, datablock_replace_double_array_1d, &
            datablock_replace_int_array_1d, datablock_replace_string
    end interface

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

    function datablock_num_sections(block) result(n)
        integer(cosmosis_block) :: block
        integer :: n

        n = c_datablock_num_sections_wrapper(block)
    end function datablock_num_sections


    function datablock_get_array_length(block, section, name) result(n)
        integer(c_int) :: n
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name

        n = c_datablock_get_array_length_wrapper(block, trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR)
    end function datablock_get_array_length     


    function datablock_get_section_name(block, i) result(name)
        character(DATABLOCK_MAX_STRING_LENGTH) :: name
        integer(cosmosis_block) :: block
        integer :: i

        type(c_ptr) :: c_name
        c_name = c_datablock_get_section_name_wrapper(block, i)
        name = c_string_to_fortran(c_name, DATABLOCK_MAX_STRING_LENGTH)
        !No free here because we do not get
        !a newly allocated copy
    end function datablock_get_section_name

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

    !Load the named integer from the given section
    function datablock_get_int_default(block, section, name, default, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(*) :: section
        character(*) :: name
        integer :: default, value

        status = c_datablock_get_int_default_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, default, value)

    end function datablock_get_int_default


    function datablock_put_logical(block, section, name, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        logical :: value
        logical(c_bool) c_value

        c_value = value
        status = c_datablock_put_bool_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, c_value)

    end function datablock_put_logical

    function datablock_replace_logical(block, section, name, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        logical :: value
        logical(c_bool) c_value

        c_value = value
        status = c_datablock_replace_bool_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, c_value)

    end function datablock_replace_logical

    !Load the named integer from the given section
    function datablock_get_logical(block, section, name, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(*) :: section
        character(*) :: name
        logical :: value
        logical(c_bool) :: c_value

        status = c_datablock_get_bool_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, c_value)
        value = c_value

    end function datablock_get_logical

    !Load the named integer from the given section
    function datablock_get_logical_default(block, section, name, default, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(*) :: section
        character(*) :: name
        logical :: default, value
        logical(c_bool) :: c_default, c_value

        c_default = default
        status = c_datablock_get_bool_default_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, c_default, c_value)
        value = c_value
    end function datablock_get_logical_default


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


    function datablock_get_double_default(block, section, name, default, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(*) :: section
        character(*) :: name
        real(c_double) :: value, default

        status = c_datablock_get_double_default_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, default, value)

    end function datablock_get_double_default


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


    function datablock_get_complex_default(block, section, name, default, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(*) :: section
        character(*) :: name
        complex(c_double_complex) :: default, value

        status = c_datablock_get_complex_default_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, default, value)

    end function datablock_get_complex_default


    function datablock_put_string(block, section, name, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        character(len=*) :: value

        status = c_datablock_put_string_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, trim(value)//C_NULL_CHAR)

    end function datablock_put_string

    function datablock_replace_string(block, section, name, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        character(len=*) :: value

        status = c_datablock_replace_string_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, trim(value)//C_NULL_CHAR)

    end function datablock_replace_string

    function datablock_get_string(block, section, name, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        character(len=*) :: value
        type(c_ptr) :: c_string  !This is actually a pointer-to-a-pointer, I think.

        !Call the C function, which returns a c_ptr.
        status = c_datablock_get_string_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, c_string)

        ! Convert the c_ptr into a fortran string.
        ! This will (silently) truncate the string if
        ! the value put in is not long enough,
        ! but this is apparently standard in Fortran.
        if (status==0) then
            value = c_string_to_fortran(c_string, len(value))
            !Need to free the C string!  Becuase it was allocated with strdup
            call wrap_free(c_string)
        else 
            value=""
        endif

    end function datablock_get_string


    function datablock_get_string_default(block, section, name, default, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        character(len=*) :: default, value
        type(c_ptr) :: c_string  !This is actually a pointer-to-a-pointer, I think.

        !Call the C function, which returns a c_ptr.
        status = c_datablock_get_string_default_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, trim(default)//C_NULL_CHAR, c_string)

        ! Convert the c_ptr into a fortran string.
        ! This will (silently) truncate the string if
        ! the value put in is not long enough,
        ! but this is apparently standard in Fortran.
        value = c_string_to_fortran(c_string, len(value))
        !Need to free the C string!  Becuase it was allocated with strdup
        call wrap_free(c_string)

    end function datablock_get_string_default


!       function c_datablock_put_int_array_1d_wrapper(s, section, name, value, size)

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

    function datablock_put_double_array_2d(block, section, name, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        real(c_double), dimension(:, :) :: value
        integer(c_int) :: extents(2)

        !Opposite ordering in fortran vs C
        extents(2)=size(value, 1)
        extents(1)=size(value, 2)

        write(*,*) extents
 
        status = c_datablock_put_double_array_wrapper(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, 2, extents)
 
    end function datablock_put_double_array_2d


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


!    function datablock_get_complex_array_1d(block, section, name, value, size) result(status)
!        integer(cosmosis_status) :: status
!        integer(cosmosis_block) :: block
!        character(len=*) :: section
!        character(len=*) :: name
!        complex(c_double_complex), dimension(:), allocatable :: value
!        integer(c_int) :: size
!        integer(c_int) :: maxsize
! 
!        maxsize = datablock_get_array_length(block, section, name)
!        ! We don't actually know which failure we have here
!        ! So we just return 1
!        if (maxsize<0) then
!            status = 1
!        else
!            allocate(value(maxsize))
!            status = c_datablock_get_complex_array_1d_preallocated_wrapper(block, &
!                trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, size, maxsize)
!        endif
! 
!    end function datablock_get_complex_array_1d


!    function datablock_put_complex_array_1d(block, section, name, value) result(status)
!        integer(cosmosis_status) :: status
!        integer(cosmosis_block) :: block
!        character(len=*) :: section
!        character(len=*) :: name
!        complex(c_double_complex), dimension(:) :: value
!        integer(c_int) :: sz
! 
!        sz=size(value)
! 
!        status = c_datablock_put_complex_array_1d_wrapper(block, &
!            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, sz)
! 
!    end function datablock_put_complex_array_1d
! 

 !   function datablock_replace_complex_array_1d(block, section, name, value) result(status)
 !       integer(cosmosis_status) :: status
 !       integer(cosmosis_block) :: block
 !       character(len=*) :: section
 !       character(len=*) :: name
 !       complex(c_double_complex), dimension(:) :: value
 !       integer(c_int) :: sz
 !
 !       sz=size(value)
 !
 !       status = c_datablock_replace_complex_array_1d_wrapper(block, &
 !           trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, value, sz)
 !
 !   end function datablock_replace_complex_array_1d
 

    function datablock_put_double_grid(s, section, &
        x_name, x, y_name, y, z_name, z) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: s
        character(*) :: section, x_name, y_name, z_name
        character(256) :: name_x, name_y, name_z
        integer :: nz
        real(8) :: x(:), y(:), z(:,:)
        real(8), allocatable, dimension(:) :: z_flat
        character(512) :: sentinel_key, sentinel_value

        status = 0
        name_x = x_name
        name_y = y_name
        name_z = z_name
        call lowercase_ascii(name_x)
        call lowercase_ascii(name_y)
        call lowercase_ascii(name_z)

        status = status + datablock_put_double_array_1d(s, section, name_x, x)
        status = status + datablock_put_double_array_1d(s, section, name_y, y)
        nz = size(z)

        ! Save z as 1D for now, since 2D not ready
        allocate(z_flat(nz))
        z_flat = reshape(z, shape(z_flat))
        status = status + datablock_put_double_array_1d(s, section, name_z, z_flat)
        deallocate(z_flat)

        write(sentinel_key, '("_cosmosis_order_", A)') trim(name_z)
        write(sentinel_value, '(A,"_cosmosis_order_", A)') trim(name_y), trim(name_x)

        status = status + datablock_put_string(s, section, trim(sentinel_key), trim(sentinel_value))
    end function datablock_put_double_grid

    !There appears to be no intrinsic to do this!
    !This does mean our code will not work for non-ascii characters
    !When this becomes an issue because Mandarin becomes the international
    !Lingua Franca then we should probably switch to wrapping
    !something from libc.
    subroutine lowercase_ascii(s)
        character(*) :: s
        integer i,n, ic
        character(26), parameter :: lower = "abcdefghijklmnopqrstuvwxyz"
        character(26), parameter :: upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        n=len(trim(s))
        do i=1,n
            ic = index(upper, s(i:i))
            if (ic>0) s(i:i) = lower(ic:ic)
        enddo
    end subroutine lowercase_ascii

    function datablock_get_double_grid(s, section, &
        x_name, x, y_name, y, z_name, z) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: s
        character(*) :: section, x_name, y_name, z_name
        character(256) :: name_x, name_y, name_z
        integer :: nx, ny, nz
        real(8), allocatable :: x(:), y(:), z(:,:), z_T(:,:)
        real(8), allocatable, dimension(:) :: z_flat
        character(512) :: sentinel_key, sentinel_value

        status = 0

        name_x = x_name
        name_y = y_name
        name_z = z_name


        call lowercase_ascii(name_x)
        call lowercase_ascii(name_y)
        call lowercase_ascii(name_z)


        status = status + datablock_get_double_array_1d(s, section, name_x, x, nx)
        status = status + datablock_get_double_array_1d(s, section, name_y, y, ny)
        status = status + datablock_get_double_array_1d(s, section, name_z, z_flat, nz)

        allocate(z(nx,ny))

        !Ordering check
        write(sentinel_key, '("_cosmosis_order_", A)') trim(name_z)
        status = status + datablock_get_string(s, section, sentinel_key, sentinel_value)

        if (status .ne. 0) then
            if (allocated(x)) deallocate(x)
            if (allocated(y)) deallocate(y)
            if (allocated(z)) deallocate(z)
            if (allocated(z_flat)) deallocate(z_flat)
            return
        endif

        if (sentinel_value==(trim(name_y) // "_cosmosis_order_" // trim(name_x))) then
            !Simple ordering
            z = reshape(z_flat, shape(z))
        elseif (sentinel_value==(trim(name_x) // "_cosmosis_order_" // trim(name_y))) then
            !Need to transpose first
            allocate(z_T(ny,nx))
            z_T = reshape(z_flat, shape(z_T))
            z = transpose(z_T)
            deallocate(z_T)
        else
            !Something went wrong
            write(*,*) "Marker error", trim(sentinel_key), "  ",trim(sentinel_value)
            write(*,*) (trim(name_y) // "_cosmosis_order_" // trim(name_x))
            write(*,*) (trim(name_x) // "_cosmosis_order_" // trim(name_y))
            status = 8
        endif

        if (allocated(z_flat)) deallocate(z_flat)    
    end function datablock_get_double_grid



    function datablock_put_metadata(block, section, name, key, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        character(len=*) :: key
        character(len=*) :: value

        status = c_datablock_put_metadata(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, &
            trim(key)//C_NULL_CHAR, trim(value)//C_NULL_CHAR)

    end function datablock_put_metadata

    function datablock_replace_metadata(block, section, name, key, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        character(len=*) :: key
        character(len=*) :: value

        status = c_datablock_replace_metadata(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, &
            trim(key)//C_NULL_CHAR, trim(value)//C_NULL_CHAR)

    end function datablock_replace_metadata


   function datablock_get_metadata(block, section, name, key, value) result(status)
        integer(cosmosis_status) :: status
        integer(cosmosis_block) :: block
        character(len=*) :: section
        character(len=*) :: name
        character(len=*) :: key
        character(len=*) :: value
        type(c_ptr) :: c_string  !This is actually a pointer-to-a-pointer, I think.

        !Call the C function, which returns a c_ptr.
        status = c_datablock_get_metadata(block, &
            trim(section)//C_NULL_CHAR, trim(name)//C_NULL_CHAR, &
            trim(key)//C_NULL_CHAR, c_string)

        ! Convert the c_ptr into a fortran string.
        ! This will (silently) truncate the string if
        ! the value put in is not long enough,
        ! but this is apparently standard in Fortran.
        if (status==0) then
            value = c_string_to_fortran(c_string, len(value))
            !Need to free the C string!  Becuase it was allocated with strdup
            call wrap_free(c_string)
        else 
            value=""
        endif

    end function datablock_get_metadata



    !Create a datablock.
    !Unless you are writing a sampler you should not
    !have to use this function - you will be given the 
    !datablock you need.
    function make_datablock()
        integer(kind=cosmosis_block) :: make_datablock
        make_datablock =  make_c_datablock()
    end function make_datablock

end module cosmosis_modules

