!This is just a scratch space for testing
!not a proper test suite
program test_cosmosis
    use cosmosis_modules
    implicit none


    call test_scalar()
    call test_defaults()
    call test_array()
    call test_double_array()
    call test_2d()


    contains 

    subroutine cosmosis_assert(predicate, message)
        logical :: predicate
        character(*) :: message

        if (.not. predicate) then
            write(*,*) "Test Failure: ", trim(message) 
            stop 1
        endif
    end subroutine



    subroutine test_template()
        integer(cosmosis_block) :: block
        integer(cosmosis_status) :: status
        ! ...
        block = make_datablock()

        !.. tests here

        status = destroy_c_datablock(block)
        call cosmosis_assert(status==0, "Destroy failed")
    end subroutine test_template




    subroutine test_defaults()
        integer(cosmosis_block) :: block
        integer(cosmosis_status) :: status
        integer n
        real(8) :: x
        complex(c_double_complex) :: z
        block = make_datablock()

        status = datablock_get_int_default(block, "fish", "n", 14, n)
        status = status + datablock_get_double_default(block, "fish", "x", 7.5_8, x)
        status = status + datablock_get_complex_default(block, "fish", "z", dcmplx(0,-2), z)
        call cosmosis_assert(status==0, "Default get failed")
        call cosmosis_assert(n==14, "Default get int wrong answer")
        call cosmosis_assert(x==7.5, "Default get double wrong answer")
        call cosmosis_assert(z==dcmplx(0,-2), "Default get complex wrong answer")

        !These tests now lead to failure since get with default leads to
        !implicit put
        !status = datablock_put_int(block, "fish", "n", 3)
        !status = status + datablock_put_double(block, "fish", "x", 4.0_8)
        !status = status + datablock_put_complex(block, "fish", "z", dcmplx(1,-6))
        !call cosmosis_assert(status==0, "Default puts failed")
        
        !status = datablock_get_int_default(block, "fish", "n", 14, n)
        !status = status + datablock_get_double_default(block, "fish", "x", 7.5_8, x)
        !status = status + datablock_get_complex_default(block, "fish", "z", dcmplx(0,-2), z)
        !call cosmosis_assert(status==0, "Default get 2 failed")
        !call cosmosis_assert(n==3, "Default get int 2 wrong answer")
        !call cosmosis_assert(x==4.0, "Default get double 2 wrong answer")
        !call cosmosis_assert(z==dcmplx(1,-6), "Default get complex 2 wrong answer")

        status = destroy_c_datablock(block)
        call cosmosis_assert(status==0, "Destroy failed")
    end subroutine test_defaults


    subroutine test_scalar()
        integer(cosmosis_block) :: block
        integer(cosmosis_status) :: status
        integer n
        real(8) :: x
        complex(c_double_complex) :: z
        character(len=20) :: s
        logical :: b

        block = make_datablock()


        n=15
        x=14.8
        z=dcmplx(1.4,-1.2)
        b = .true.
        status = 0
        status = datablock_put_int(block, "MAGIC", "NUMBER", n)
        status = status + datablock_put_double(block, "MAGIC", "WEIGHT", x)
        status = status + datablock_put_complex(block, "MAGIC", "FOURIER", z)
        status = status + datablock_put_logical(block, "MAGIC", "LOGIC", b)
        call cosmosis_assert(status==0, "Put scalar failed")

        n=666
        x=666.
        z=dcmplx(666.,666.)
        b = .false.
        status = datablock_get_int(block, "MAGIC", "NUMBER", n)
        status = status + datablock_get_double(block, "MAGIC", "WEIGHT", x)
        status = status + datablock_get_complex(block, "MAGIC", "FOURIER", z)
        status = status + datablock_get_logical(block, "MAGIC", "LOGIC", b)

        call cosmosis_assert(status==0, "Get scalar failed")
        call cosmosis_assert(n==15, "Get int wrong answer")
        call cosmosis_assert(x==14.8, "Get float wrong answer")
        call cosmosis_assert(z==dcmplx(1.4,-1.2), "Get complex wrong answer")
        call cosmosis_assert(b, "Get bool wrong answer")

        s = "cat"
        status = datablock_put_string(block, "STRINGS", "ANIMALS", s)
        s = ""
        call cosmosis_assert(status==0, "Put string failed")
        status = datablock_get_string(block, "STRINGS", "ANIMALS", s)
        call cosmosis_assert(s=="cat", "Put string failed")


        status = destroy_c_datablock(block)
        call cosmosis_assert(status==0, "Destroy failed")

    end subroutine test_scalar



    subroutine test_array()
        integer(cosmosis_block) :: block
        integer(cosmosis_status) :: status
        integer size_recovered, i, j
        integer(c_int), dimension(10) :: int_arr, slice
        integer, dimension(:), allocatable :: int_arr_recovered
        integer, dimension(10,10) :: int_arr_2d


        block = make_datablock()

        do i=1,10
            int_arr(i) = i*i
        enddo


        status = datablock_put_int_array_1d(block, "MAGIC", "FIELD", int_arr)
        call cosmosis_assert(status==0, "Put int array failed")

        call cosmosis_assert(.not. allocated(int_arr_recovered), "Put int array did not allocate")
        status = datablock_get_int_array_1d(block, "MAGIC", "FIELD", int_arr_recovered, size_recovered)
        call cosmosis_assert(status==0, "Put int array did not work")
        call cosmosis_assert(allocated(int_arr_recovered), "Put int array did not allocate")
        call cosmosis_assert(size(int_arr_recovered)==10, "Wrong size array back")
        call cosmosis_assert(size_recovered==10, "Wrong size array back")
        do i=1,10
            call cosmosis_assert(int_arr_recovered(i)==i*i, "get int array wrong answer")
        enddo

        call cosmosis_assert(datablock_num_sections(block)==1, "wrong section count")
        deallocate(int_arr_recovered)

        call cosmosis_assert(trim(datablock_get_section_name(block,0))=="magic", "Wrong section name")

        do i=1,10
            do j=1,10
                int_arr_2d(i,j) = i+10*j
            enddo
        enddo

        slice = int_arr_2d(1,:)
        status = datablock_put_int_array_1d(block, "MAGIC", "SLICE", int_arr_2d(1,:))
        call cosmosis_assert(status==0, "Put int slice did not work")
        status = datablock_get_int_array_1d(block, "MAGIC", "SLICE", int_arr_recovered, size_recovered)
        call cosmosis_assert(status==0, "Get int slice did not work")
        call cosmosis_assert(all(slice==int_arr_recovered), "Slice dim 1 fail")
        deallocate(int_arr_recovered)


        slice = int_arr_2d(:,1)
        status = datablock_replace_int_array_1d(block, "MAGIC", "SLICE", int_arr_2d(:,1))
        call cosmosis_assert(status==0, "Replace int slice 2 did not work")
        status = datablock_get_int_array_1d(block, "MAGIC", "SLICE", int_arr_recovered, size_recovered)
        call cosmosis_assert(status==0, "Get int slice 2 did not work")
        call cosmosis_assert(all(slice==int_arr_recovered), "Slice dim 2 fail")
        deallocate(int_arr_recovered)


        status = destroy_c_datablock(block)
        call cosmosis_assert(status==0, "Destroy failed")
    end subroutine test_array


    subroutine test_2d()
        integer(cosmosis_block) :: block
        integer(cosmosis_status) :: status, status2
        real(8), dimension(10, 10) :: arr
        real(8), dimension(:,:), allocatable :: recovered

        integer, dimension(10, 10) :: iarr
        integer, dimension(:,:), allocatable :: irecovered
        integer i, j

        do i=1,10
            do j=1,10
                arr(i,j) = i+10*j
                iarr(i,j) = j+10*i + 3
            enddo
        enddo

        block = make_datablock()

        status = datablock_put_double_array_2d(block, "xxx", "yyy", arr)
        call cosmosis_assert(status==0, "put double 2d failed")

        status = datablock_get_double_array_2d(block, "xxx", "yyy", recovered)
        call cosmosis_assert(status==0, "get double 2d failed")
        call cosmosis_assert(all(arr==recovered), "2d double compare failed")

        status = datablock_put_int_array_2d(block, "xxx", "iyyy", iarr)
        call cosmosis_assert(status==0, "put double 2d failed")

        status = datablock_get_int_array_2d(block, "xxx", "iyyy", irecovered)
        call cosmosis_assert(status==0, "get int 2d failed")
        call cosmosis_assert(all(iarr==irecovered), "2d int compare failed")


    end subroutine test_2d

    subroutine test_double_array()
        integer(cosmosis_block) :: block
        integer(cosmosis_status) :: status
        integer size_recovered, i, j
        real(8), dimension(10) :: arr, slice
        real(8), dimension(:), allocatable :: arr_recovered
        real(8), dimension(10,10) :: arr_2d


        block = make_datablock()

        do i=1,10
            arr(i) = i*i*2.0
        enddo


        status = datablock_put_double_array_1d(block, "MAGIC", "FIELD", arr)
        call cosmosis_assert(status==0, "Put double array failed")

        call cosmosis_assert(.not. allocated(arr_recovered), "Put double array did not allocate")
        status = datablock_get_double_array_1d(block, "MAGIC", "FIELD", arr_recovered, size_recovered)
        call cosmosis_assert(status==0, "Put double array did not work")
        call cosmosis_assert(allocated(arr_recovered), "Put double array did not allocate")
        call cosmosis_assert(size(arr_recovered)==10, "Wrong size array back")
        call cosmosis_assert(size_recovered==10, "Wrong size array back")
        do i=1,10
            call cosmosis_assert(arr_recovered(i)==i*i*2.0, "get double array wrong answer")
        enddo

        call cosmosis_assert(datablock_num_sections(block)==1, "wrong section count")
        deallocate(arr_recovered)

        call cosmosis_assert(trim(datablock_get_section_name(block,0))=="magic", "Wrong section name")

        do i=1,10
            do j=1,10
                arr_2d(i,j) = i+0.1*j
            enddo
        enddo

        slice = arr_2d(1,:)
        status = datablock_put_double_array_1d(block, "MAGIC", "SLICE", arr_2d(1,:))
        call cosmosis_assert(status==0, "Put double slice did not work")
        status = datablock_get_double_array_1d(block, "MAGIC", "SLICE", arr_recovered, size_recovered)
        call cosmosis_assert(status==0, "Get double slice did not work")
        call cosmosis_assert(all(slice==arr_recovered), "Slice dim 1 fail")
        deallocate(arr_recovered)


        slice = arr_2d(:,1)
        status = datablock_replace_double_array_1d(block, "MAGIC", "SLICE", arr_2d(:,1))
        call cosmosis_assert(status==0, "Replace double slice 2 did not work")
        status = datablock_get_double_array_1d(block, "MAGIC", "SLICE", arr_recovered, size_recovered)
        call cosmosis_assert(status==0, "Get double slice 2 did not work")
        call cosmosis_assert(all(slice==arr_recovered), "Slice dim 2 fail")
        deallocate(arr_recovered)


        status = destroy_c_datablock(block)
        call cosmosis_assert(status==0, "Destroy failed")
    end subroutine test_double_array


end program test_cosmosis
