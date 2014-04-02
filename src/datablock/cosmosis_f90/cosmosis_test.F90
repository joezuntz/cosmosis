!This is just a scratch space for testing
!not a proper test suite
program test_cosmosis
    use cosmosis_modules
    implicit none
    integer(cosmosis_block) :: block
    integer(cosmosis_status) :: status
    integer n, size_recovered, i, j
    real(8) :: x
    integer(c_int), dimension(10) :: int_arr, slice
    integer, dimension(:), allocatable :: int_arr_recovered
    integer, dimension(10,10) :: int_arr_2d
    complex(c_double_complex) :: z
    character(len=20) :: s

    n=15
    x=14.8
    z=dcmplx(1.4,-1.2)
    block = make_datablock()
    status = 0
    status = datablock_put_int(block, "MAGIC", "NUMBER", n)
    status = status + datablock_put_double(block, "MAGIC", "WEIGHT", x)
    status = status + datablock_put_complex(block, "MAGIC", "FOURIER", z)
    call cosmosis_assert(status==0, "Put scalar failed")

    n=666
    x=666.
    z=dcmplx(666.,666.)

    status = datablock_get_int(block, "MAGIC", "NUMBER", n)
    status = status + datablock_get_double(block, "MAGIC", "WEIGHT", x)
    status = status + datablock_get_complex(block, "MAGIC", "FOURIER", z)

    call cosmosis_assert(status==0, "Get scalar failed")
    call cosmosis_assert(n==15, "Get int wrong answer")
    call cosmosis_assert(x==14.8, "Get float wrong answer")
    call cosmosis_assert(z==dcmplx(1.4,-1.2), "Get int wrong answer")

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

    call cosmosis_assert(trim(datablock_get_section_name(block,0))=="MAGIC", "Wrong section name")

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

    s = "cat"
    status = datablock_put_string(block, "STRINGS", "ANIMALS", s)
    s = ""
    call cosmosis_assert(status==0, "Put string failed")
    status = datablock_get_string(block, "STRINGS", "ANIMALS", s)
    call cosmosis_assert(s=="cat", "Put string failed")


    contains 

    subroutine cosmosis_assert(predicate, message)
        logical :: predicate
        character(*) :: message

        if (.not. predicate) then
            write(*,*) "Test Failure: ", trim(message) 
            stop
        endif
    end subroutine


end program test_cosmosis