!This is just a scratch space for testing
!not a proper test suite
program test_cosmosis
	use cosmosis_modules
	implicit none
	integer(cosmosis_block) :: block
	integer(cosmosis_status) :: status
	integer n, size_recovered, i
	real(8) :: x
	integer(c_int), dimension(10) :: int_arr
	integer, dimension(:), allocatable :: int_arr_recovered
	complex(c_double_complex) :: z

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

	deallocate(int_arr_recovered)

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