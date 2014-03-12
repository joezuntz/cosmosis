!This is just a scratch space for testing
!not a proper test suite
program test_cosmosis
	use cosmosis_modules
	implicit none
	integer(cosmosis_block) :: block
	integer(cosmosis_status) :: status
	integer n
	real(8) :: x
	complex(c_double_complex) :: z

	n=15
	x=14.8
	z=dcmplx(1.4,-1.2)
	block = make_datablock()

	write(*,*) "saved values = ", n, x, z
	status = datablock_put_int(block, "MAGIC", "NUMBER", n)
	status = datablock_put_double(block, "MAGIC", "WEIGHT", x)
	status = datablock_put_complex(block, "MAGIC", "FOURIER", z)
	n=666
	x=666.
	z=dcmplx(666.,666.)

	write(*,*) "status = ", status
	status = datablock_get_int(block, "MAGIC", "NUMBER", n)
	status = datablock_get_double(block, "MAGIC", "WEIGHT", x)
	status = datablock_get_complex(block, "MAGIC", "FOURIER", z)
	write(*,*) "status = ", status
	write(*,*) "recovered values = ", n, x, z



end program test_cosmosis