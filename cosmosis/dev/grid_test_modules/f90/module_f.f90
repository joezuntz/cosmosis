
function setup(options) result(result)
        USE cosmosis_modules
        implicit none
        integer(cosmosis_block), value :: options
        integer result
        result = 0
end function setup


function execute(block, config) result(status)
	use cosmosis_modules
	integer(cosmosis_block), value :: block
	integer(cosmosis_status) :: status
	type(c_ptr), value :: config
	integer, parameter :: nx = 10, ny=5
	real(8), dimension(nx) :: x
	real(8), dimension(ny) :: y
	real(8), dimension(nx, ny) :: z

	real(8), dimension(:), allocatable :: x_c, y_c, x_py, y_py
	real(8), dimension(:,:), allocatable :: z_c, z_py

	integer i,j;

	x = (/ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.  /)
	y = (/ -2.0, -4.0, -8.0, -16.0, -32.0  /)
	do i=1,nx
		do j=1,ny
			z(i,j) = 10*(i-1)+(j-1)
		enddo
	enddo

	status = datablock_put_double_grid(block, "f90_put", "x", x, "y", y, "z", z)
	if (status .ne. 0) write(*,*) "Failed to put in f90"

	status = datablock_get_double_grid(block, "c_put", "x", x_c, "y", y_c, "z", z_c)
	if (status == 0) then
		if (.not. all(shape(x)==shape(x_c))) write(*,*) "X C SHAPE WRONG!"
		if (.not. all(shape(y)==shape(y_c))) write(*,*) "Y C SHAPE WRONG!"
		if (.not. all(shape(z)==shape(z_c))) write(*,*) "Z C SHAPE WRONG!"
		if (.not. all(x_c==x)) write(*,*) "X C IS WRONG!"
		if (.not. all(y_c==y)) write(*,*) "Y C IS WRONG!"
		if (.not. all(z_c==z)) write(*,*) "Z C IS WRONG!"
	else
		write(*,*) "Failed to get C in f90", status
	endif

	status = datablock_get_double_grid(block, "py_put", "x", x_py, "y", y_py, "z", z_py)
	if (status == 0) then
		if (.not. all(shape(x)==shape(x_py))) write(*,*) "X PY SHAPE WRONG!"
		if (.not. all(shape(y)==shape(y_py))) write(*,*) "Y PY SHAPE WRONG!"
		if (.not. all(shape(z)==shape(z_py))) write(*,*) "Z PY SHAPE WRONG!"
		if (.not. all(x_py==x)) write(*,*) "X PY IS WRONG!"
		if (.not. all(y_py==y)) write(*,*) "Y PY IS WRONG!"
		if (.not. all(z_py==z)) write(*,*) "Z PY IS WRONG!"
	else
		write(*,*) "Failed to get py in f90", status
	endif


	status = 0



end function execute


function cleanup(options) result(result)
        USE cosmosis_modules
        implicit none
        integer(cosmosis_block), value :: options
        integer result
        result = 0
end function cleanup
