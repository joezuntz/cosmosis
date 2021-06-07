from cosmosis import DataBlock
from cosmosis.datablock.cosmosis_py.errors import BlockSectionNotFound
import numpy as np

def setup(options):
	return 1

def execute(block, config):
	x = np.array([-1.,-2.,-3.,-4.,-5.,-6.,-7.,-8.,-9.,-10.])
	y = np.array([-2.,-4.,-8.,-16.,-32.])
	nx = len(x)
	ny = len(y)
	z = np.zeros([nx,ny])
	for i in range(nx):
		for j in range(ny):
			z[i,j] = 10*i+j
	block.put_grid("py_put", "x", x, "y", y, "z", z)

	try:
		x_c, y_c, z_c = block.get_grid("c_put", "x", "y", "z")
		assert (x==x_c).all()
		assert (y==y_c).all()
		assert (z==z_c).all()
	except BlockSectionNotFound:
		print("No C in py")

	try:
		x_f90, y_f90, z_f90 = block.get_grid("f90_put", "x", "y", "z")
		assert (x==x_f90).all()
		assert (y==y_f90).all()
		assert (z==z_f90).all()
	except BlockSectionNotFound as err:
		print("No f90 in py ", err)
	return 0

def cleanup(config):
	pass