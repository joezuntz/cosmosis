import ctypes as ct
from . import lib
from .errors import BlockError
import numpy as np

class Block(object):
	def __init__(self, ptr=None):
		self.owns=False
		if ptr is None:
			ptr = lib.make_c_datablock()
			self.owns=True
		self._ptr = ptr
	#TODO: add destructor

	@staticmethod
	def python_to_c_complex(value):
		if isinstance(value, lib.c_complex):
			return value
		elif isinstance(value, complex):
			return lib.c_complex(value.real,value.imag)
		elif isinstance(value, tuple):
			return lib.c_complex(value[0],value[1])
		else:
			return lib.c_complex(value, 0.0)

	@staticmethod
	def python_to_1d_c_array(value, c_type):
		value = np.array(value, dtype=np.intc)
		#This function is for 1D arrays only
		assert value.ndim==1
		#check strides same as itemsize.
		#This may be false if e.g. the object
		#was made by looking at a slice through
		#a matrix.
		if value.itemsize != value.strides[0]:
			#If so we need to copy the data to create
			#a new object with sensible strides
			value = value.copy()
		assert value.itemsize==value.strides[0]
		#Now return pointer to start of the data
		array = np.ctypeslib.as_ctypes(value)
		array_size = value.size
		#OK, here's the difficult part.
		# We have to return the value, as well as the
		# array we have converted it to.
		# That's because the array object
		# does not maintain a pointer to the value,
		# so it is garbage collected if we don't
		# and then the array just contains junk memory
		return value, array, array_size



	def get_int(self, section, name):
		r = ct.c_int()
		status = lib.c_datablock_get_int(self._ptr,section,name,r)
		if status!=0:
			raise BlockError(status, section, name)
		return r.value

	def get_double(self, section, name):
		r = ct.c_double()
		status = lib.c_datablock_get_double(self._ptr,section,name,r)
		if status!=0:
			raise BlockError(status, section, name)
		return r.value

	def get_complex(self, section, name):
		r = lib.c_complex()
		status = lib.c_datablock_get_complex(self._ptr,section,name,r)
		if status!=0:
			raise BlockError(status, section, name)
		return r.real+1j*r.imag

	def get_int_array_1d(self, section, name):
		n = lib.c_datablock_get_array_length(self._ptr, section, name)
		r = np.zeros(n, dtype=np.intc)
		arr = np.ctypeslib.as_ctypes(r)
		sz = lib.c_int()
		status = lib.c_datablock_get_int_array_1d_preallocated(self._ptr, section, name, arr, ct.byref(sz), n)
		if status!=0:
			raise BlockError(status, section, name)
		return r

	def put_int(self, section, name, value):
		status = lib.c_datablock_put_int(self._ptr,section,name,value)
		if status!=0:
			raise BlockError(status, section, name)

	def put_double(self, section, name, value):
		status = lib.c_datablock_put_double(self._ptr,section,name,value)
		if status!=0:
			raise BlockError(status, section, name)

	def put_complex(self, section, name, value):
		value=self.python_to_c_complex(value)
		status = lib.c_datablock_put_complex(self._ptr,section,name,value)
		if status!=0:
			raise BlockError(status, section, name)

	def put_int_array_1d(self, section, name, value):
		value_ref, value,n=self.python_to_1d_c_array(value, ct.c_int)
		status = lib.c_datablock_put_int_array_1d(self._ptr, section, name, value, n)
		if status!=0:
			raise BlockError(status, section, name)


	def replace_int(self, section, name, value):
		status = lib.c_datablock_replace_int(self._ptr,section,name,value)
		if status!=0:
			raise BlockError(status, section, name)

	def replace_double(self, section, name, value):
		r = ct.c_double()
		status = lib.c_datablock_replace_double(self._ptr,section,name,value)
		if status!=0:
			raise BlockError(status, section, name)

	def replace_complex(self, section, name, value):
		value=self.python_to_c_complex(value)
		status = lib.c_datablock_replace_complex(self._ptr,section,name,value)
		if status!=0:
			raise BlockError(status, section, name)

	def replace_int_array_1d(self, section, name, value):
		value_ref, value,n=self.python_to_1d_c_array(value, ct.c_int)
		status = lib.c_datablock_replace_int_array_1d(self._ptr, section, name, value, n)
		if status!=0:
			raise BlockError(status, section, name)
