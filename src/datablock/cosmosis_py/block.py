import ctypes as ct
from . import lib
from . import errors
from . import types
from .errors import BlockError
import numpy as np

option_section = "module_options"


class DataBlock(object):
	GET=0
	PUT=1
	REPLACE=2
	def __init__(self, ptr=None, own=None):
		self.owns=own
		if ptr is None:
			ptr = lib.make_c_datablock()
			self.owns=True
		if own is not None:
			self.owns=own
		self._ptr = ptr
	#TODO: add destructor.  destroy block if owned

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
	def python_to_1d_c_array(value, numpy_type):
		value = np.array(value, dtype=numpy_type)
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



	def get_int(self, section, name, default=None):
		r = ct.c_int()
		if default is None:
			status = lib.c_datablock_get_int(self._ptr,section,name,r)
		else:
			status = lib.c_datablock_get_int_default(self._ptr,section,name,default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.value

	def get_bool(self, section, name, default=None):
		r = ct.c_bool()
		if default is None:
			status = lib.c_datablock_get_bool(self._ptr,section,name,r)
		else:
			status = lib.c_datablock_get_bool_default(self._ptr,section,name,default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.value

	def get_double(self, section, name, default=None):
		r = ct.c_double()
		if default is None:
			status = lib.c_datablock_get_double(self._ptr,section,name,r)
		else:
			status = lib.c_datablock_get_double_default(self._ptr,section,name,default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.value

	def get_complex(self, section, name, default=None):
		r = lib.c_complex()
		if default is None:
			status = lib.c_datablock_get_complex(self._ptr,section,name,r)
		else:
			status = lib.c_datablock_get_complex_default(self._ptr,section,name,default,default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r.real+1j*r.imag

	def get_string(self, section, name, default=None):
		r = lib.c_str()
		if default is None:
			status = lib.c_datablock_get_string(self._ptr,section,name,r)
		else:
			status = lib.c_datablock_get_string_default(self._ptr,section,name,default,r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return str(r.value)

	def get_int_array_1d(self, section, name):
		n = lib.c_datablock_get_array_length(self._ptr, section, name)
		r = np.zeros(n, dtype=np.intc)
		arr = np.ctypeslib.as_ctypes(r)
		sz = lib.c_int()
		status = lib.c_datablock_get_int_array_1d_preallocated(self._ptr, section, name, arr, ct.byref(sz), n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r

	def get_double_array_1d(self, section, name):
		n = lib.c_datablock_get_array_length(self._ptr, section, name)
		r = np.zeros(n, dtype=np.double)
		arr = np.ctypeslib.as_ctypes(r)
		sz = lib.c_int()
		status = lib.c_datablock_get_double_array_1d_preallocated(self._ptr, section, name, arr, ct.byref(sz), n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r

	def put_int(self, section, name, value):
		status = lib.c_datablock_put_int(self._ptr,section,name,int(value))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_bool(self, section, name, value):
		status = lib.c_datablock_put_bool(self._ptr,section,name,bool(value))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_double(self, section, name, value):
		status = lib.c_datablock_put_double(self._ptr,section,name,float(value))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_complex(self, section, name, value):
		value=self.python_to_c_complex(value)
		status = lib.c_datablock_put_complex(self._ptr,section,name,value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_string(self, section, name, value):
		status = lib.c_datablock_put_string(self._ptr,section,name,str(value))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_int_array_1d(self, section, name, value):
		value_ref, value,n=self.python_to_1d_c_array(value, np.intc)
		status = lib.c_datablock_put_int_array_1d(self._ptr, section, name, value, n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_double_array_1d(self, section, name, value):
		value_ref, value,n=self.python_to_1d_c_array(value, np.double)
		status = lib.c_datablock_put_double_array_1d(self._ptr, section, name, value, n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def _method_for_type(self, T, method_type):
		method={ int:    (self.get_int,     self.put_int,     self.replace_int),
		         float:  (self.get_double,  self.put_double,  self.replace_double),
		         bool:   (self.get_bool,    self.put_bool,    self.replace_bool),
		         complex:(self.get_complex, self.put_complex, self.replace_complex),
		         str:    (self.get_string,  self.put_string,  self.replace_string)
		         }.get(T)
		if method:
			return method[method_type]
		return None

	def _method_for_datatype_code(self, code, method_type):
		T={ 
			types.DBT_INT:     (self.get_int,     self.put_int,     self.replace_int),
			types.DBT_BOOL:    (self.get_bool,     self.put_bool,     self.replace_bool),
			types.DBT_DOUBLE:         (self.get_double,  self.put_double,  self.replace_double),
			types.DBT_COMPLEX: (self.get_complex, self.put_complex, self.replace_complex),
			types.DBT_STRING:  (self.get_string,  self.put_string,  self.replace_string),
			types.DBT_INT1D:   (self.get_int_array_1d,     self.put_int_array_1d,     self.replace_int_array_1d),
			types.DBT_DOUBLE1D:(self.get_double_array_1d,  self.put_double_array_1d,  self.replace_double_array_1d),
			# types.COMPLEX1D:   (self.get_complex_array_1d, self.put_complex_array_1d, self.replace_complex_array_1d),
			# types.STRING1D:    (self.get_string_array_1d,  self.put_string_array_1d,  self.replace_string_array_1d)
			# types.DBT_INT2D:   (self.get_int_array_2d,     self.put_int_array_2d,     self.replace_int_array_2d)
			# types.DBT_DOUBLE2D:(self.get_double_array_2d,  self.put_double_array_2d,  self.replace_double_array_2d)
			# types.COMPLEX2D:   (self.get_complex_array_2d, self.put_complex_array_2d, self.replace_complex_array_2d)
			# types.STRING2D:    (self.get_string_array_2d,  self.put_string_array_2d,  self.replace_string_array_2d)
		         }.get(code)
		if T is not None:
			return T[method_type]
		return None


	def _method_for_value(self, value, method_type):
		T = type(value)
		method = self._method_for_type(T, method_type)
		if method: 
			return method
		if hasattr(value,'__len__'):
			array = np.array(value)
			method = {
				(1,'i'):(self.get_int_array_1d,self.put_int_array_1d,self.replace_int_array_1d),
				#These are not implemented yet
				# (2,'i'):(self.get_int_array_2d,self.put_int_array_1d,self.replace_int_array_1d),
				(1,'f'):(self.get_double_array_1d,self.put_double_array_1d,self.replace_double_array_1d),
				# (2,'f'):(self.get_double_array_2d,self.put_double_array_1d,self.replace_double_array_1d),
				# (1,'c'):(self.get_complex_array_1d,self.put_complex_array_1d,self.replace_complex_array_1d),
				# (2,'c'):(self.get_complex_array_2d,self.put_complex_array_1d,self.replace_complex_array_1d),
			}.get((array.ndim,array.dtype.kind))
			if method:
				return method[method_type]
		raise ValueError("I do not know how to handle this type %r %r"%(value,type(value)))
	
	def get(self, section, name):
		type_code_c = lib.c_datatype()
		status = lib.c_datablock_get_type(self._ptr, section, name, ct.byref(type_code_c))
		if status:
			raise BlockError.exception_for_status(status, section, name)
		type_code = type_code_c.value
		method = self._method_for_datatype_code(type_code,self.GET)
		if method:
			return method(section,name)
		raise ValueError("Cosmosis internal error; unknown type of data")

	def put(self, section, name, value):
		method = self._method_for_value(value,self.PUT)
		method(section, name, value)

	def replace(self, section, name, value):
		method = self._method_for_value(value,self.REPLACE)
		method(section, name, value)


	def replace_int(self, section, name, value):
		status = lib.c_datablock_replace_int(self._ptr,section,name,value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_bool(self, section, name, value):
		status = lib.c_datablock_replace_int(self._ptr,section,name,value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_double(self, section, name, value):
		r = ct.c_double()
		status = lib.c_datablock_replace_double(self._ptr,section,name,value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_complex(self, section, name, value):
		value=self.python_to_c_complex(value)
		status = lib.c_datablock_replace_complex(self._ptr,section,name,value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_string(self, section, name, value):
		status = lib.c_datablock_replace_string(self._ptr,section,name,str(value))
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_int_array_1d(self, section, name, value):
		value_ref, value,n=self.python_to_1d_c_array(value, np.intc)
		status = lib.c_datablock_replace_int_array_1d(self._ptr, section, name, value, n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_double_array_1d(self, section, name, value):
		value_ref, value,n=self.python_to_1d_c_array(value, np.double)
		status = lib.c_datablock_replace_double_array_1d(self._ptr, section, name, value, n)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def has_section(self, section):
		status = lib.c_datablock_has_section(self._ptr, section)
		return bool(status)

	def has_section(self, section):
		has = lib.c_datablock_has_section(self._ptr, section)
		return has

	def has_value(self, section, name):
		has = lib.c_datablock_has_value(self._ptr, section, name)
		return has

	def __getitem__(self, section_name):
		try:
			(section,name) = section_name
		except ValueError:
			raise ValueError("You must specify both a section and a name to get or set a block item: b['section','name']")
		return self.get(section, name)

	def __setitem__(self, section_name, value):

		try:
			(section,name) = section_name
		except ValueError:
			raise ValueError("You must specify both a section and a name to get or set a block item: b['section','name']")
		if self.has_value(section, name):
			self.replace(section, name, value)
		else:
			self.put(section, name, value)

	def __contains__(self, section_name):
		if isinstance(section_name, basestring):
			return self.has_section(section_name)
		try:
			(section,name) = section_name
		except ValueError:
			raise ValueError("You must specify both a section and a name to get or set a block item: b['section','name']")
		return self.has_value(section, name)

	def sections(self):
		n = lib.c_datablock_num_sections(self._ptr)
		return [lib.c_datablock_get_section_name(self._ptr, i) for i in xrange(n)]


	def keys(self, section=None):
		if section is None:
			sections = self.sections()
		else:
			sections = [section]
		keys = []
		for section in sections:
			n_value = lib.c_datablock_num_values(self._ptr, section)
			for i in xrange(n_value):
				name = lib.c_datablock_get_value_name(self._ptr, section, i)
				keys.append((section,name))
		return keys


	def _delete_section(self, section):
		"Internal use only!"
		status = lib.c_datablock_delete_section(self._ptr, section)
		if status!=0:
			raise BlockError.exception_for_status(status, section, "")


