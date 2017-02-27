import ctypes as ct
from . import lib
from . import errors
from . import dbt_types as types
from .errors import BlockError
import numpy as np
import os
import collections
import tarfile
import StringIO

option_section = "module_options"
metadata_prefix = "cosmosis_metadata:"

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

	def __del__(self):
		try:
			if self.owns:
				lib.destroy_c_datablock(self._ptr)
		except:
			pass
				
	def clone(self):
		ptr = lib.clone_c_datablock(self._ptr)
		return DataBlock(ptr,own=True)


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

	def _get_array_nd(self, section, name, dtype):

		if dtype is complex or dtype is str:
			raise ValueError("Sorry - cosmosis support for 2D complex and string values is incomplete")

		ndim = lib.c_int()
		status = lib.c_datablock_get_array_ndim(self._ptr, section, name, ct.byref(ndim))
		if status:
			raise BlockError.exception_for_status(status, section, name)

		ctype, shape_function, get_function = {
			int: (ct.c_int, lib.c_datablock_get_int_array_shape, lib.c_datablock_get_int_array),
			float: (ct.c_double, lib.c_datablock_get_double_array_shape, lib.c_datablock_get_double_array),
			#complex: (lib.c_complex, lib.c_datablock_get_complex_array_shape, lib.c_datablock_get_complex_array),
			#str: (lib.c_str, lib.c_datablock_get_string_array_shape, lib.c_datablock_get_string_array),
		}[dtype]

		#Get the array extent
		extent = (ct.c_int * ndim.value)()
		status = shape_function(self._ptr, section, name, ndim, extent)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

		#Make the space for it
		N = tuple([extent[i] for i in xrange(ndim.value)])
		r = np.zeros(N, dtype=ctype)
		arr = r.ctypes.data_as(ct.POINTER(ctype))

		#Fill in with the data
		status = get_function(self._ptr, section, name, arr, ndim, extent)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return r

	def _put_replace_array_nd(self, section, name, value, dtype, mode):
		shape = value.shape
		ndim = len(shape)
		extent = (ct.c_int * ndim)()
		for i in xrange(ndim): extent[i] = shape[i]
		value = value.flatten()
		p, arr, arr_size = self.python_to_1d_c_array(value, dtype)
		put_function={
			(np.intc, self.PUT):lib.c_datablock_put_int_array,
			(np.double, self.PUT):lib.c_datablock_put_double_array,
			(np.intc, self.REPLACE):lib.c_datablock_replace_int_array,
			(np.double, self.REPLACE):lib.c_datablock_replace_double_array,
		}.get((dtype,mode))
		if put_function is None:
			raise ValueError("I do not know how to save %s in %s of type %s"% (section,name, dtype))
		status = put_function(self._ptr, section, name, arr, ndim, extent)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)


	def put_double_array_nd(self, section, name, value):
		self._put_replace_array_nd(section, name, value, np.double, self.PUT)

	def put_int_array_nd(self, section, name, value):
		self._put_replace_array_nd(section, name, value, np.intc, self.PUT)

	def replace_double_array_nd(self, section, name, value):
		self._put_replace_array_nd(section, name, value, np.double, self.REPLACE)

	def replace_int_array_nd(self, section, name, value):
		self._put_replace_array_nd(section, name, value, np.intc, self.REPLACE)

	def get_double_array_nd(self, section, name):
		return self._get_array_nd(section, name, float)

	def get_int_array_nd(self, section, name):
		return self._get_array_nd(section, name, int)

	#def get_complex_array_2d(self, section, name):
	#	return self._get_array_2d(section, name, complex)

	#def get_string_array_2d(self, section, name):
	#	return self._get_array_2d(section, name, str)

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
			types.DBT_DOUBLEND:(self.get_double_array_nd,  self.put_double_array_nd, None),
			types.DBT_INTND:(self.get_int_array_nd,  self.put_int_array_nd, None),
			# types.COMPLEX2D:   (self.get_complex_array_2d, self.put_complex_array_2d, self.replace_complex_array_2d)
			# types.STRING2D:    (self.get_string_array_2d,  self.put_string_array_2d,  self.replace_string_array_2d)
				 }.get(code)
		if T is not None:
			return T[method_type]
		return None


	def _method_for_value(self, value, method_type):
		if isinstance(value, np.float32) or isinstance(value, np.float64):
			value = float(value)
		if isinstance(value, np.int32) or isinstance(value, np.int64):
			value = int(value)

		T = type(value)

		method = self._method_for_type(T, method_type)
		if method: 
			return method

		if hasattr(value,'__len__'):
			#let numpy work out what type this should be.
			array = np.array(value)
			kind = array.dtype.kind
			ndim = array.ndim
			if ndim==1:
				#1D arrays have their own specific methods,
				#for integer and float
				if kind=='i':
					method = (self.get_int_array_1d,self.put_int_array_1d,self.replace_int_array_1d)
				elif kind=='f':
					method = (self.get_double_array_1d,self.put_double_array_1d,self.replace_double_array_1d)
			#otherwise we just use the generic n-d arrays
			elif kind=='i':
				method = (self.get_int_array_nd,self.put_int_array_nd,self.replace_int_array_nd)
			elif kind=='f':
				method = (self.get_double_array_nd,self.put_double_array_nd,self.replace_double_array_nd)
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

	def put(self, section, name, value, **meta):
		method = self._method_for_value(value,self.PUT)
		method(section, name, value)
		for (key, val) in meta.items():
			self.put_metadata(section, name, str(key), str(val))

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
		has = lib.c_datablock_has_section(self._ptr, section)
		return bool(has)

	def has_value(self, section, name):
		has = lib.c_datablock_has_value(self._ptr, section, name)
		return bool(has)

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
			raise BlockError.exception_for_status(status, section, "<tried to delete>")

	def _copy_section(self, source, dest):
		"Internal use only!"
		status = lib.c_datablock_copy_section(self._ptr, source, dest)
		if status!=0:
			raise BlockError.exception_for_status(status, dest, "<tried to copy>")


	@staticmethod
	def _parse_metadata_key(key):
		key = key[len(metadata_prefix):].strip(":")
		s = key.index(":")
		if s==-1:
			raise ValueError("Could not understand metadata")
		name = key[:s]
		meta = key[s+1:]
		return name, meta

	def save_to_file(self, dirname, clobber=False):
		filename = dirname + ".tgz"

		base_dirname,base_filename=os.path.split(filename)
		if base_dirname:
			try:
				os.mkdir(base_dirname)
			except OSError:
				pass

		if os.path.exists(filename) and not clobber:
			raise ValueError("File %s already exists and not clobbering"%filename)

		tar = tarfile.open(filename, "w:gz")

		for (section, scalar_outputs, vector_outputs, meta) in self._save_paths():
			#Save all the vector outputs as individual files
			for name, value in vector_outputs:
				vector_outfile = os.path.join(dirname,section,name+'.txt')
				header = "%s\n"%name
				if value.ndim>2:
					header += "shape = %s\n"%str(value.shape)
					print "Flattening %s--%s when saving; shape info in header" % (section,name)
					value = value.flatten()
				if name in meta:
					for key,val in meta[name].items():
						header+='%s = %s\n' % (key,val)

				#Save this file into the tar file
				string_output = StringIO.StringIO()
				np.savetxt(string_output, value, header=header.rstrip("\n"))
				string_output.seek(0)
				info = tarfile.TarInfo(name=vector_outfile)
				info.size=len(string_output.buf)
				tar.addfile(tarinfo=info, fileobj=string_output)

			#Save all the scalar outputs together as a single file
			#inside the tar file
			if scalar_outputs:
				scalar_outfile = os.path.join(dirname,section,"values.txt")
				string_output = StringIO.StringIO()
				for s in scalar_outputs:
					string_output.write("%s = %r\n"%s)
					if s[0] in meta:
						for key,val in meta[s[0]].items():
							string_output.write("#%s %s = %s\n"%(s[0],key,val))
				string_output.seek(0)
				info = tarfile.TarInfo(name=scalar_outfile)
				info.size=len(string_output.buf)
				tar.addfile(tarinfo=info, fileobj=string_output)
		tar.close()


	def save_to_directory(self, dirname, clobber=False):
		try:
			os.mkdir(dirname)
		except OSError:
			if not clobber:
				print "Not clobbering", clobber
				raise
		for (section, scalar_outputs, vector_outputs, meta) in self._save_paths():
			
			#Create the sub-directory for this 
			#section
			try:
				os.mkdir(os.path.join(dirname,section))
			except OSError:
				if not clobber:
					raise

			#Save all the vector outputs as individual files
			for name, value in vector_outputs:
				vector_outfile = os.path.join(dirname,section,name+'.txt')
				header = "%s\n"%name
				if value.ndim>2:
					header += "shape = %s\n"%str(value.shape)
					print "Flattening %s--%s when saving; shape info in header" % (section,name)
					value = value.flatten()
				if name in meta:
					for key,val in meta[name].items():
						header+='%s = %s\n' % (key,val)
				np.savetxt(vector_outfile, value, header=header.rstrip("\n"))

			#Save all the scalar outputs together as a single file
			if scalar_outputs:
				f=open(os.path.join(dirname,section,"values.txt"), 'w')
				for s in scalar_outputs:
					f.write("%s = %r\n"%s)
					if s[0] in meta:
						for key,val in meta[s[0]].items():
							f.write("#%s %s = %s\n"%(s[0],key,val))
				f.close()

	def _save_paths(self):
		keys = self.keys()
		sections = set(k[0] for k in keys)
		for section in sections:
			scalar_outputs = []
			meta = collections.defaultdict(dict)
			vector_outputs = []
			for k in keys:
				sec, name = k
				if sec!=section: continue
				if name.startswith(metadata_prefix):
					target, metakey = self._parse_metadata_key(name)
					meta[target][metakey] = self[section,name]
					continue
				value = self[section,name]
				if np.isscalar(value):
					scalar_outputs.append((name,value))
				else:
					vector_outputs.append((name,value))
			yield section, scalar_outputs, vector_outputs, meta


	def report_failures(self):
		status = lib.c_datablock_report_failures(self._ptr)
		if status!=0:
			raise BlockError.exception_for_status(status, "", "")

	def print_log(self):
		status = lib.c_datablock_print_log(self._ptr)
		if status!=0:
			raise BlockError.exception_for_status(status, "", "")

	def get_log_count(self):
		return lib.c_datablock_get_log_count(self._ptr)

	def get_log_entry(self, i):
		smax = 128
		ptype = ct.create_string_buffer(smax)
		section = ct.create_string_buffer(smax)
		name = ct.create_string_buffer(smax)
		dtype = ct.create_string_buffer(smax)
		status = lib.c_datablock_get_log_entry(self._ptr, i, smax, ptype, section, name, dtype)
		if status:
			raise ValueError("Asked for log entry above maximum or less than zero")
		return ptype.value, section.value, name.value, dtype.value

	def log_access(self, log_type, section, name):
		status = lib.c_datablock_log_access(self._ptr, log_type, section, name)
		if status!=0:
			raise BlockError.exception_for_status(status, "", "")

	def get_metadata(self, section, name, key):
		r = lib.c_str()
		status = lib.c_datablock_get_metadata(self._ptr,section,name,key, r)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)
		return str(r.value)

	def put_metadata(self, section, name, key, value):
		status = lib.c_datablock_put_metadata(self._ptr,section,name,key, value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def replace_metadata(self, section, name, key, value):
		status = lib.c_datablock_replace_metadata(self._ptr,section,name,key, value)
		if status!=0:
			raise BlockError.exception_for_status(status, section, name)

	def put_grid(self, section, name_x, x, name_y, y, name_z, z):
		self._grid_put_replace(section, name_x, x, name_y, y, name_z, z, False)

	def get_grid(self, section, name_x, name_y, name_z):
		name_x = name_x.lower()
		name_y = name_y.lower()
		name_z = name_z.lower()
		x = self[section, name_x]
		y = self[section, name_y]
		z = self[section, name_z]
		sentinel_key = "_cosmosis_order_%s"%name_z
		sentinel_value = self[section, sentinel_key].lower()

		if sentinel_value== "%s_cosmosis_order_%s" % (name_x, name_y):
			assert z.shape==(x.size, y.size)
			return x, y, z
		elif sentinel_value== "%s_cosmosis_order_%s" % (name_y, name_x):
			assert z.T.shape==(x.size, y.size)
			return x, y, z.T
		else:
			raise BlockError.exception_for_status(errors.DBS_WRONG_VALUE_TYPE, section, name_z)



	def replace_grid(self, section, name_x, x, name_y, y, name_z, z):
		self._grid_put_replace(section, name_x, x, name_y, y, name_z, z, True)

	def _grid_put_replace(self, section, name_x, x, name_y, y, name_z, z, replace):
		# These conversions do not create new objects if x,y,z are already arrays.
		x = np.asarray(x)
		y = np.asarray(y)
		z = np.asarray(z)

		if x.ndim!=1 or y.ndim!=1 or z.ndim!=2 or z.shape!=(x.size,y.size):
			msg = """
	Your code tried to save or replace a grid {name_z}[{name_x}, {name_y}] in section {}. 
	This requires 1D {name_x}, 1D {name_y}, 2D {name_z} and the shape of {name_z} to be (len({name_x}),len({name_y})).
	Whereas your code tried:
	{name_x} ndim = {}    [{}]
	{name_y} ndim = {}    [{}]
	{name_z} ndim = {}    [{}]
	{name_x} shape = {}
	{name_y} shape = {}
	{name_z} shape = {}   [{}]
			""".format(section, 
				x.ndim, "OK" if x.ndim==1 else "WRONG",
				y.ndim, "OK" if y.ndim==1 else "WRONG",
				z.ndim, "OK" if z.ndim==2 else "WRONG",
				x.shape,
				y.shape,
				z.shape, "OK" if z.shape==(x.size, y.size) else "WRONG",
				name_z=name_z, name_x=name_x, name_y=name_y
				)
			raise ValueError(msg)

		self[section, name_x] = x
		self[section, name_y] = y
		self[section, name_z] = z

		sentinel_key = "_cosmosis_order_%s"%name_z
		sentinel_value = "%s_cosmosis_order_%s" % (name_x, name_y)
		self[section, sentinel_key] = sentinel_value.lower()



class SectionOptions(object):
	"""
	The SectionOptions object wraps is a handy short-cut to let you
	look up objects in a DataBlock object, but looking specifically at
	the special section in an ini file that refers to "the section that 
	defines the current module"

	"""
	def __init__(self, block):
		self.block=block

	def has_value(self, name):
		has = self.block.has_value(option_section, name)
		return bool(has)



def _make_getter(cls, name):
	if name=='__getitem__':
		def getter(self, key):
			return self.block[option_section, key]
	elif "array" in name:
		def getter(self, key):
			return getattr(self.block, name)(option_section, key)
	else:
		def getter(self, key, default=None):
			return getattr(self.block, name)(option_section, key, default=default)

	return getter




for name in dir(DataBlock):
	if name.startswith('get') or name=='__getitem__':
		setattr(SectionOptions, name, _make_getter(SectionOptions, name))

