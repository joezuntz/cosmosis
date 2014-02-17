import collections
import cStringIO
import contextlib
import pyfits
import numpy as np
import warnings
from . import section_names
import desglue

warnings.filterwarnings('ignore', category=pyfits.verify.VerifyWarning)

_FUNDAMENTAL_KEY = "FUNDAMENTAL"
_DUMMY_KEY = "_DUMMY"
_FITS_RESERVED_KEYS = ["SIMPLE","BITPIX","NAXIS","EXTEND","COMMENT","DATE","TFIELDS","PCOUNT","GCOUNT","TTYPE","TFORM","TUNIT"]


ParamsData = collections.namedtuple("ParamsData","params data")


class DesDataPackage(object):
	def __init__(self):
		self.sections = {}
		self.add_section(_FUNDAMENTAL_KEY)
	def add_section(self,name):
		if name not in self.sections:
			self.sections[name] = self._empty_section()
	@staticmethod
	def _empty_section():
		return ParamsData(params={},data={})
	@staticmethod
	def _update_header(header, key, value):
		key = key.upper()
		# if len(key)>8 and not key.startswith("HIERARCH"):
		# 	key = "HIERARCH"+key
		# print "Updating with ", key, value
		header.update(key, value)
	def get_section(self,section):
		if section not in self.sections:
			self.sections[section] = self._empty_section()
		return self.sections[section]
	def set_param(self,section,name,value):
		name=name.upper()
		if name.startswith("HIERARCH"):
			name = name[8:]
		params = self.get_section(section).params
		params[name] = value
	def get_param(self,section,name, default=None):
		name=name.upper()

		try:
			return self.sections[section].params[name]
		except KeyError:
			if not name.startswith("HIERARCH") and len(name)>8:
				return self.get_param(section,"HIERARCH"+name,default=default)
			if default is not None:
				return default
			if section in self.sections:
				raise KeyError("Parameter %s not found in data section %s" % (name,section))
			else:
				raise KeyError("Data section %s not found" % section)

	def set_data(self,section,name,value):
		name=name.upper()
		if section==_FUNDAMENTAL_KEY:
			raise ValueError("The FUNDAMENTAL section is not designed for data, just common parameters.")
		data = self.get_section(section).data
		data[name] = value
	def get_data(self,section,name,default=None):
		name=name.upper()
		if section==_FUNDAMENTAL_KEY:
			raise ValueError("The FUNDAMENTAL section is not designed for data, just common parameters.")
		try:
			return self.sections[section].data[name]
		except KeyError:
			if default is not None:
				return default
			if section in self.sections:
				raise KeyError("Data column %s not found in data section %s" % (name,section))
			else:
				raise KeyError("Data section %s not found" % section)
		
	@staticmethod
	def get_fits_type(array):
		dtype=array.dtype
		code='%s%d' % (dtype.kind,dtype.base.itemsize)
		return pyfits.column.NUMPY2FITS[code]
	
	@classmethod
	def from_fits_handle(cls,handle):
		string = desglue.read_fits(handle).read()
		return cls.from_fits_string(string)
	
	@classmethod
	def from_fits_string(cls, string):
		data = cls()
		sio = cStringIO.StringIO(string)
		hdulist = pyfits.open(sio, mode='readonly')
		primary = hdulist[0]
		header = primary.header
		for key in header:
			if not _matches_fits_reserved(key):
				data.set_param(_FUNDAMENTAL_KEY,key,header[key])
		for extension in hdulist[1:]:
			data.add_section(extension.name)
			header = extension.header
			for key in header:
				data.set_param(extension.name,key,header[key])
			for i,column in enumerate(extension.data.columns):
				if column.name != _DUMMY_KEY:
					data.set_data(extension.name, column.name, extension.data.field(i).copy())
		return data
	
	@classmethod
	def from_cosmo_params(cls, params):
		data = cls()
		for name,value in params.items():
			data.set_param(section_names.cosmological_parameters,name,value)
		return data
	def add_mixed_params(self, dict_of_dicts):
		for (section_name, parameters) in dict_of_dicts.items():
			if hasattr(section_names, section_name):
				section_name = getattr(section_names, section_name)
			for (parameter_name, value) in parameters.items():
				self.set_param(section_name, parameter_name, value)
	@classmethod
	def from_mixed_params(cls, dict_of_dicts):
		data = cls()
		data.add_mixed_params(dict_of_dicts)
		return data
	@classmethod
	def handle_from_cosmo_params(cls, params):
		return cls.from_cosmo_params(params).to_new_fits_handle()
		
	
	def write_to_fits_handle(self,handle):
		string = self.to_fits_string()
		desglue.rewrite_fits(handle,string)
				
	def to_new_fits_handle(self):
		string = self.to_fits_string()
		return desglue.create_fits(string)
		
	def save_to_file(self,filename):
		s = self.to_fits_string()
		open(filename,'w').write(s)
	
	@classmethod
	def from_file(cls, filename):
		s = open(filename).read()
		return cls.from_fits_string(s)

	def extract_likelihoods(self, names):
		like = 0.0
		like_section = section_names.likelihoods
		for likelihood_name in names:
			like += self.get_param(like_section,likelihood_name+"_LIKE")
		return like

		

	def to_fits_string(self):
		#Primary HDU holds fundamental parameters (not derived)
		hdulist = pyfits.HDUList()
		primary=pyfits.PrimaryHDU()
		hdulist.append(primary)
		for key,value in self.sections[_FUNDAMENTAL_KEY].params.iteritems():
			self._update_header(primary.header, key, value)
		
		#Loop through additional sections creating extensions for them
		for sectionName in self.sections:
			#Ignore the primary
			if sectionName==_FUNDAMENTAL_KEY: continue
			section = self.sections[sectionName]
			header = pyfits.Header()
			self._update_header(header, "EXTNAME", sectionName)
			#Set parameters in header
			for key,value in section.params.iteritems():
				self._update_header(header, key, value)
			columns = []
			#Set data as columns
			for key,value in section.data.iteritems():
				format=self.get_fits_type(value)
				col = pyfits.Column(name=key, format=format, array=value)
				columns.append(col)
			#Pyfits chokes on empty tables.
			#Set up a dummy one if need be
			if not columns:
				value = np.zeros(1,dtype=int)
				format=self.get_fits_type(value)
				columns = [pyfits.Column(name=_DUMMY_KEY, format=format, array=value)]
			for (key,value) in section.params.iteritems():
				self._update_header(header, key, value)
			#Add this section extension to the list.
			ext=pyfits.new_table(columns,header=header)
			hdulist.append(ext)
		#Convert the whole FITS file to a string.
		s=cStringIO.StringIO()
		hdulist.writeto(s)
		s.seek(0)
		return s.read()
		
	
	def get_data_2d(self, section, x_name, y_name, z_name, nx, ny, x_changing_fastest):
		""" Load a 2D array that is specified on a grid of two other parameters.
		"""
		x = self.get_data(section, x_name).reshape((nx,ny))
		y = self.get_data(section, y_name).reshape((nx,ny))
		z = self.get_data(section, z_name).reshape((nx,ny))
		x_does_change_fastest = x[0,0]!=x[0,1]
		if x_does_change_fastest:
			x=x[0]
			y=y[:,0]
		else:
			x=x[:,0]
			y=y[0]
		swap = 	(x_does_change_fastest) != bool(x_changing_fastest)
		if swap:
			z=z.T
		return x, y, z, swap
	


@contextlib.contextmanager
def open_handle(handle):
	""" This context manager allows you to write code that automatically manages opening, saving to, and closing a handle to a data package
	All you need to do is:
	
	with pydesglue.open_handle(handle) as package:
		package.get_param(...)
		package.get_data(...)
		...
		package.set_data(...)
	
	And then all your changes to the package are saved at the end automatically
	"""
	package = DesDataPackage.from_fits_handle(handle)
	try:
		yield package
	finally:
		package.write_to_fits_handle(handle)
		


def _matches_fits_reserved(key):
	for reserved_key in _FITS_RESERVED_KEYS:
		if key.startswith(reserved_key):
			return True
	return False
