import os
import sys
import collections
from . import parser
from . import odict
from . import section_names
from . import utils
from . import data_package


class Inifile(parser.IncludingConfigParser):
	def __init__(self, defaults=None):
		parser.IncludingConfigParser.__init__(self,
			defaults=defaults, 
			dict_type=odict.OrderedDict)


	def iter_all(self):
		for section in self.sections():
			for name, value in self.items(section):
				yield section, name, value

	@classmethod
	def from_file(cls, filename):
		if not os.path.exists(filename):
			raise ValueError("Tried to find non-existent ini file called %s" % filename)
		ini = cls()
		ini.read(filename)
		return ini

class ParameterFile(Inifile):
	def __init__(self, defaults=None):
		Inifile.__init__(self, defaults=defaults)

	def params_from_section(self, section):
		output = {}
		for name, value in self.items(section):
			output[name] = utils.try_numeric(value)
		return output
	def data_package_from_section(self, section):
		params = self.params_from_section(section)
		return data_package.DesDataPackage.from_cosmo_params(params)


class ParameterRangesFile(Inifile):
	@classmethod
	def parse_range(cls, info):
		words = info.split()
		if len(words)==1:
			value = utils.try_numeric(words[0])
			return (value, value, value)
		elif len(words)==3:
			low, start, high = [utils.try_numeric(p) for p in words]
			return (low, start, high)
		else:
			raise ValueError("Was expecting there to be one (for fixed param) or three (for varied) words in line %s of parameter range file" % info)
	def get(self, section, param):
		return self.parse_range(Inifile.get(self,section, param))
	def iter_all(self):
		for (section, name, value) in Inifile.iter_all(self):
			yield getattr(section_names.section_names, section, section), name.upper(), self.parse_range(value)
	def to_parameter_dicts(self):
		params = {}
		for section in self.sections():
			section_code = getattr(section_names.section_names, section, section)
			params[section_code] = {}
			for param, value in self.items(section):
				param = param.upper()
				params[section_code][param] = self.parse_range(value)
		return params
	def to_fixed_parameter_dicts(self):
		params = {}
		for section in self.sections():
			section_code = getattr(section_names.section_names, section, section)
			params[section_code] = {}
			for param, value in self.items(section):
				param = param.upper()
				params[section_code][param] = self.parse_range(value)[1]
		return params

	@staticmethod
	def is_fixed(param_range):
		return param_range[0]==param_range[2]

	@staticmethod
	def is_varied(param_range):
		return param_range[0]!=param_range[2]

	def param_list(self):
		return list(self.iter_all())
	def varied_param_list(self):
		return [(section,name,value) 
				  for (section, name, param_range) in self.iter_all()
				  if self.is_varied(param_range)
		]

	def fixed_param_list(self):
		return [(section,name,value) 
				  for (section, name, param_range) in self.iter_all()
				  if self.is_fixed(param_range)
		]

	@staticmethod
	def list_to_dicts(param_list):
		output = {}
		for (section, _, _) in param_list:
			output.setdefault(section,dict())
		for (section, name, value) in param_list:
			output[section][name] = value

	def varied_param_dicts(self):
		return self.list_to_dicts(self.varied_param_dicts())

	def fixed_param_dicts(self):
		return self.list_to_dicts(self.fixed_param_dicts())




