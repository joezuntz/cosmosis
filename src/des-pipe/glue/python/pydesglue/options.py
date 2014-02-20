import os
import desglue
from . import utils


class OptionPackage(object):
	def __init__(self, handle):
		self.handle = handle

	def get(self, section, param, default=None):
		value = self.get_string(section, param, default=default)
		return utils.try_numeric(value)

	def get_string(self, section, param, default=None):
		try:
			value = desglue.get_option(self.handle, section, param)
			return value
		except KeyError:
			if default is not None:
				return default
			raise KeyError("Could not fined param %s in section %s" % (param, section))

	def get_int(self,section, param, default=None):
		return int(self.get_string(section, param, default))

	def get_float(self,section, param, default=None):
		return float(self.get_string(section, param, default))


class DesOptionPackage(OptionPackage):
	_default_section = "config"

	def data_path(self, relative_path):
		data_root = self.get_string('pipeline', 'data')
		return os.path.join(data_root, relative_path)

	def __getitem__(self,section_and_param):
		if isinstance(section_and_param, tuple):
			section, param = section_and_param
		else:
			section = self._default_section
			param = section_and_param
		return self.get(section, param)

