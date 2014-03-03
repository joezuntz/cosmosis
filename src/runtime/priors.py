from .parser import IncludingConfigParser
from .data_package import  open_handle
from .section_names import section_names
import numpy as np



class PriorCalculator(object):
	UNIFORM = 1
	GAUSSIAN = 2
	EXPONENTIAL = 3
	def __init__(self, filenames):
		if isinstance(filenames, basestring):
			filenames = [filenames]
		self.prior_data = {}
		self.prior_functions = {}			
		for filename in filenames:
			self.parse_prior_file(filename)

	def __nonzero__(self):
		#return false if no priors listed
		return bool(self.prior_data)

	def parse_prior_file(self, filename):
		ini = IncludingConfigParser()
		ini.read(filename)
		for section in ini.sections():
			section_code = getattr(section_names, section, section)
			for name,value in ini.items(section):
				name = name.upper()
				prior_type, prior_function = self.parse_prior(name, value)
				parameters = [float(p) for p in value.split()[1:]]
				self.prior_data[(section_code,name)] = (prior_type, parameters)
				self.prior_functions[(section_code,name)] = prior_function

	def parse_prior(self, name, line):
		function_name, parameters = line.split(' ',1)
		try:
			parameters = [float(p) for p in parameters.split()]
		except ValueError:
			raise ValueError("Expected numerical parameters for prior %s"%name)
		if function_name.lower().startswith('uni'):
			try:
				a, b = parameters
			except ValueError:
				raise ValueError("Uniform prior on %s should have two parameters"%name)
			if not b>a:
				raise ValueError("For uniform prior on %s, upper limit < lower limit (%r<%r)"%(name,b,a))
			def uniform_prior(x):
				if not (x>a and x<b): return -np.inf
				return 0.0
			return (self.UNIFORM,uniform_prior)
		elif function_name.lower()[:3] in ['gau','nor']:
			try:
				mu, sigma = parameters
			except ValueError:
				raise ValueError("Gaussian prior on %s should have two parameters"%name)				
			sigma2 = sigma**2
			def gaussian_prior(x):
				return -0.5 * (x-mu)**2 / sigma2
			return (self.GAUSSIAN, gaussian_prior)
		elif function_name.lower().startswith('exp'):
			try:
				beta, = parameters
			except ValueError:
				raise ValueError("Exponential prior on %s should have one parameter"%name)									
			if not beta>0:
				raise ValueError("Exponential prior beta is negative for parameter %s (%r)" % (name, beta))
			def exponential_prior(x):
				if not x>0: return -np.inf
				return -x/beta
			return (self.EXPONENTIAL,exponential_prior)
		raise ValueError("Could not parse this as a prior: %s"%line)

	def add_prior_into_handle(self, handle):
		with open_handle(handle) as package:
			log_prior = self.get_prior_from_package(handle)
			package.set_param(section_names.likelihoods, "PRIOR")

	def get_prior_from_package(self, package):
		log_prior = 0.0
		for (section, name), prior_function in self.prior_functions.items():
			value = package.get_param(section, name)
			log_prior += self.get_prior_for_parameter(section, name, value)
		return log_prior

	def get_prior_for_parameter(self, section, name, value):
		prior_function = self.prior_functions.get((section,name), None)
		if prior_function:
			return prior_function(value)
		else:
			return 0.0

	def get_prior_for_collected_parameter_dict(self, param_dict):
		log_prior = 0.0
		for (section,name), value in param_dict.items():
			log_prior += self.get_prior_for_parameter(section, name, value)
		return log_prior

	def get_prior_for_nested_dict(self, section_dict):
		log_prior = 0.0
		for section,param_dict in section_dict.items():
			for (name, value) in param_dict.items():
				log_prior += self.get_prior_for_parameter(section, name, value)
		return log_prior






