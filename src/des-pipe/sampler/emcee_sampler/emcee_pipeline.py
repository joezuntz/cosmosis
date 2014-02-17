import numpy as np
import pydesglue
import sys
import emcee

EMCEE_INI_SECTION = "emcee"
PIPELINE_INI_SECTION = "pipeline"

class EverythingIsNan(object):
	def __getitem__(self, param):
		return np.nan
everythingIsNan = EverythingIsNan()

class EmceePipeline(pydesglue.LikelihoodPipeline):
	def __init__(self, arg=None, quiet=False, id=""):
		super(EmceePipeline, self).__init__(arg=arg, quiet=quiet)
		param_values_filename = self.get_option(PIPELINE_INI_SECTION,"values")

		if id:
			self.id_code = "[%s] " % str(id)
		else:
			self.id_code = ""
		self.n_iterations = 0

		#We want to save some parameter results from the run for further output
		extra_saves = self.get_option(EMCEE_INI_SECTION,"extra_output", "")
		self.extra_saves = []
		for extra_save in extra_saves.split():
			section, name = extra_save.split('/')
			section = section.upper()
			name = name.upper()
			section = getattr(pydesglue.section_names, section.lower(), section)
			self.extra_saves.append((section, name))


		#Load the values file and get all the parameters from it.
		param_value_file = pydesglue.ParameterRangesFile.from_file(param_values_filename)
		param_info = param_value_file.param_list()

		#Now split these into varied (sampled) parameters and fixed ones
		self.varied_params = []
		self.fixed_params = []
		#Loop throuh the parameters and get their ranges.
		#a range where min==max indicates a fixed parameter.
		# min<max indicates varied
		# and anything else is an error
		for (section, param, param_range) in param_info:
			param_min, param_start, param_max = param_range
			#Check for some obvious errors
			if param_min>param_max:
				raise ValueError("In your values file you set min>max for parameter %s in section %s (%g>%g) " % (param,section,param_min, param_max) )
			elif param_start>param_max or param_start<param_min:
				raise ValueError("In your values file you set a starting value (%g) for parameter %s in section %s outside the min-max range (%g to %g)  " % (param_start, param,section,param_min, param_max) )
			elif param_min==param_max: 
				#parameter is fixed - no problem.
				self.fixed_params.append((section, param, param_start))
			else:
				#parameter is varied
				self.varied_params.append((section, param, param_range))

		#pull out all the section names and likelihood names for later
		self.section_names = set([name for (name, param, value) in self.fixed_params+self.varied_params])
		self.likelihood_names = self.get_option("pipeline","likelihoods").split()

	def randomized_start(self):
		output = np.zeros(len(self.varied_params))
		for i,(section, name, param_range) in enumerate(self.varied_params):
			(pmin, pstart, pmax) = param_range
			output[i] = pstart + (pmax-pmin)/100.0 * np.random.randn()
		return output
