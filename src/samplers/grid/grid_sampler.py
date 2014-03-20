import sys
import itertools
import numpy as np
import copy

from sampler import Sampler


GRID_INI_SECTION = "grid"


class GridSampler(Sampler)

    def config():
        self.nsample_dimension = self.ini.getint(GRID_INI_SECTION, "nsample_dimension", 1)

    def execute():

                
        
def grid_sample_parameters(baseline, ranges_file, params, nsample_dimension):
	param_value_arrays = []
	for section, param in params:
		try:
			min_value, base_value, max_value = ranges_file.get(section, param)
		except ConfigParser.NoSectionError:
			section = pydesglue.section_friendly_names[section]
			min_value, base_value, max_value = ranges_file.get(section, param)
		param_value_array = np.linspace(min_value,max_value,nsample_dimension)
		param_value_arrays.append(param_value_array)
	output = []
	for param_values in itertools.product(*param_value_arrays):
		param_dict = copy.deepcopy(baseline)
		for (param_data,value) in zip(params,param_values):
			section,param = param_data
			try:
				param_dict[section][param] = value
			except KeyError:
				section = getattr(pydesglue.section_names,section)
				print section,param,value,param_dict.keys()
				param_dict[section][param] = value
		output.append(param_dict)
	return output



PIPELINE_INI_SECTION = "pipeline"
globalPipeline = None
def task(inputs):
	parameters, filename = inputs
	print parameters
:wq	post, extra = globalPipeline.posterior(parameters, filename=filename)
	p = [post]
	for (section, param, param_range) in globalPipeline.varied_params:
		p.append(parameters[section][param])
	row_string="\t".join(["%-20s"%str(value) for value in p])
	return row_string


def grid_sample(inifile, parameters, nsample_dimension, outfile_base, pool=None, save=False):
	#Load the main ini file, which describes the pipeline, and create a pipeline from it
	#This loads all the modules specified in the ini file
	ini = pydesglue.ParameterFile.from_file(inifile)
	pipeline = pydesglue.LikelihoodPipeline(ini, debug=False)

	global globalPipeline
	globalPipeline = pipeline
	if hasattr(pool,'is_master') and not pool.is_master():
		pool.wait()
		return


	#One of the parameters in the ini file is "values", which 
	#specified where to look for a file of parameter values to use in the pipeline.
	#Find that value, and then load in the file.  It is called a ranges file because in general
	#it can include min and max values as well as starting ones, though here we will just use the start
	ranges_filename = pipeline.get_option(PIPELINE_INI_SECTION, "values")
	ranges_file = pydesglue.ParameterRangesFile.from_file(ranges_filename)
	baseline = ranges_file.to_fixed_parameter_dicts()


	parameters = [p.split("--") for p in parameters]
	parameter_sets = grid_sample_parameters(baseline, ranges_file, parameters, nsample_dimension)
	if save:
		filenames = [outfile_base+'_%d.fits'%i for i in xrange(len(parameter_sets))]
	else:
		filenames = [None for i in xrange(len(parameter_sets))]
	parameter_data = zip(parameter_sets,filenames)
	if pool:
		results = list(pool.map(task, parameter_data))
		pool.close()
	else:
		results = map(task,parameter_data)

	param_names = []
	for (section, param, param_range) in pipeline.varied_params:
		param_names.append("%s--%s"%(section,param))
	if outfile_base.endswith('.txt'):
		outfile_base = outfile_base[:-4]
	outfile = open(outfile_base+'.txt','w')
	outfile.write('# likelihood   \t')

	outfile.write('\t'.join(["%-20s"%p for p in param_names]))
	outfile.write('\n')

	for result in results:
		outfile.write("%s\n" %(result))
	outfile.close()
	#Return the results.
	return results
