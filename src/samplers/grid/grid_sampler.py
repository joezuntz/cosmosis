#!/usr/bin/env python
import pydesglue
import sys
import argparse
import itertools
import numpy as np
import copy
import ConfigParser
try:
	import progressbar
except ImportError:
	progressbar=None


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
	post, extra = globalPipeline.posterior(parameters, filename=filename)
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

parser = argparse.ArgumentParser(description="Sample parameters on a grid",add_help=True)
parser.add_argument("inifile",type=str,help="Input parameter ini file")
parser.add_argument("outfile",type=str,help="Output text file")
parser.add_argument("-n","--nsample_dimension",type=int,default=10,help="Number of samples per dimension")
parser.add_argument("--save",action='store_true',default=False,help="Save data for each parameter combination to fits files")
parser.add_argument("-p","--param",dest='params',action='append',help="List of parameters to grid over, in the form section--param")
parser.add_argument("--mpi",action='store_true',help="Run in MPI mode.  Requires emcee")

def main(argv):
	args = parser.parse_args(argv)
	pool=None
	if args.mpi:
		import temporary_utils
		pool=temporary_utils.MPIPool(debug=False)
	grid_sample(args.inifile, args.params, args.nsample_dimension, args.outfile, pool=pool, save=args.save)

if __name__=="__main__":
	main(sys.argv[1:])
