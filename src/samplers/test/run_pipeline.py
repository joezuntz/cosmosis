#!/usr/bin/env python
"""

The syntax for this script is: 
 
> python line.py ini_file  output_fits_file

"""
import pydesglue
import sys
try:
	import argparse
except ImportError:
	sys.stderr.write("Please install argparse using 'easy_install argparse --user' or 'pip install argparse --user'. ")
	sys.exit(1)


PIPELINE_INI_SECTION = "pipeline"

def usage():
	sys.stderr.write(__doc__)

def run_pipeline(inifile, start=None, extra_variables=None, extra_parameters=None, timing=False):
	#Load the main ini file, which describes the pipeline 
	ini = pydesglue.ParameterFile.from_file(inifile)

	# Add any extra over-ride parameters
	if extra_parameters:
		for (section, param), value in extra_parameters.items():
			print 'setting', section, param, value
			ini.set(section, param, value)

	#Create a pipeline from the ini file
	#This loads all the modules specified in the ini file
	pipeline = pydesglue.LikelihoodPipeline(ini, debug=True, timing=timing)

	#One of the parameters in the ini file is "values", which 
	#specified where to look for a file of parameter values to use in the pipeline.
	#Find that value, and then load in the file.  It is called a ranges file because in general
	#it can include min and max values as well as starting ones, though here we will just use the start
	ranges_filename = pipeline.get_option(PIPELINE_INI_SECTION, "values")
	ranges_file = pydesglue.ParameterRangesFile.from_file(ranges_filename)

	# Add any extra over-ride variables
	if extra_variables:
		for (section, param), value in extra_variables.items():
			ranges_file.set(section, param, value)


	# Run the pipeline on the parameters
	result = pipeline.run_ranges_file(ranges_file, start=start)
	try:
		prior  = pipeline.prior(ranges_file.to_fixed_parameter_dicts())
		like   = pipeline.extract_likelihood(result)
		print "Prior      = ", prior
		print "Likelihood = ", like
		print "Posterior  = ", like+prior
	except:
		print "(Could not get a likelihood)"

	#Return the result.  You could just print out one result from it if you wanted, or save the whole thing.
	return result

class ParseExtraParameters(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
    	result = {}
    	for arg in values:
    		section, param_value = arg.split('.',1)
    		param,value = param_value.split('=',1)
    		result[(section,param)] = value
        setattr(args, self.dest, result)

parser = argparse.ArgumentParser(description="Run a pipeline with a single set of parameters", add_help=True)
parser.add_argument("inifile", help="Input ini file of parameters")
parser.add_argument("outfile", help="Output results FITS file")
parser.add_argument("--start", default=None, help="Resume run from this FITS file")
parser.add_argument("-t", "--timing", action='store_true', default=False, help='Time each module in the pipeline')
parser.add_argument("-p", "--params", nargs="*", action=ParseExtraParameters, help="Over-ride parameters in inifile, with format section.name=value")
parser.add_argument("-v", "--variables", nargs="*", action=ParseExtraParameters, help="Over-ride variables in values file, with format section.name=value")

def main(argv):
	args = parser.parse_args(argv)

	data = run_pipeline(args.inifile, start=args.start, 
		extra_variables=args.variables, extra_parameters=args.params,
		timing=args.timing)

	if data is None:
		sys.stderr.write("The pipeline failed, so nothing is being saved.\n")
	else:
		data.save_to_file(args.outfile)
	return data

if __name__=="__main__":
	main(sys.argv[1:])
