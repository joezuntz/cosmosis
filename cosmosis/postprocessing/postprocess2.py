#!/usr/bin/env python
from cosmosis.postprocessing.postprocess import postprocessor_for_sampler
from cosmosis.runtime.config import Inifile
import sys
import argparse


parser = argparse.ArgumentParser(description="Post-process cosmosis output")
parser.add_argument("inifile")

def main(args):
	#Read the command line arguments and load the
	#ini file that created the run
	args = parser.parse_args(args)
	ini_filename = args.inifile
	ini = Inifile(ini_filename)

	#Determine the sampler and get the class
	#designed to postprocess the output of that sampler
	sampler = ini.get("runtime", "sampler")
	processor_class = postprocessor_for_sampler(sampler)

	#Create and run the postprocessor
	processor = processor_class(ini)
	processor.run()

if __name__=="__main__":
	main(sys.argv[1:])



# ini = 
# m = MetropolisHastingsProcessor(ini)
# m.run()


# #This script needs to:
# # read an ini file 
# # decide which postprocessor elements to use
# # load in the data
# # run the postprocessor elements