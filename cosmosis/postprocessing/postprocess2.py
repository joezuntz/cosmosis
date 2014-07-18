#!/usr/bin/env python
from cosmosis.postprocessing.postprocess import postprocessor_for_sampler
from cosmosis.runtime.config import Inifile
import sys
import argparse


parser = argparse.ArgumentParser(description="Post-process cosmosis output")
parser.add_argument("inifile")
mcmc=parser.add_argument_group(title="MCMC", description="Options for MCMC-type samplers")
mcmc.add_argument("--burn", type=float, help="Fraction or number of samples to burn at the start")
mcmc.add_argument("--thin", type=int, help="Keep every n'th sampler in MCMC")

test=parser.add_argument_group(title="Test", description="Options for the test sampler")
test.add_argument("-f", "--file-type", default="png", help="Filename suffix for plots")
test.add_argument("-o","--outdir", default=".", help="Output directory for plots")
test.add_argument("-p","--prefix", default="", help="Prefix for plots")

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
	processor = processor_class(ini, **vars(args))
	processor.run()
	#At some point we might run the processor on multiple chains
	#in that case we would call run more than once
	#and then finalize at the end
	processor.finalize()

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