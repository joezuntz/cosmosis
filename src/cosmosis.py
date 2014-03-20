#!/usr/bin/env python

import sys
import argparse

# TODO: better search path handling or python module configuration
sys.path.append("runtime")
sys.path.append("datablock")

from config import Inifile
from pipeline import LikelihoodPipeline


RUNTIME_INI_SECTION = "runtime"


# TODO: find better home for this.  Utils?
class ParseExtraParameters(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        result = {}
        for arg in values:
            section, param_value = arg.split('.',1)
            param,value = param_value.split('=',1)
            result[(section,param)] = value
        setattr(args, self.dest, result)


def main(argv):
    parser = argparse.ArgumentParser(description="Run a pipeline with a single set of parameters", add_help=True)
    parser.add_argument("inifile", help="Input ini file of parameters")
#    parser.add_argument("outfile", help="Output results to file")
#    parser.add_argument("--mpi",action='store_true',help="Run in MPI mode.")
#    parser.add_argument("--parallel",action='store_true',help="Run in multiprocess parallel mode.")
#    parser.add_argument("--debug", action='store_true', default=False, help="Print additional debugging information")
    parser.add_argument("-t", "--timing", action='store_true', default=False, help='Time each module in the pipeline')
    parser.add_argument("-p", "--params", nargs="*", action=ParseExtraParameters, help="Over-ride parameters in inifile, with format section.name=value")
    parser.add_argument("-v", "--variables", nargs="*", action=ParseExtraParameters, help="Over-ride variables in values file, with format section.name=value")
    args = parser.parse_args(argv)

    # load configuration 
    ini = Inifile(args.inifile, override=args.variables)

    # create pipeline
    pipeline = LikelihoodPipeline(ini) 

    # create sampler object
    # TODO: better job of selecting sampler and importing/instantiating
    sample_method = ini.get(RUNTIME_INI_SECTION, "sampler", "test")
    sys.path.append("samplers/"+sample_method)

    if sample_method == "pymc":
        import pymc_sampler
        sampler = pymc_sampler.PyMCSampler(ini, pipeline)
    elif sample_method == "emcee":
        import emcee_sampler
        sampler = EmceeSampler(ini, pipeline)
    elif sample_method == "maxlike":
        import maxlike_sampler
        sampler = MaxlikeSampler(ini, pipeline)
    elif sample_method == "grid":
        import grid_sampler
        sampler = grid_sampler.GridSampler(ini, pipeline)
    elif sample_method == "test":
        import test_sampler
        sampler = test_sampler.TestSampler(ini, pipeline)
    else:
        raise ValueError("Unknown sampler method %s" % (sample_method,))

    sampler.config()

    # run the sampler
    while not sampler.is_converged():
        sampler.execute()

if __name__=="__main__":
    main(sys.argv[1:])
