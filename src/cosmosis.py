#!/usr/bin/env python

import sys
import argparse

# TODO: better search path handling or python module configuration
sys.path.append("runtime")
sys.path.append("datablock")

from config import Inifile
from pipeline import LikelihoodPipeline
import mpi_pool
from sampler import sampler_registry, ParallelSampler
import samplers


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

def main(args, pool=None):
    # load configuration 
    ini = Inifile(args.inifile, override=args.variables)

    # create pipeline
    pipeline = LikelihoodPipeline(ini) 

    # create sampler object
    sample_method = ini.get(RUNTIME_INI_SECTION, "sampler", "test")

    if sample_method not in sampler_registry:
        raise ValueError("Unknown sampler method %s" % (sample_method,))

    if pool:
        if not issubclass(sampler_registry[sample_method],ParallelSampler):
            raise ValueError("Sampler does not support parallel execution!")
        sampler = sampler_registry[sample_method](ini, pipeline, pool)
    else:
        sampler = sampler_registry[sample_method](ini, pipeline)
 
    sampler.config()

    if not pool or pool.is_master():
        while not sampler.is_converged():
            sampler.execute()
    else:
        sampler.worker()

#    if pool:
#        pool.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run a pipeline with a single set of parameters", add_help=True)
    parser.add_argument("inifile", help="Input ini file of parameters")
#    parser.add_argument("outfile", help="Output results to file")
    parser.add_argument("--mpi",action='store_true',help="Run in MPI mode.")
#    parser.add_argument("--parallel",action='store_true',help="Run in multiprocess parallel mode.")
#    parser.add_argument("--debug", action='store_true', default=False, help="Print additional debugging information")
    parser.add_argument("-t", "--timing", action='store_true', default=False, help='Time each module in the pipeline')
    parser.add_argument("-p", "--params", nargs="*", action=ParseExtraParameters, help="Over-ride parameters in inifile, with format section.name=value")
    parser.add_argument("-v", "--variables", nargs="*", action=ParseExtraParameters, help="Over-ride variables in values file, with format section.name=value")
    args = parser.parse_args(sys.argv[1:])

    # initialize parallel workers
    if args.mpi:
        with mpi_pool.MPIPool() as pool:
            main(args,pool)
#    elif parallel:
#        pool = ProcessPool()
    else:
        main(args)
