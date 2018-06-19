from builtins import zip
from builtins import range
from builtins import str
from .. import ParallelSampler
import numpy as np
import sys


def log_probability_function(p):
    return pipeline.posterior(p)[0]

def prior_transform(p):
    return pipeline.denormalize_vector_from_prior(p)

class DynestySampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [('log_weight', float), ("like", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            self.mode = self.read_ini_choices("mode", str, ["static", "dynamic"], "static")
            self.nlive = self.read_ini("nlive", int, 500)
            self.bound = self.read_ini_choices("bound", str, ["none", "single", "multi", "balls", "cube"], "multi")
            self.sample = self.read_ini_choices("sample", str, ["unif", "rwalk", "slice", "rslice", "hslice"], "rwalk")
            self.update_interval = self.read_ini("update_interval", float, 0.6)
            self.min_ncall = self.read_ini("min_ncall",int, 2*self.nlive)
            self.min_eff = self.read_ini("min_eff",float, 10.0)
            default_queue_size = 1 if self.pool is None else self.pool.size
            self.queue_size = self.read_ini("queue_size", int, default_queue_size)
            self.parallel_prior = self.read_ini("parallel_prior", bool, True)
            self.max_call = self.read_ini("max_call", int, sys.maxsize)
            self.dlogz = self.read_ini("dlogz", float, 0.01)
            self.print_progress = self.read_ini("print_progress", bool, True)

            if self.mode=='dynamic':
                raise ValueError("Dynesty mode 'dynamic' not yet implemented (sorry)")

            for sec,name in pipeline.extra_saves:
                col = "{}--{}".format(sec,name)
                print("WARNING: DYNESTY DOES NOT SUPPORT DERIVED PARAMS - NOT SAVING {}".format(col))
                self.output.del_column(col)

        self.converged = False



    def execute(self):
        from dynesty import NestedSampler

        ndim = self.pipeline.nvaried
        sampler = NestedSampler(
            log_probability_function,
            prior_transform, 
            ndim, 
            nlive = self.nlive,
            bound = self.bound,
            sample = self.sample,
            update_interval = self.update_interval,
            first_update = {'min_ncall':self.min_ncall, 'min_eff':self.min_eff},
            queue_size = self.queue_size,
            pool = self.pool
            )
        sampler.run_nested()
        results = sampler.results
        results.summary()

        for sample, logwt, logl in zip(results['samples'],results['logwt'], results['logl']):
            self.output.parameters(sample, logwt, logl)

        self.output.final("ncall", results['ncall'])
        self.output.final("efficiency", results['eff'])
        self.output.final("log_z", results['logz'])
        self.output.final("log_z_err", results['logzerr'])

        self.converged = True



    def is_converged(self):
        return self.converged
