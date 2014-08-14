from .. import ParallelSampler
import numpy as np
from . import metropolis

#We need a global pipeline
#object for MPI to work properly
pipeline=None

def posterior(p):
    return pipeline.posterior(p)


class MetropolisSampler(ParallelSampler):
    parallel_output = True
    sampler_outputs = [("like", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline
        self.num_samples = self.read_ini("num_samples", int, default=20000)
        #Any other options go here

        start = ?
        limits = ?
        covariance = ?

        self.sampler = metropolis.MCMC(start, limits, posterior, covariance, self.pool)

    def execute(self):
        #Run the emcee sampler.
        samples = self.sampler.sample()
        for vector, like in samples:
        	self.output.parameters(vector, like)
        self.sampler.tune()

    def is_converged(self):
        return self.num_samples >= self.samples
