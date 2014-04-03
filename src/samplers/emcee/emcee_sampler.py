import numpy as np
import sys
import emcee
import os


EMCEE_INI_SECTION = "emcee"
				

def EmceeSampler(ParallelSampler):

    def config(self):
        def log_probability_function(p):
            return self.pipeline.posterior(p)

        if self.pool:
            self.pool.worker_function = log_probability_function

        if self.is_master():
            self.nwalkers = self.ini.getint(EMCEE_INI_SECTION, "walkers", 1)
            self.samples = self.ini.getint(EMCEE_INI_SECTION, "samples", 1000)

            ndim = len(self.pipeline.varied_params)
            self.p0 = [self.pipeline.randomized_start() for i in xrange(nwalkers)]

            self.ensemble = emcee.EnsembleSampler(self.nwalkers, ndim, 
                                                  log_probabilty_function, 
                                                  pool=self.pool)
            self.sampler = None

    def execute(self):
        if not self.sampler:
            self.sampler = self.ensemble(self.p0, 
                                         iterations=self.nsample,
                                         storechain=True)

        try:
            pos, prob, rstate, extra_info = self.sampler.next()
            self.num_samples += self.nsample
        except StopIteration:
            raise RuntimeError("Emcee sampler stopped before Cosmosis determined convergence")

    def is_converged(self):
        return self.num_samples >= self.sample
