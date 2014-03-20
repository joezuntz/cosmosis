import numpy as np
import sys
import emcee
import os


EMCEE_INI_SECTION = "emcee"
				

def EmceeSampler(Sampler):

    def config(self):
        nwalkers = int(self.ini.getint(EMCEE_INI_SECTION, "walkers", 1))
        nsample = int(self.ini.getint(EMCEE_INI_SECTION, "samples"))

        ndim = len(self.pipeline.varied_params)
        self.p0 = [self.pipeline.randomized_start() for i in xrange(nwalkers)]

        def log_probability_function(p):
            return self.pipeline.posterior(p)

        self.ensemble = emcee.EnsembleSampler(nwalkers, ndim, log_probabilty_function)
        self.sampler = None

    def execute(self):
        if not self.sampler:
            self.sampler = self.ensemble(self.p0,self.

        try:
            pos, prob, rstate, extra_info = self.sampler.next()
        except StopIteration:
            raise RuntimeError("Emcee sampler stopped before Cosmosis determined convergence")

    def is_converged(self):
        return False
