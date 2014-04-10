import itertools
import numpy as np

from sampler import ParallelSampler


GRID_INI_SECTION = "grid"


def task(p):
    posterior, extra = pipeline.posterior(p)
    return posterior


class GridSampler(ParallelSampler):

    def config(self):
        global pipeline
        pipeline = self.pipeline

        self.converged = False
        self.nsample = self.ini.getint(GRID_INI_SECTION, "nsample_dimension", 1)

    def execute(self):
        samples = list(itertools.product(*[np.linspace(*p.limits, 
                                                       num=self.nsample)
                                         for p in self.pipeline.varied_params]))
        if self.pool:
            results = self.pool.map(task, samples)
        else:
            results = map(task, samples)

        print zip(samples,results)
        self.converged = True

    def is_converged(self):
        return self.converged 
