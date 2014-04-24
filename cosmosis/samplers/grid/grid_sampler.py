import itertools
import numpy as np

from .. import ParallelSampler


GRID_INI_SECTION = "grid"


def task(p):
    return pipeline.posterior(p)


class GridSampler(ParallelSampler):

    def config(self):
        global pipeline
        pipeline = self.pipeline

        self.converged = False
        self.nsample = self.ini.getint(GRID_INI_SECTION,
                                       "nsample_dimension", 1)

    def execute(self):
        samples = list(itertools.product(*[np.linspace(*param.limits,
                                                       num=self.nsample)
                                           for param
                                           in self.pipeline.varied_params]))
        if self.pool:
            results = self.pool.map(task, samples)
        else:
            results = map(task, samples)

        for sample, (prob, extra) in zip(samples, results):
            self.output.parameters(sample, extra)
        self.converged = True

    def is_converged(self):
        return self.converged
