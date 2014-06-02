import itertools
import numpy as np

from .. import ParallelSampler


GRID_INI_SECTION = "grid"


def task(p):
    return grid_pipeline.posterior(p)


class GridSampler(ParallelSampler):
    parallel_output = False

    def config(self):
        global grid_pipeline
        grid_pipeline = self.pipeline

        self.converged = False
        self.nsample = self.ini.getint(GRID_INI_SECTION,
                                       "nsample_dimension", 1)

    def execute(self):
        self.output.comment("Running with %d samples per dimension"%self.nsample)
        samples = list(itertools.product(*[np.linspace(*param.limits,
                                                       num=self.nsample)
                                           for param
                                           in self.pipeline.varied_params]))
        if self.pool:
            results = self.pool.map(task, samples)
        else:
            results = map(task, samples)

        for sample, (prob, extra) in itertools.izip(samples, results):
            self.output.parameters(sample, extra)
        self.converged = True

    def is_converged(self):
        return self.converged
