import itertools
import numpy as np

from .. import ParallelSampler


def task(p):
    return grid_pipeline.posterior(p)


class GridSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("like", float)]

    def config(self):
        global grid_pipeline
        grid_pipeline = self.pipeline

        self.converged = False
        self.nsample = self.read_ini("nsample_dimension", int, 1)

    def execute(self):
        samples = list(itertools.product(*[np.linspace(*param.limits,
                                                       num=self.nsample)
                                           for param
                                           in self.pipeline.varied_params]))
        if self.pool:
            results = self.pool.map(task, samples)
        else:
            results = map(task, samples)

        for sample, (prob, extra) in itertools.izip(samples, results):
            self.output.parameters(sample, extra, prob)
        self.converged = True

    def is_converged(self):
        return self.converged
