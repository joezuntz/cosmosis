from cosmosis.samplers import ParallelSampler
import numpy as np
import sys


def task(p):
    r = pipeline.run_results(p)
    return r.vector, r.extra, r.prior, r.post

# This (not very useful, except possibly for debugging)
# sampler draws a line between the two
# extreme corners of the distribution and evaluates points
# along it.

class CornerLineSampler(ParallelSampler):
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        self.ndim = self.pipeline.nvaried
        self.nsample = self.read_ini('nsample', int)
        corner1 = np.array([p.limits[0] for p in self.pipeline.varied_params])
        corner2 = np.array([p.limits[1] for p in self.pipeline.varied_params])
        x = np.linspace(0, 1, self.nsample)
        self.samples = corner1 + x[:, np.newaxis] * (corner2 - corner1)
        self.complete = False


    def execute(self):
        # Evaluate points optionally in parallel
        jobs = list(self.samples)

        if self.pool:
            results = self.pool.map(task, jobs)
        else:
            results = list(map(task, jobs))

        for (vector, extra, prior, post) in results:
            self.output.parameters(vector, extra, prior, post)
        self.complete = True

    def is_converged(self):
        return self.complete
