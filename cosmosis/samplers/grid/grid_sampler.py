import itertools
import numpy as np

from .. import ParallelSampler


def task(p):
    return grid_sampler.pipeline.posterior(p, return_data=grid_sampler.save_name)


class GridSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("like", float)]

    def config(self):
        global grid_sampler
        grid_sampler = self

        self.converged = False
        self.nsample = self.read_ini("nsample_dimension", int, 1)
        self.save_name = self.read_ini("save_name", str, "")

    def execute(self):
        samples = list(itertools.product(*[np.linspace(*param.limits,
                                                       num=self.nsample)
                                           for param
                                           in self.pipeline.varied_params]))
        if self.pool:
            results = self.pool.map(task, samples)
        else:
            results = map(task, samples)

        #Save the results of the sampling
        for i,(sample, result)  in enumerate(itertools.izip(samples, results)):
            #Optionally save all the results calculated by each
            #pipeline run to files
            if self.save_name:
                (prob, extra, data) = result
                #If prior is violated no results are returned
                if data is not None:
                    data.save_to_file(self.save_name+"_%d"%i)
            else:
                (prob, extra) = result
            #always save the usual text output
            self.output.parameters(sample, extra, prob)
        #We only ever run this once, though that could 
        #change if we decide to split up the runs
        self.converged = True

    def is_converged(self):
        return self.converged
