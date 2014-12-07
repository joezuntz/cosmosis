import itertools
import numpy as np

from .. import ParallelSampler


def task(p):
    i,p = p
    results = grid_sampler.pipeline.posterior(p, return_data=grid_sampler.save_name)
    #If requested, save the data to file
    if grid_sampler.save_name and results[2] is not None:
        results[2].save_to_file(grid_sampler.save_name+"_%d"%i, clobber=True)
    return (results[0], results[1])




class GridSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("like", float)]

    def config(self):
        global grid_sampler
        grid_sampler = self

        self.converged = False
        self.nsample = self.read_ini("nsample_dimension", int, 1)
        self.save_name = self.read_ini("save", str, "")

    def execute(self):
        samples = list(itertools.product(*[np.linspace(*param.limits,
                                                       num=self.nsample)
                                           for param
                                           in self.pipeline.varied_params]))
        sample_index = range(len(samples))
        jobs = list(zip(sample_index, samples))

        if self.pool:
            results = self.pool.map(task, jobs)
        else:
            results = map(task, jobs)

        #Save the results of the sampling
        for sample, result  in itertools.izip(samples, results):
            #Optionally save all the results calculated by each
            #pipeline run to files
            (prob, extra) = result
            #always save the usual text output
            self.output.parameters(sample, extra, prob)
        #We only ever run this once, though that could 
        #change if we decide to split up the runs
        self.converged = True

    def is_converged(self):
        return self.converged
