from .. import ParallelSampler
import numpy as np


def log_probability_function(p):
    return abc_pipeline.posterior(p)


class ABCSampler(ParallelSampler):
    parallel_output = False #True for 
    sampler_outputs = [("like", float)]

    def config(self):
        global abc_pipeline
        abc_pipeline = self.pipeline

        if self.is_master():
            # Parameters of the emcee sampler
            self.int_parameter = self.read_ini("int_parameter", int, 2)
            self.bool_parameter = self.read_ini("bool_parameter", bool, False)
            self.string_parameter  = self.read_ini("string_parameter", str, "default")
            self.ndim = len(self.pipeline.varied_params)

    def output_samples(self, pos, prob, extra_info):
        for p,l,e in zip(pos,prob,extra_info):
            self.output.parameters(p, e, l)

    def execute(self):
        # called only by master. Other processors will be in self.pool
        # generate outputs somehow by running log_probability_function
        for (samples, likelihoods, extra_outputs) in outputs:
            self.output_samples(samples, likelihoods, extra_outputs)

    def is_converged(self):
        return something >= something_else
