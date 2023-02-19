from .. import ParallelSampler
import numpy as np
import sys


def log_probability_function(p):
    return pipeline.posterior(p)[0]


def prior_transform(p):
    return pipeline.denormalize_vector_from_prior(p)


class NautilusSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [('log_weight', float), ("like", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            self.n_live = self.read_ini("n_live", int, 1500)
            self.n_update = self.read_ini("n_update", int, self.n_live)
            self.enlarge = self.read_ini(
                "enlarge", float, 1.1**self.pipeline.nvaried)
            self.n_batch = self.read_ini("n_batch", int, 100)
            self.random_state = self.read_ini("random_state", int, -1)
            if self.random_state < 0:
                self.random_state = None
            self.filepath = self.read_ini("filepath", str, 'None')
            if self.filepath.lower() == 'none':
                self.filepath = None
            self.resume = self.read_ini("resume", bool, True)
            self.f_live = self.read_ini("f_live", float, 0.01)
            self.n_shell = self.read_ini("n_shell", int, self.n_batch)
            self.n_eff = self.read_ini("n_eff", float, 10000.0)
            self.discard_exploration = self.read_ini(
                "discard_exploration", bool, False)
            self.verbose = self.read_ini("verbose", bool, False)

        self.converged = False

    def execute(self):
        from nautilus import Sampler

        n_dim = self.pipeline.nvaried

        sampler = Sampler(
            prior_transform,
            log_probability_function,
            n_dim,
            n_live=self.n_live,
            n_update=self.n_update,
            enlarge=self.enlarge,
            n_batch=self.n_batch,
            random_state=self.random_state,
            filepath=self.filepath,
            resume=self.resume,
            pool=self.pool
        )

        sampler.run(f_live=self.f_live,
                    n_shell=self.n_shell,
                    n_eff=self.n_eff,
                    discard_exploration=self.discard_exploration,
                    verbose=self.verbose)

        for sample, logwt, logl in zip(*sampler.posterior()):
            self.output.parameters(sample, logwt, logl)

        self.output.final(
            "efficiency", sampler.effective_sample_size() / sampler.n_like)
        self.output.final("nsample", len(sampler.posterior()[0]))
        self.output.final("log_z", sampler.evidence())
        self.converged = True

    def is_converged(self):
        return self.converged
