from .. import ParallelSampler, sample_ellipsoid, sample_ball
import numpy as np
import sys
import hashlib


def log_likelihood(p):
    like, _ = pipeline.likelihood(p)
    return like

def log_prior(p):
    return pipeline.prior(p)


class PocoSampler(ParallelSampler):
    parallel_output = False
    supports_resume = True
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            import pocomc
            self.pocomc = pocomc

            # Parameters of the sampler
            self.nparticles = self.read_ini("n_particles", int)
            seed = self.read_ini("seed", int, 0)
            if seed == 0:
                seed = None

            self.ndim = len(self.pipeline.varied_params)

            #Starting positions and values for the chain
            self.num_samples = 0
            self.prob0 = None

            self.p0 = np.array([self.pipeline.randomized_start() for i in range(self.nparticles)])
            self.output.log_info("Generating random starting positions from within prior")

            bounds = np.array([p.limits for p in self.pipeline.varied_params])

            #Finally we can create the sampler
            self.sampler = self.pocomc.Sampler(n_dim = self.ndim, n_particles=self.nparticles, log_likelihood=log_likelihood,
                log_prior=log_prior, random_state=seed, vectorize_likelihood=False, vectorize_prior=False,
                infer_vectorization=False, pool=self.pool,
                bounds=bounds)

            self.converged = False

            for sec,name in pipeline.extra_saves:
                col = "{}--{}".format(sec,name)
                print("WARNING: POCO DOES NOT SUPPORT DERIVED PARAMS - NOT SAVING {}".format(col))
                self.output.del_column(col)


    def execute(self):
        # There's no checkpointing yet
        self.sampler.run(self.p0)

        results = self.sampler.results

        samples = results['samples']
        likes = results['loglikelihood']
        priors = results['logprior']
        n = len(samples)
        outputs = []
        for (p, prior, like) in zip(samples, priors, likes):
            post = prior + like
            self.output.parameters(p, prior, post)

        self.output.final("log_z", results['logz'][-1])
        self.converged = True

    def is_converged(self):
        return self.converged
