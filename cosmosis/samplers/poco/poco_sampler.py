from .. import ParallelSampler
from ...runtime import logs
import numpy as np


def log_likelihood(p):
    like, _ = pipeline.likelihood(p)
    # PocoMC cannot cope with infinities
    if not np.isfinite(like):
        like = -1e30
    return like


def log_prior(p):
    return pipeline.prior(p)


class PocoSampler(ParallelSampler):
    parallel_output = False
    supports_resume = False
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            import pocomc

            self.pocomc = pocomc

            # Parameters of the sampler
            self.ndim = len(self.pipeline.varied_params)
            self.nparticles = self.read_ini("n_particles", int)
            self.nsample_add = self.read_ini("add_samples", int, 0)
            self.threshold = self.read_ini("threshold", float, 1.0)
            self.scale = self.read_ini("scale", bool, True)
            self.rescale = self.read_ini("rescale", bool, False)
            self.diagonal = self.read_ini("diagonal", bool, True)
            self.ess = self.read_ini("ess", float, 0.95)
            self.gamma = self.read_ini("gamma", float, 0.75)
            self.n_min = self.read_ini("n_min", int, self.ndim // 2)
            self.n_max = self.read_ini("n_max", int, self.ndim * 10)
            seed = self.read_ini("seed", int, 0)
            if seed == 0:
                seed = None

            # Starting positions and values for the chain
            self.num_samples = 0
            self.prob0 = None

            self.p0 = np.array(
                [self.pipeline.randomized_start() for i in range(self.nparticles)]
            )
            logs.overview(
                "Generating random starting positions from within prior"
            )

            bounds = np.array([p.limits for p in self.pipeline.varied_params])

            # Finally we can create the sampler
            self.sampler = self.pocomc.Sampler(
                n_dim=self.ndim,
                n_particles=self.nparticles,
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                random_state=seed,
                threshold=self.threshold,
                scale=self.scale,
                rescale=self.rescale,
                diagonal=self.diagonal,
                vectorize_likelihood=False,
                vectorize_prior=False,
                infer_vectorization=False,
                pool=self.pool,
                bounds=bounds,
            )

            self.converged = False

            for sec, name in pipeline.extra_saves:
                col = "{}--{}".format(sec, name)
                logs.warning(
                    "WARNING: POCO DOES NOT SUPPORT DERIVED PARAMS - NOT SAVING {}".format(
                        col
                    )
                )
                self.output.del_column(col)

    def execute(self):
        # There's no checkpointing yet
        self.sampler.run(
            self.p0, n_min=self.n_min, n_max=self.n_max, gamma=self.gamma, ess=self.ess
        )

        if self.nsample_add > 0:
            self.sampler.add_samples(self.nsample_add)

        results = self.sampler.results

        samples = results["samples"]
        likes = results["loglikelihood"]
        priors = results["logprior"]
        n = len(samples)
        outputs = []
        for (p, prior, like) in zip(samples, priors, likes):
            post = prior + like
            self.output.parameters(p, prior, post)

        self.output.final("log_z", results["logz"][-1])
        self.converged = True

    def is_converged(self):
        return self.converged
