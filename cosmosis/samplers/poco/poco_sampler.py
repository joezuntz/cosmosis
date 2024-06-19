from .. import ParallelSampler
from ...runtime import logs
import numpy as np


def log_likelihood(p):
    r = pipeline.run_results(p)
    like = r.like
    # PocoMC cannot cope with -inf posteriors
    if not np.isfinite(like):
        like = -1e30
    return like, r.extra


class Prior:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.dim = len(self.pipeline.varied_params)
        self.bounds = np.array([p.limits for p in self.pipeline.varied_params])

    def logpdf(self, x):
        return np.array([self.pipeline.prior(p) for p in x])
    
    def rvs(self, size=None):
        return np.array([self.pipeline.randomized_start() for i in range(size)])


class PocoMCSampler(ParallelSampler):
    parallel_output = False
    supports_resume = True
    #internal_resume = True
    sampler_outputs = [('log_weight', float), ('prior', float),
                       ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            import pocomc

            self.pocomc = pocomc

            # Parameters of the sampler
            self.n_effective = self.read_ini("n_effective", int, 512)
            self.n_active = self.read_ini("n_active", int, 256)
            self.flow = self.read_ini("flow", str, "nsf6")
            self.precondition = self.read_ini("precondition", bool, True)
            self.dynamic = self.read_ini("dynamic", bool, True)
            self.output_dir = self.read_ini("output_dir", str, "output")            
            seed = self.read_ini("seed", int, 0)
            if seed == 0:
                seed = None

            # Parameters of the run
            self.n_total = self.read_ini("n_total", int, 4096)
            self.n_evidence = self.read_ini("n_evidence", int, 4096)
            self.progress = self.read_ini("progress", bool, True)
            self.resume_state_path = self.read_ini("resume_state_path", str, "")
            if self.resume_state_path == "":
                self.resume_state_path = None
            self.save_every = self.read_ini("save_every", int, -1)
            if self.save_every == -1:
                self.save_every = None

            # Finally we can create the sampler
            self.sampler = self.pocomc.Sampler(
                prior=Prior(self.pipeline),
                likelihood=log_likelihood,
                random_state=seed,
                n_effective=self.n_effective,
                n_active=self.n_active,
                flow=self.flow,
                precondition=self.precondition,
                dynamic=self.dynamic,
                output_dir=self.output_dir,
                pool=self.pool,
                blobs_dtype=float,
            )

            self.converged = False

    def execute(self):
        self.sampler.run(
            n_total=self.n_total,
            n_evidence=self.n_evidence,
            progress=self.progress,
            resume_state_path=self.resume_state_path,
            save_every=self.save_every,
        )

        logz, logz_err = self.sampler.evidence()

        for sample, logw, logl, logp, blob in zip(*self.sampler.posterior(return_blobs=True, return_logw=True)):
            extra = np.atleast_1d(blob)
            prior = logp
            logpost = logl + prior
            self.output.parameters(sample, extra, logw, prior, logpost)

        self.output.final("log_z", logz)
        self.output.final("log_z_error", logz_err)
        self.converged = True

    def is_converged(self):
        return self.converged
