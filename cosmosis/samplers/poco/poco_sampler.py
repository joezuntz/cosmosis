from .. import ParallelSampler
from ...runtime import logs
import numpy as np
import os

def likelihood(p):
    like, _ = pipeline.likelihood(p)
    # PocoMC cannot cope with infinities
    if not np.isfinite(like):
        like = -1e30
    return like


class CustomPrior:
    def __init__(self, pipeline):
        self.bounds = np.array([p.limits for p in pipeline.varied_params])
        self.ndim = len(pipeline.varied_params)
        self.pipeline = pipeline
    def logpdf(self, x):
        p = np.array([self.pipeline.prior(x_i) for x_i in x])
        p[~np.isfinite(p)] = -1e30
        return p
    def rvs(self, size):
        return np.array([self.pipeline.randomized_start() for i in range(size)])


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
            self.ndim = len(self.pipeline.varied_params)
            self.n_total = self.read_ini("n_total", int, 5000)
            self.n_evidence = self.read_ini("n_evidence", int, 5000)
            self.progress = self.read_ini("progress", bool, True)
            self.save_every = self.read_ini("save_every", int, 10)

            self.n_active = self.read_ini("n_active", int, 250)
            self.n_ess = self.read_ini("n_ess", int, 1000)
            self.precondition = self.read_ini("precondition", bool, True)
            self.n_prior = self.read_ini("n_prior", int, 2*(self.n_ess//self.n_active)*self.n_active)
            self.sample_method = self.read_ini_choices("sample_method", str, ["pcn", "rwm"], "pcn")
            self.max_steps = self.read_ini("max_steps", int, 5*self.ndim)
            self.patience = self.read_ini("patience", int, -1)

            self.ess_threshold = self.read_ini("ess_threshold", int, self.ndim * 4)
            seed = self.read_ini("seed", int, -1)
            if seed == -1:
                seed = None


            # Training configuration keys.
            # These defaults are listed in the documentation.

            validation_split = self.read_ini("validation_split", float, 0.5)
            epochs = self.read_ini("epochs", int, 2000)
            batch_size = self.read_ini("batch_size", int, 512)
            patience = self.read_ini("patience", int, 50)
            learning_rate = self.read_ini("learning_rate", float, 1e-3)
            annealing = self.read_ini("annealing", bool, False)
            gaussian_scale = self.read_ini("gaussian_scale", float, np.nan)
            laplace_scale = self.read_ini("laplace_scale", float, np.nan)
            noise = self.read_ini("noise", float, np.nan)
            shuffle = self.read_ini("shuffle", bool, True)
            clip_grad_norm = self.read_ini("clip_grad_norm", float, 1.0)
            verbose = self.read_ini("verbose", int, 0)

            if np.isnan(gaussian_scale):
                gaussian_scale = None
            if np.isnan(laplace_scale):
                laplace_scale = None
            if np.isnan(noise):
                noise = None


            self.train_config={
                "validation_split": validation_split,
                "epochs": epochs,
                "batch_size": batch_size,
                "patience": patience,
                "learning_rate": learning_rate,
                "annealing": annealing,
                "gaussian_scale": gaussian_scale,
                "laplace_scale": laplace_scale,
                "noise": noise,
                "shuffle": shuffle,
                "clip_grad_norm": clip_grad_norm,
                "verbose": verbose,
             }

            resume_file_name = self.output.name_for_sampler_resume_info()

            # We override the save_state method to use a single file path specified
            # by CosmoSIS instead of the default behavior of PocoMC.
            class Sampler(self.pocomc.Sampler):
                def save_state(self, filename):
                    super().save_state(resume_file_name)

            prior = CustomPrior(self.pipeline)

            # Finally we can create the sampler
            self.sampler = Sampler(
                prior=prior,
                likelihood=likelihood,
                n_dim=self.ndim,
                n_ess=self.n_ess,
                n_active=self.n_active,
                vectorize=False,
                pool=self.pool,
                # We do not currently support the custom normalizing flow option
                train_config=self.train_config,
                precondition=self.precondition,
                n_prior=self.n_prior,
                sample=self.sample_method,
                max_steps=self.max_steps,
                patience = self.patience if self.patience > 0 else None,
                ess_threshold=self.ess_threshold,
                random_state=seed,
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

    def resume(self):
        # We do not need to do anything here as the sampler will handle
        # resuming from the state file.
        pass

    def execute(self):
        # If checkpointing is enabled then we need to load the state
        # otherwise we just run the sampler from the start; it does its
        # own initialization.
        resume_file_name = self.output.name_for_sampler_resume_info()
        if os.path.exists(resume_file_name) and os.path.getsize(resume_file_name) > 0:
            logs.overview(
                "Ignore the line above where it says it will start sampling afresh.\n"
                "That's just because CosmoSIS doesn't know that PocoMC does a warm-up "
                "phase before starting to sample.\nIt will actually resume from the state "
                f"file called {resume_file_name}"
            )
        else:
            logs.overview(f"Starting PocoMC from scratch, since state file {resume_file_name} does not exist or is empty")
            resume_file_name = None

        self.sampler.run(
            n_total = self.n_total,
            n_evidence = self.n_evidence,
            resume_state_path = resume_file_name,
            save_every = self.save_every,
        )

        results = self.sampler.results

        samples = results["samples"]
        likes = results["loglikelihood"]
        priors = results["logprior"]
        n = len(samples)
        for (p, prior, like) in zip(samples, priors, likes):
            post = prior + like
            self.output.parameters(p, prior, post)

        self.output.final("log_z", results["logz"][-1])
        self.converged = True

    def is_converged(self):
        return self.converged
