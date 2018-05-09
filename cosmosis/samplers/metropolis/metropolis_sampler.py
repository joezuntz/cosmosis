from __future__ import print_function
from builtins import zip
from builtins import str
from .. import ParallelSampler
import numpy as np
from . import metropolis
from cosmosis.runtime.analytics import Analytics
import os

#We need a global pipeline
#object for MPI to work properly
pipeline=None


METROPOLIS_INI_SECTION = "metropolis"

def posterior(p):
    p = pipeline.denormalize_vector(p, raise_exception=False)
    return pipeline.posterior(p)


class MetropolisSampler(ParallelSampler):
    parallel_output = True
    sampler_outputs = [("post", float)]
    understands_fast_subspaces = True

    def config(self):
        global pipeline
        pipeline = self.pipeline
        self.samples = self.read_ini("samples", int, default=20000)
        random_start = self.read_ini("random_start", bool, default=False)
        self.Rconverge = self.read_ini("Rconverge", float, -1.0)
        self.oversampling = self.read_ini("oversampling", int, 5)
        tuning_frequency = self.read_ini("tuning_frequency", int, -1)
        tuning_grace = self.read_ini("tuning_grace", int, 5000)
        tuning_end = self.read_ini("tuning_end", int, 100000)
        self.n = self.read_ini("nsteps", int, default=100)
        self.exponential_probability = self.read_ini("exponential_probability", float, default=0.333)
        self.split = None #work out later
        if self.Rconverge==-1.0:
            self.Rconverge=None
        self.interrupted = False
        self.num_samples = 0
        #Any other options go here

        #start values from prior
        start = self.define_parameters(random_start)
        print("MCMC starting point:")
        for param, x in zip(self.pipeline.varied_params, start):
            print("    ", param, x)

        #Covariance matrix
        covmat = self.load_covariance_matrix()

        #Sampler object itself.
        quiet = self.pipeline.quiet
        start_norm = self.pipeline.normalize_vector(start)
        covmat_norm = self.pipeline.normalize_matrix(covmat)
        self.sampler = metropolis.MCMC(start_norm, posterior, covmat_norm, 
            quiet=quiet, 
            tuning_frequency=tuning_frequency * self.oversampling, 
            tuning_grace=tuning_grace,
            tuning_end=tuning_end,
            exponential_probability=self.exponential_probability)
        self.analytics = Analytics(self.pipeline.varied_params, self.pool)
        self.fast_slow_done = False

    def worker(self):
        while not self.is_converged():
            self.execute()
            if self.output:
                self.output.flush()





    def execute(self):
        #Run the MCMC  sampler.
        if self.pipeline.do_fast_slow and not self.fast_slow_done:
            self.fast_slow_done = True
            self.sampler.set_fast_slow(
                self.pipeline.fast_param_indices,
                self.pipeline.slow_param_indices, 
                self.oversampling
            )

        try:
            samples = self.sampler.sample(self.n)
        except KeyboardInterrupt:
            self.interrupted=True
            return
        self.num_samples += self.n
        traces = np.empty((self.n,len(self.pipeline.varied_params)))
        likes = np.empty((self.n))
        for i, (vector, like, extra) in enumerate(samples):
            vector = self.pipeline.denormalize_vector(vector)
            self.output.parameters(vector, extra, like)
            traces[i,:] = vector
        self.analytics.add_traces(traces)	

        rate = self.sampler.accepted * 100.0 / self.sampler.iterations
        print("Accepted %d / %d samples (%.2f%%)\n" % \
            (self.sampler.accepted, self.sampler.iterations, rate))

    def is_converged(self):
         # user has pressed Ctrl-C
        if self.interrupted:
            return True
        if self.num_samples >= self.samples:
            print("Full number of samples generated; sampling complete")
            return True
        elif self.num_samples > 0 and \
                self.pool is not None and \
                self.Rconverge is not None:
            R = self.analytics.gelman_rubin(quiet=False)
            R1 = abs(R - 1)
            return np.all(R1 <= self.Rconverge)
        else:
            return False



    def load_covariance_matrix(self):
        covmat_filename = self.read_ini("covmat", str, "").strip()
        if covmat_filename == "" and self.distribution_hints.has_cov():
                covmat =  self.distribution_hints.get_cov() 
        elif covmat_filename == "":
            covmat = np.array([p.width()/100.0 for p in self.pipeline.varied_params])
        elif not os.path.exists(covmat_filename):
            raise ValueError(
            "Covariance matrix %s not found" % covmat_filename)
        else:
            covmat = np.loadtxt(covmat_filename)

        if covmat.ndim == 0:
            covmat = covmat.reshape((1, 1))
        elif covmat.ndim == 1:
            covmat = np.diag(covmat ** 2)

        nparams = len(self.pipeline.varied_params)
        if covmat.shape != (nparams, nparams):
            raise ValueError("The covariance matrix was shape (%d x %d), "
                    "but there are %d varied parameters." %
                    (covmat.shape[0], covmat.shape[1], nparams))
        return covmat



    def define_parameters(self, random_start):
        if random_start:
            return self.pipeline.randomized_start()
        else:
            return self.pipeline.start_vector()
