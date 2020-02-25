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
    r = pipeline.run_results(p)
    return r
    #r.post, (r.prior, r.extra)


class MetropolisSampler(ParallelSampler):
    parallel_output = True
    sampler_outputs = [("prior", float),("post", float)]
    understands_fast_subspaces = True
    supports_resume = True

    def config(self):
        global pipeline
        pipeline = self.pipeline
        self.samples = self.read_ini("samples", int, default=20000)
        random_start = self.read_ini("random_start", bool, default=False)
        use_cobaya = self.read_ini("cobaya", bool, default=False)
        self.Rconverge = self.read_ini("Rconverge", float, -1.0)
        self.drag = self.read_ini("drag", int, 0)
        self.oversampling = self.read_ini("oversampling", int, 5)
        tuning_frequency = self.read_ini("tuning_frequency", int, -1)
        tuning_grace = self.read_ini("tuning_grace", int, 5000)
        self.tuning_end = self.read_ini("tuning_end", int, 100000)
        self.n = self.read_ini("nsteps", int, default=100)
        self.exponential_probability = self.read_ini("exponential_probability", float, default=0.333)
        self.split = None #work out later
        if self.Rconverge==-1.0:
            self.Rconverge=None
        self.interrupted = False
        self.num_samples = 0
        self.ndim = len(self.pipeline.varied_params)
        self.num_samples_post_tuning = 0
        self.last_accept_count = 0
        #Any other options go here

        if (self.drag > 0) and not self.pipeline.do_fast_slow:
            print("You asked for dragging, but the pipeline does not have fast/slow enabled"
                  ", so no draggng will be done."
                )

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

        if use_cobaya:
            print("Using the Cobaya proposal")


        self.sampler = metropolis.MCMC(start_norm, posterior, covmat_norm, 
            quiet=quiet, 
            tuning_frequency=tuning_frequency, # Will be multiplied by the oversampling
            tuning_grace=tuning_grace,         # within the sampler if needed
            tuning_end=self.tuning_end,
            exponential_probability=self.exponential_probability,
            use_cobaya=use_cobaya,
            n_drag = self.drag,
        )
        self.analytics = Analytics(self.pipeline.varied_params, self.pool)
        self.fast_slow_done = False

    def worker(self):
        while not self.is_converged():
            self.execute()
            if self.output:
                self.output.flush()


    def resume(self):
        resume_info = self.read_resume_info()
        if resume_info is None:
            return

        sampler, self.num_samples, self.num_samples_post_tuning = resume_info

        self.sampler = sampler

        # If we started main sampling (as opposed to tuning phase)
        # then we will have some existing chain, but this is not always the case
        try:
            data = np.genfromtxt(self.output._filename, invalid_raise=False)[:, :self.ndim]
            self.analytics.add_traces(data)
        except IndexError:
            data = None

        
        if self.num_samples >= self.samples:
            print("You told me to resume the chain - it has already completed (with {} samples), so sampling will end.".format(len(data)))
            print("Increase the 'samples' parameter to keep going.")
        elif self.is_converged():
            print("The resumed chain was already converged.  You can change the converged testing parameters to extend it.")
        elif data is None:
            print("Continuing metropolis from existing chain - you were in the tuning phase, which will continue")
        else:
            print("Continuing metropolis from existing chain - have {} samples already".format(len(data)))






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
        self.num_samples_post_tuning = self.num_samples - self.tuning_end


        # Only output samples once tuning is complete
        if self.num_samples_post_tuning > 0:
            traces = np.empty((self.n, self.ndim))
            likes = np.empty((self.n))


            samples = samples[-self.num_samples_post_tuning:]
            for i, result in enumerate(samples):
                self.output.parameters(result.vector, result.extra, result.prior, result.post)
                traces[i,:] = result.vector

            self.analytics.add_traces(traces)

            overall_rate = self.sampler.accepted / self.sampler.iterations
            recent_accepted = self.sampler.accepted - self.last_accept_count
            recent_rate = recent_accepted / self.n
            print("Overall accepted {} / {} samples ({:.1%})" .format(
                self.sampler.accepted, self.sampler.iterations, overall_rate))
            print("Last {0} accepted {1} / {0} samples ({2:.1%})\n" .format(
                self.n, recent_accepted, recent_rate))
            self.last_accept_count = self.sampler.accepted
        else:
            print("Done {} samples. Tuning proposal until {} so no output yet\n".format(
                self.num_samples, self.tuning_end))

        self.write_resume_info([self.sampler, self.num_samples, self.num_samples_post_tuning])

    def is_converged(self):
         # user has pressed Ctrl-C
        if self.interrupted:
            return True
        if self.num_samples >= self.samples:
            print("Full number of samples generated; sampling complete")
            return True
        elif (self.num_samples > 0 and
              self.pool is not None and
              self.Rconverge is not None and
              self.num_samples_post_tuning > 0):
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
