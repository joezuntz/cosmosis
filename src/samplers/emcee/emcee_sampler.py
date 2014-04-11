from sampler import ParallelSampler

EMCEE_INI_SECTION = "emcee"


def log_probability_function(p):
    return pipeline.posterior(p)


class EmceeSampler(ParallelSampler):

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            import emcee
            self.emcee = emcee

            self.nwalkers = self.ini.getint(EMCEE_INI_SECTION, "walkers", 2)
            self.samples = self.ini.getint(EMCEE_INI_SECTION, "samples", 1000)
            self.nsteps = self.ini.getint(EMCEE_INI_SECTION, "nsteps", 100)
            self.num_samples = 0

            ndim = len(self.pipeline.varied_params)
            self.p0 = [self.pipeline.randomized_start()
                       for i in xrange(self.nwalkers)]

            self.ensemble = self.emcee.EnsembleSampler(self.nwalkers, ndim,
                                                       log_probability_function,
                                                       pool=self.pool)
            self.sampler = None

    def execute(self):
        if not self.sampler:
            self.sampler = self.ensemble.sample(self.p0,
                                                iterations=self.nsteps,
                                                storechain=True)

        try:
            pos, prob, rstate, extra_info = self.sampler.next()
            self.num_samples += self.nsteps
        except StopIteration:
            raise RuntimeError("Emcee sampler stopped before "
                               "Cosmosis determined convergence")

    def is_converged(self):
        return self.num_samples >= self.samples
