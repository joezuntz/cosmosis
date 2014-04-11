from sampler import ParallelSampler
import numpy as np
import diagnostics
import logging

PYMC_INI_SECTION = "pymc"


class PyMCSampler(ParallelSampler):

    def config(self):
        # lazy pymc import to avoid dependency when using other samplers
        import pymc
        self.pymc = pymc

        # load sampling parameters
        self.num_samples = 0
        burn = self.ini.getfloat(PYMC_INI_SECTION, "burn", 0.0)
        self.nsteps = self.ini.getint(PYMC_INI_SECTION, "nsteps", 100)
        self.samples = self.ini.getint(PYMC_INI_SECTION, "samples", 1000)
        self.verbose = logging.getLogger().level > logging.WARNING
        self.Rconverge = self.ini.getfloat(PYMC_INI_SECTION, "Rconverge", 1.02)

        params = self.define_parameters()

        @pymc.data
        @pymc.stochastic(verbose=self.verbose)
        def data_likelihood(params=params, value=0.0):
            like, extra = self.pipeline.likelihood(params)
            return like

        self.mcmc = self.pymc.MCMC(input={'data_likelihood': data_likelihood,
                                          'params': params},
                                   db='ram', verbose=0)

        # determine step method
        self.do_adaptive = self.ini.getboolean(PYMC_INI_SECTION,
                                               "adaptive_mcmc",
                                               False)
        if self.do_adaptive:
            covmat = self.load_covariance_matrix()
            self.mcmc.use_step_method(self.pymc.AdaptiveMetropolis,
                                      params,
                                      cov=covmat,
                                      interval=100,
                                      delay=100,
                                      verbose=0)
        else:
            for p in params:
                self.mcmc.use_step_method(self.pymc.Metropolis, p, verbose=0)

        # track trace data for output at end (hack until Output is included)
        #self.trace = { param:np.array([]) for param in self.pipeline.varied_params }
        #self.trace["likelihood"] = np.array([])

        self.diagnostics = diagnostics.Diagnostics(self.pipeline.varied_params, self.pool)

    def sample(self):
        steps = min(self.nsteps, self.samples - self.num_samples)
        self.mcmc.sample(steps, progress_bar=False, tune_throughout=False)
        self.num_samples += steps

        traces = np.array([[param.denormalize(x)
                           for x in self.mcmc.trace(str(param))]
                           for param in self.pipeline.varied_params]).T
        for trace in traces:
            self.output.parameters(trace)

        self.output.log_noisy("Done %d iterations"%self.num_samples)

        self.diagnostics.add_traces(traces)

        #self.trace["likelihood"] = np.append(self.trace["likelihood"], -0.5*np.array(self.mcmc.trace('deviance')[:]))
        #for param in self.pipeline.varied_params:
        #    self.trace[param] = np.append( self.trace[param], np.array([param.denormalize(x) for x in self.mcmc.trace(str(param))]))

        #if self.is_converged():
        #    if self.pool:
        #        suffix = "_%02d.txt" % (self.pool.rank,)
        #    else:
        #        suffix = ".txt"
        #
        #    np.savetxt("likelihood"+suffix, self.trace["likelihood"])
        #    min_index = np.argmin(self.trace["likelihood"])
        #    for param in self.pipeline.varied_params:
        #        print param, self.trace[param][min_index], np.mean(self.trace[param]), np.std(self.trace[param])
        #        np.savetxt(str(param)+suffix, self.trace[param])

    def worker(self):
        while not self.is_converged():
            self.sample()

    def execute(self):
        self.sample()

    def is_converged(self):
        if self.num_samples >= self.samples:
            print "samples done"
            return True
        elif self.num_samples > 0 and self.pool is not None:
            return np.all(self.diagnostics.gelman_rubin() <= self.Rconverge)
        else:
            return False

    def load_covariance_matrix(self):
        covmat_filename = self.ini.get(PYMC_INI_SECTION, "covmat", "")
        covmat = np.loadtxt(covmat_filename)

        if covmat.ndim == 0:
            covmat = covmat.reshape((1, 1))
        elif covmat.ndim == 1:
            covmat = np.diag(covmat**2)

        nparams = len(self.pipeline.varied_params)
        if covmat.shape != (nparams, nparams):
            raise ValueError("The covariance matrix was shape (%d x %d), "
                             "but there are %d varied parameters." %
                             (covmat.shape[0], covmat.shape[1], nparams))

        # normalize covariance matrix
        r = np.array([param.width() for param
                      in self.pipeline.varied_params])
        for i in xrange(covmat.shape[0]):
            covmat[i, :] /= r
            covmat[:, i] /= r

        # reorder to PyMC variable ordering

        return covmat

    # create PyMC parameter objects
    def define_parameters(self):
        ''' Create PyMC parameter objects based on varied params '''
        priors = []
        for param in self.pipeline.varied_params:
            prior = param.prior
            start_value = param.normalize(param.random_point())

            if prior is None or isinstance(prior, UniformPrior):
                # uniform prior
                priors.append(self.pymc.Uniform(str(param),
                                                lower=0.0,
                                                upper=1.0,
                                                value=start_value))
            elif isinstance(prior, GaussianPrior):
                width = param.width()
                mu = (prior.mu-param.limits[0])/width
                tau = width**2/prior.sigma2

                priors.append(self.pymc.Normal(str(param),
                                               mu=mu,
                                               tau=tau,
                                               value=start_value))
            elif isinstance(prior, ExponentialPrior):
                width = param.width()
                priors.append(self.pymc.Exponential(str(param),
                                                    beta=width / prior.beta,
                                                    value=start_value))
            else:
                raise RuntimeError("Unknown prior type in PyMC sampler")
        return priors
