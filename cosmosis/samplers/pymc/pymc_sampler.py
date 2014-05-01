from .. import ParallelSampler
import numpy as np
from cosmosis.runtime.analytics import Analytics
import logging

PYMC_INI_SECTION = "pymc"


class PyMCSampler(ParallelSampler):

    def config(self):
        # lazy pymc import to avoid dependency when using other samplers
        import pymc
        self.pymc = pymc

        self.verbose = logging.getLogger().level > logging.WARNING
        self.interrupted = False

        # load sampling parameters
        self.num_samples = 0

        self.nsteps = self.ini.getint(PYMC_INI_SECTION, "nsteps", 100)
        self.samples = self.ini.getint(PYMC_INI_SECTION, "samples", 1000)
        fburn = self.ini.getfloat(PYMC_INI_SECTION, "burn_fraction", 0.0)
        if not 0.0 <= fburn < 1.0:
            raise RuntimeError("Error: burn_fraction outside "
                               "allowed range: %f" % (fburn,))

        self.nburn = int(fburn*self.samples)
        self.Rconverge = self.ini.getfloat(PYMC_INI_SECTION, "Rconverge", 1.02)

        params = self.define_parameters()

        @pymc.data
        @pymc.stochastic(verbose=self.verbose)
        def data_likelihood(params=params, value=0.0):
            try:
                params = self.pipeline.denormalize_vector(params)
            except ValueError:
                return -np.inf
            like, extra = self.pipeline.likelihood(params)
            return like

        self.mcmc = self.pymc.MCMC(input={'data_likelihood': data_likelihood,
                                          'params': params},
                                   db='ram', verbose=0)

        try:
            covmat = self.load_covariance_matrix()
        except IOError:
            covmat = None

        # determine step method
        self.do_adaptive = self.ini.getboolean(PYMC_INI_SECTION,
                                               "adaptive_mcmc",
                                               False)
        if self.do_adaptive:
            delay = 100
        else:
            delay = 10000000000

        if covmat is not None or self.do_adaptive:
            self.mcmc.use_step_method(self.pymc.AdaptiveMetropolis,
                                      params,
                                      cov=covmat,
                                      interval=self.nsteps,
                                      delay=delay,
                                      verbose=0)
        else:
            for p in params:
                self.mcmc.use_step_method(self.pymc.Metropolis, p, verbose=0)

        self.analytics = Analytics(self.pipeline.varied_params, self.pool)

    def sample(self):
        if self.num_samples < self.nburn:
            steps = min(self.nsteps, self.nburn - self.num_samples)
        else:
            steps = min(self.nsteps, self.samples - self.num_samples)

        # take steps MCMC steps
        self.mcmc.sample(steps, progress_bar=False, tune_throughout=False)
        if self.mcmc._current_iter != steps:
            # user must have pressed ctrl-C,
            # or something else went wrong
            self.interrupted = True

        self.num_samples += self.mcmc._current_iter

        traces = np.array([[param.denormalize(x)
                           for x in self.mcmc.trace(str(param))]
                           for param in self.pipeline.varied_params]).T

        likes = -0.5 * self.mcmc.trace('deviance')[:]
        # TODO: do we output burned samples?
        for trace, like in zip(traces, likes):
            extra = {'LIKE':like}
            self.output.parameters(trace, extra)

        self.analytics.add_traces(traces)

        self.output.log_noisy("Done %d iterations"%self.num_samples)

    def worker(self):
        while not self.is_converged():
            self.sample()

    def execute(self):
        self.sample()

    def is_converged(self):
        #user has pressed Ctrl-C
        if self.interrupted:
            return True
        if self.num_samples >= self.samples:
            return True
        elif self.num_samples > 0 and self.pool is not None:
            return np.all(self.analytics.gelman_rubin() <= self.Rconverge)
        else:
            return False

    def load_covariance_matrix(self):
        covmat_filename = self.ini.get(PYMC_INI_SECTION, "covmat", "").strip()
        if covmat_filename=="":
            return None
        if not os.path.exists(covmat_filename):
            raise ValueError("Covariance matrix %s not found"% covmat_filename)
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
        desired_order = [m.__name__ for m in self.mcmc.stochastics]
        actual_order = [str(m) for m in self.pipeline.varied_params]
        covmat = self.reorder_matrix(actual_order, desired_order, covmat)
        return covmat

    @staticmethod
    def reorder_matrix(old_order, new_order, cov):
            n = len(old_order)
            cov2 = np.zeros((n,n))
            for i in xrange(n):
                    old_i = old_order.index(new_order[i])
                    for j in xrange(n):
                            old_j = old_order.index(new_order[j])
                            cov2[i,j] = cov[old_i, old_j]
            return cov2

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
