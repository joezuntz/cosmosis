from sampler import Sampler
import config
import pipeline
import prior

import pymc
import np


PYMC_INI_SECTION = "pymc"


class PyMCSampler(ParallelSampler):

    def config(self):
        # load sampling parameters
        burn = self.ini.getfloat(PYMC_INI_SECTION, "burn", 0.0)
        self.nsteps = self.ini.getint(PYMC_INI_SECTION, "nsteps", 100)
        self.samples = self.ini.getint(PYMC_INI_SECTION, "samples", 1000)
        
        params = self.define_parameters()

        @pymc.data
        @pymc.stochastic(verbose=verbose)
        def data_likelihood(params = params, value = 0.0):
            like, extra = self.pipeline.likelihood(p)
            return like
        
        self.mcmc = pymc.MCMC(model={'data_likelihood':data_likelihood,
                                     'params':params}, 
                              db='ram', verbose=2)

        # determine step method
        self.do_adaptive = self.ini.getboolean(PYMC_INI_SECTION, 
                                               "adaptive_mcmc",
                                               False)
        if self.do_adaptive:
            covmat = self.load_covariance_matrix()
            self.mcmc.use_step_method(pymc.AdaptiveMetropolis,
                                      params,
                                      cov=covmat,
                                      interval=100,
                                      delay=100,
                                      verbose=0) 
        else:
            for p in params:
                self.mcmc.use_step_method(pymc.Metropolis, p, verbose=0)

        # create Diagnostics object

    def sample(self):
        mcmc.sample(self.nsteps, progress_bar=False, tune_throughout=False)

    def worker(self):
        while True:
            self.sample()
            # send data to root

    def execute(self):
        self.sample()

        # add traces to diagnostics    
        # output traces
        # collect trace information from other workers for diagnostics

    def is_converged(Self):
        return self.num_samples > self.samples

    def load_covariance_matrix(self):
        covmat_filename = self.ini.get(PYMC_INI_SECTION, "covmat", "")
        covmat = np.loadtxt(covmat_filename)

        if covmat.ndim == 0:
            covmat = covmat.reshape((1,1))
        elif covmat.ndim == 1:
            covmat = np.diag(covmat**2)

        nparams = len(self.pipeline.varied_params)
        if covmat.shape != (nparams, nparams):
            raise ValueError("The covariance matrix was shape (%d x %d), but there are %d varied parameters." % (*covmat.shape, nparams))

        # normalize covariance matrix
        r = np.array([param.width() for param in self.pipeline.varied_params])
        for i in xrange(covmat.shape[0]):
            covmat[i,:] /= r
            covmat[:,i] /= r

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
                priors.append(pymc.Uniform(str(param),
                                           lower = 0.0,
                                           upper = 1.0,
                                           value = start_value)))
            elif isinstance(prior, GaussianPrior):
                width = param.width()
                mu = (prior.mu-param.limits[0])/width
                tau = width**2/prior.sigma2

                priors.append(pymc.Normal(str(param),
                                          mu = mu,
                                          tau = tau,
                                          value = start_value))
            elif isinstance(prior, ExponentialPrior):
                width = param.width()
                priors.append(pymc.Exponential(str(param),
                                               beta = width / prior.beta,
                                               value = start_value))
            else:
                raise RuntimeError("Unknown prior type in PyMC sampler")
        return priors
