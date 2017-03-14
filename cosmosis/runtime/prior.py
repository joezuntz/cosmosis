from . import config
import numpy as np
# It would make sense to import scipy.stats here.
# BUT: I have moved it down to the functions where it is used as a temporary band-aid
# because its presence was causing BLAS or LAPACK problems when it gets imported before 
# the version linked by e.g. multinest.


class Prior(object):
    def __init__(self):
        self.dist=None

    def __call__(self, x):
        if self.dist is None:
            self.setup_dist()
        return self.dist.logpdf(x)

    def sample(self, n=None):
        if n is None:
            n = 1
        if self.dist is None:
            self.setup_dist()
        return self.dist.rvs(size=n)

    def denormalize_from_prior(self, x):
        if self.dist is None:
            self.setup_dist()
        return self.dist.ppf(x)

    @classmethod
    def parse_prior(cls, value):
        prior_type, parameters = value.split(' ', 1)
        prior_type = prior_type.lower()
        try:
            parameters = [float(p) for p in parameters.split()]

            if prior_type.startswith("uni"):
                return UniformPrior(*parameters)
            elif prior_type.startswith("gau") or \
                    prior_type.startswith("nor"):
                return GaussianPrior(*parameters)
            elif prior_type.startswith("exp"):
                return ExponentialPrior(*parameters)
            else:
                raise ValueError("Unable to parse %s as prior" %
                                 (value,))
        except TypeError:
            raise ValueError("Unable to parse %s as prior" %
                             (value,))
    @classmethod
    def load_priors(cls,prior_files):
        priors = {}
        for f in prior_files:
            ini = config.Inifile(f) 
            for option, value in ini:
                if option in priors:
                    raise ValueError("Duplicate prior identified")
                priors[option] = cls.parse_prior(value)

        return priors


class UniformPrior(Prior):
    def __init__(self, a, b):
        self.a=a
        self.b=b
        super(UniformPrior,self).__init__()

    def setup_dist(self):
        import scipy.stats
        self.dist = scipy.stats.uniform(loc=self.a, scale=self.b-self.a)

    def __str__(self):
        return "U({}, {})".format(self.a,self.b)

    def truncate(self, lower, upper):
        a = max(lower, self.a)
        b = min(upper, self.b)
        if a>b:
            raise ValueError("One of your priors is inconsistent with the range described in the values file")
        return UniformPrior(a, b)

class GaussianPrior(Prior):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        super(GaussianPrior,self).__init__()

    def setup_dist(self):
        import scipy.stats
        self.dist = scipy.stats.norm(loc=self.mu, scale=self.sigma)

    def __str__(self):
        return "N({}, {} ** 2)".format(self.mu, self.sigma)

    def truncate(self, lower, upper):
        return TruncatedGaussianPrior(self.mu, self.sigma, lower, upper)


class TruncatedGaussianPrior(Prior):
    def __init__(self, mu, sigma, lower, upper):
        # Stupid scipy handling of limits - they are defined
        # on the normalized space.
        self.lower = lower
        self.upper = upper
        self.mu = mu
        self.sigma = sigma
        super(TruncatedGaussianPrior,self).__init__()

    def setup_dist(self):
        import scipy.stats
        a = (self.lower - self.mu) / self.sigma
        b = (self.upper - self.mu) / self.sigma
        self.dist = scipy.stats.truncnorm(a=a, b=b, loc=self.mu, scale=self.sigma)

    def __str__(self):
        return "N({}, {} ** 2)   [{} < x < {}]".format(self.mu, self.sigma, self.lower, self.upper)


class ExponentialPrior(Prior):
    def __init__(self, beta):
        self.beta = beta
        super(ExponentialPrior,self).__init__()

    def setup_dist(self):
        import scipy.stats
        self.dist = scipy.stats.expon(scale=self.beta)

    def __str__(self):
        return "Expon({})".format(self.beta)

    def truncate(self, lower, upper):
        return TruncatedExponentialPrior(self.beta, lower, upper)


class TruncatedExponentialPrior(Prior):
    def __init__(self, beta, lower, upper):
        self.beta = beta
        self.lower = lower
        self.upper = upper
        super(TruncatedExponentialPrior,self).__init__()

    def setup_dist(self):
        import scipy.stats
        self.dist = scipy.stats.truncexpon(b=(self.upper-self.lower)/self.beta, loc=self.lower, scale=self.beta)

    def __str__(self):
        return "Expon({})   [{} < x < {}]".format(self.beta, self.lower, self.upper)


class DeltaFunctionPrior(Prior):
    "In case this is useful later on"
    def __init__(self, x0):
        self.x0 = x0
        super(DeltaFunctionPrior,self).__init__()

    def setup_dist(self):
        self.dist=True

    def __call__(self, x):
        if x==self.x0:
            return 0.0
        return -np.inf

    def sample(self, n=None):
        if n is None:
            n = 1
        return np.repeat(self.x0, n)

    def __str__(self):
        return "delta({})".format(self.x0)

    def denormalize_from_prior(self, x):
        return self.x0
        
