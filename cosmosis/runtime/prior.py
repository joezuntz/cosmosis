from . import config
import numpy as np
import scipy.stats

class Prior(object):
    def __call__(self, x):
        return self.dist.logpdf(x)

    def sample(self, n=None):
        if n is None:
            n = 1
        return self.dist.rvs(size=n)

    def denormalize_from_prior(self, x):
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
        self.dist = scipy.stats.uniform(loc=a, scale=b-a)

    def __str__(self):
        dist_a = self.dist.kwds['loc']
        dist_b = dist_a + self.dist.kwds['scale']
        return "U({}, {})".format(dist_a,dist_b)

    def truncate(self, lower, upper):
        dist_a = self.dist.kwds['loc']
        dist_b = dist_a + self.dist.kwds['scale']
        a = max(lower, dist_a)
        b = min(upper, dist_b)
        if a>b:
            raise ValueError("One of your priors is inconsistent with the range described in the values file")
        return UniformPrior(a, b)

class GaussianPrior(Prior):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dist = scipy.stats.norm(loc=mu, scale=sigma)

    def __str__(self):
        return "N({}, {} ** 2)".format(self.mu, self.sigma)

    def truncate(self, lower, upper):
        return TruncatedGaussianPrior(self.mu, self.sigma, lower, upper)


class TruncatedGaussianPrior(Prior):
    def __init__(self, mu, sigma, lower, upper):
        # Stupid scipy handling of limits - they are defined
        # on the normalized space.
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        self.lower = lower
        self.upper = upper
        self.mu = mu
        self.sigma = sigma

        self.dist = scipy.stats.truncnorm(a=a, b=b, loc=mu, scale=sigma)

    def __str__(self):
        return "N({}, {} ** 2)   [{} < x < {}]".format(self.mu, self.sigma, self.lower, self.upper)


class ExponentialPrior(Prior):
    def __init__(self, beta):
        self.beta = beta
        self.dist = scipy.stats.expon(scale=beta)

    def __str__(self):
        return "Expon({})".format(self.beta)

    def truncate(self, lower, upper):
        return TruncatedExponentationPrior(self.dist.kwds['scale'], lower, upper)


class TruncatedExponentationPrior(Prior):
    def __init__(self, beta, lower, upper):
        self.beta = beta
        self.lower = lower
        self.upper = upper
        self.dist = scipy.stats.truncexpon(b=(upper-lower)/beta, loc=lower, scale=beta)

    def __str__(self):
        return "Expon({})   [{} < x < {}]".format(self.beta, self.lower, self.upper)


class DeltaFunctionPrior(Prior):
    "In case this is useful later on"
    def __init__(self, x0):
        self.x0 = x0

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
