import config
import numpy as np


class Prior(object):
    def __call__(self, p):
        raise NotImplementedError()

    @staticmethod
    def load_priors(prior_files):
        priors = {}
        for f in prior_files:
            ini = config.Inifile(f) 
            for option, value in ini:
                if option in priors:
                    raise ValueError("Duplicate prior identified")

                prior_type, parameters = value.split(' ', 1)
                prior_type = prior_type.lower()

                try:
                    parameters = [float(p) for p in parameters.split()]

                    if prior_type.startswith("uni"):
                        priors[option] = UniformPrior(*parameters)
                    elif prior_type.startswith("gau") or \
                            prior_type.startswith("nor"):
                        priors[option] = GaussianPrior(*parameters)
                    elif prior_type.startswith("exp"):
                        priors[option] = ExponentialPrior(*parameters)
                    else:
                        raise ValueError("Unable to parse %s as prior" %
                                         (value,))
                except TypeError:
                    raise ValueError("Unable to parse %s as prior" %
                                     (value,))
        return priors


class UniformPrior(Prior):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        if self.a <= x <= self.b:
            return 0
        else:
            return -np.inf

    def sample(self, n=None):
        return np.random.uniform(self.a, self.b, n)



class GaussianPrior(Prior):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma2 = sigma**2
        self.norm=0.5*np.log(2*np.pi*self.sigma2)

    def __call__(self, x):
        return -0.5 * (x-self.mu)**2 / self.sigma2 - self.norm

    def sample(self, n=None):
        return np.random.normal(self.mu, self.sigma2**0.5, n)


class ExponentialPrior(Prior):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, x):
        if x > 0:
            return -x/self.beta
        else:
            return -np.inf

    def sample(self, n=None):
        return np.random.exponential(self.beta, n)
