import config
import numpy as np


class Prior(object):
    def __call__(self, p):
        raise NotImplementedError()

    def load_priors(prior_files):
        priors = {}
        for f in prior_files:
            with Inifile(f) as ini:
                for option, value in ini:
                    if option in prior:
                        raise ValueError("Duplicate prior identified")

                    prior_type, parameters = value.split(' ', 1)
                    prior_type = prior_type.lower()

                    try:
                        parameters = [float(p) for p in parameters]

                        if prior_type.startwith("uni"):
                            prior[option] = UniformPrior(*parameters)
                        elif prior_type.startswith("gau") or \
                                prior_type.startswith("nor"):
                            prior[option] = GaussianPrior(*parameters)
                        elif prior_type.startswith("exp"):
                            prior[option] = ExponentialPrior(*parameters)
                        else:
                            raise ValueError("Unable to parse %s as prior" % (value,))
                    except TypeError:
                        raise ValueError("Unable to parse %s as prior" % (value,))
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


class GaussianPrior(Prior):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma2 = sigma**2

    def __call__(self, x):
        return -0.5 * (x-self.mu)**2 / self.sigma2


class ExponentialPrior(Prior):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, x):
        if x > 0:
            return -x/self.beta
        else:
            return -np.inf
