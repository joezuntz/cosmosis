from . import config
import numpy as np
import math


class Prior(object):
    def __init__(self):
        pass

    def sample(self, n):
        if n is None:
            n = 1
        Y = np.random.uniform(0., 1.0, n)
        return np.array([self.denormalize_from_prior(y) for y in Y])

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
        self.norm = -np.log(b-a)
        super(UniformPrior,self).__init__()

    def __call__(self, x):
        if x<self.a or x>self.b:
            return -np.inf
        return self.norm

    def sample(self, n):
        return np.random.uniform(self.a, self.b, n)

    def denormalize_from_prior(self, y):
        if y<0.0:
            x = np.nan
        elif y>1.0:
            x = np.nan
        else:
            x = y * (self.b-self.a) + self.a
        return x


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
        self.sigma2 = sigma**2
        self.norm=0.5*np.log(2*np.pi*self.sigma2)
        super(GaussianPrior,self).__init__()

    def __call__(self, x):
        return -0.5 * (x-self.mu)**2 / self.sigma2 - self.norm

    def sample(self, n):
        if n is None:
            n = 1        
        return np.random.normal(self.mu, self.sigma, n)

    def denormalize_from_prior(self, y):
        x_normal = normal_ppf(y)
        x = x_normal*self.sigma + self.mu
        return x

    def __str__(self):
        return "N({}, {} ** 2)".format(self.mu, self.sigma)

    def truncate(self, lower, upper):
        return TruncatedGaussianPrior(self.mu, self.sigma, lower, upper)

    
    
class TruncatedGaussianPrior(Prior):
    def __init__(self, mu, sigma, lower, upper):
        self.lower = lower
        self.upper = upper
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.a = (lower-mu)/sigma
        self.b = (upper-mu)/sigma
        self.phi_a = normal_cdf(self.a)
        self.phi_b = normal_cdf(self.b)
        self.norm = np.log(self.phi_b - self.phi_a) + 0.5*np.log(2*np.pi*self.sigma2)
        super(TruncatedGaussianPrior,self).__init__()
        
    def __call__(self, x):
        if x<self.lower:
            return -np.inf
        elif x>self.upper:
            return -np.inf
        return -0.5 * (x-self.mu)**2 / self.sigma2 - self.norm

    def denormalize_from_prior(self, y):
        x_normal = truncated_normal_ppf(y, self.a, self.b)
        x = x_normal*self.sigma + self.mu
        return x

    def __str__(self):
        return "N({}, {} ** 2)   [{} < x < {}]".format(self.mu, self.sigma, self.lower, self.upper)

    def truncate(self, lower, upper):
        lower = max(self.lower, lower)
        upper = min(self.upper, upper)
        return TruncatedGaussianPrior(self.mu, self.sigma, lower, upper)

class ExponentialPrior(Prior):
    def __init__(self, beta):
        self.beta = beta
        self.log_beta = np.log(beta)
        super(ExponentialPrior,self).__init__()
    
    def __call__(self, x):
        if x<0.0:
            return -np.inf
        return -x/self.beta - self.log_beta
    
    def sample(self,n):
        if n is None:
            n = 1        
        return np.random.exponential(self.beta,n)

    def denormalize_from_prior(self, y):
        #y = 1 - exp(-x/beta)
        #exp(-x/beta) = 1 - y
        #-x/beta = log(1-y)
        #x = -beta * log(1-y)
        return -self.beta * np.log(1-y)
        
    def __str__(self):
        return "Expon({})".format(self.beta)

    def truncate(self, lower, upper):
        return TruncatedExponentialPrior(self.beta, lower, upper)

class TruncatedExponentialPrior(Prior):
    def __init__(self, beta, lower, upper):
        self.beta = beta
        self.log_beta = np.log(beta)
        if lower<0:
            lower = 0.0
        self.lower = lower
        self.upper = upper
        self.a = lower/beta
        self.b = upper/beta
        self.phi_a = exponential_cdf(self.a)
        self.phi_b = exponential_cdf(self.b)
        self.norm = np.log(self.phi_b - self.phi_a) + self.log_beta
        super(TruncatedExponentialPrior,self).__init__()
    
    def __call__(self, x):
        #(1/beta)*exp(-x/beta)
        if x<self.lower:
            return -np.inf
        if x>self.upper:
            return -np.inf
        return -x/self.beta - self.norm
    
    def denormalize_from_prior(self, y):
        x_normal = truncated_exponential_ppf(y, self.a, self.b)
        x = x_normal * self.beta
        return x

    def __str__(self):
        return "Expon({})   [{} < x < {}]".format(self.beta, self.lower, self.upper)

    def truncate(self, lower, upper):
        lower = max(lower, self.lower)
        upper = min(upper, self.upper)
        return TruncatedExponentialPrior(self.beta, lower, upper)


class DeltaFunctionPrior(Prior):
    "In case this is useful later on"
    def __init__(self, x0):
        self.x0 = x0
        super(DeltaFunctionPrior,self).__init__()

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
        


# Helper functions
def inverse_function(f, y, xmin, xmax, *args, **kwargs):
    "Find x in [xmin,xmax] such that f(x)==y, in 1D, with bisection"
    import scipy.optimize
    def g(x):
        return f(x, *args, **kwargs) - y
    x = scipy.optimize.bisect(g, xmin, xmax)
    return x

SQRT2 = np.sqrt(2.)
def normal_cdf(x):
#    return 0.5*math.erf(x) + 0.5
    return 0.5*(math.erf(x/SQRT2) + 1)


def normal_ppf(y):
    if y<0:
        return np.nan
    if y>1:
        return np.nan
    return inverse_function(normal_cdf, y, -20.0, 20.0)

def truncated_normal_cdf(x, a, b):
    if x<a:
        return np.nan
    if x>b:
        return np.nan
    phi_a = normal_cdf(a)
    phi_b = normal_cdf(b)
    phi_x = normal_cdf(x)
    return (phi_x - phi_a) / (phi_b - phi_a)

def truncated_normal_ppf(y, a, b):
    if y<0:
        return np.nan
    if y>1:
        return np.nan
    return inverse_function(truncated_normal_cdf, y, a, b, a, b)

def exponential_cdf(x):
    if x<0.0:
        return np.nan
    return 1 - np.exp(-x)

def truncated_exponential_cdf(x, a, b):
    if x<a:
        return np.nan
    if x>b:
        return np.nan
    phi_a = exponential_cdf(a)
    phi_b = exponential_cdf(b)
    phi_x = exponential_cdf(x)
    return (phi_x - phi_a) / (phi_b - phi_a)

def exponential_ppf(y):
    #y = 1 - exp(-x)
    # exp(-x) = 1-y
    # x = -log(1-y)
    return -np.log(1-y)

def truncated_exponential_ppf(y, a, b):
    if y<0:
        return np.nan
    if y>1:
        return np.nan    
    return inverse_function(truncated_exponential_cdf, y, a, b, a, b)
