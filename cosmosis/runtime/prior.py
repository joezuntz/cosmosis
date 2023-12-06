#coding: utf-8


u"""Implementation of :class:`Prior` and the concrete probability distribution specializations.

The specializations are :class:`UniformPrior`, :class:`GaussianPrior`,
:class:`TruncatedGaussianPrior`, :class:`ExponentialPrior`,
:class:`TruncatedExponentialPrior`, :class:`TruncatedOneoverxPrior`,
and :class:`DeltaFunctionPrior`.

Applications can get by only knowing about the :class:`Prior` superclass.

"""

from . import config
import numpy as np
import math
from scipy import interpolate
import copy
import scipy.stats

class Prior(object):

    u"""This class serves as an abstract base for possible prior distributions, and provides a concrete distribution factory method.

    At high level, each concrete prior distribution is a function which
    takes a value and returns a probability of that value occurring;
    actually, the logarithm of the probability density is returned as this
    is more useful in a Bayesian framework where the value of evidence
    takes the form of a log-likelihood.  There are additionally methods
    which

    * Return a value corresponding to a given cumulated probability,
      i.e. the inverse of the prior distribution.

    * Obtain the value of a number of random samples from the
      distribution.

    * Optionally, a method for returning a truncated version of the
      distribution.

    The application calls a single function, :func:`Prior.load_priors`, to
    obtain a concrete set of all the priors the user specifies in the
    configuration files, and then abstractly calls the above methods on
    those priors to get the work done.

    """

    def __init__(self, rv):
        self.rv = rv

    def __call__(self, x):
        u"""Return the logarithm of the probability density, a constant value independent of `x` so long as `x` is in the proper range."""
        return self.rv.logpdf(x)
    
    def sample(self, n):
        return self.rv.rvs(n)
    
    def denormalize_from_prior(self, y, gradient=False):
        val =  self.rv.isf(1 - y)
        if gradient:
            deriv = 1 / self.rv.pdf(val)
            return val, deriv
        else:
            return val
        
    def log_pdf_derivative(self, x):
        """
        Subclasses must return the derivative of the log(pdf) at x
        """
        pass
    
    def transform_to_unbounded(self, x):
        """
        Return that parameter transformed to a new distribution defined on the full range -inf to inf,
        and the jacobian of that transformation.

        We do this by transforming first to the range 0 to 1, and then applying the logit function
        x -> log(x / (1 - x))

        For distributions that are already defined on the full range, this is just the identity transform
        and a jacobian of unity.
        """
        return x, 1
    

    @classmethod
    def parse_prior(cls, value):
        u"""Produce a concrete :class:`Prior` object based on a line of a .ini ifile.

        This is part of the implementation of :func:`load_priors` and
        should be considered private to the class.

        """
        prior_type, parameters = value.split(' ', 1)
        prior_type = prior_type.lower()
        try:
            # Most of the priors require float
            # parameters, but not all of them -
            # the InverseTransform prior needs
            # a string argument (file name).
            parameters_flt = []
            for p in parameters.split():
                try:
                    p = float(p)
                except:
                    pass
                parameters_flt.append(p)
            parameters = parameters_flt

            if prior_type.startswith("uni"):
                return UniformPrior(*parameters)
            elif prior_type.startswith("gau") or \
                    prior_type.startswith("nor"):
                return GaussianPrior(*parameters)
            elif prior_type.startswith("exp"):
                return ExponentialPrior(*parameters)
            elif  prior_type.startswith("one"):
                return TruncatedOneoverxPrior(*parameters)
            elif prior_type.startswith("tab") or \
                    prior_type.startswith("loa"):
                return TabulatedPDF(*parameters)
            else:
                raise ValueError("Unable to parse %s as prior" %
                                 (value,))
        except TypeError:
            raise ValueError("Unable to parse %s as prior" %
                             (value,))

    @classmethod
    def load_priors(cls,prior_files):
        u"""Produce a dictionary of `(section,name)` -> :class:`Prior` objects as per the instructions in `prior_files`.

        The dictionary values are concrete :class:`Prior` types, and this
        method is the applicationʼs sole constructor of :class:`Prior`
        objects.

        Can also pass in an ini file directly
        """
        priors = {}
        for f in prior_files:
            if isinstance(f, config.Inifile):
                ini = f
            else:
                ini = config.Inifile(f) 
            for option, value in ini:
                if option in priors:
                    raise ValueError("Duplicate prior identified")
                priors[option] = cls.parse_prior(value)

        return priors

class TabulatedPDF(Prior):

    u"""Load from a 2-column ASCII table containing values for x, pdf(x).
    
    pdf(x) not interpolated (assumed to be zero) outisde of range of x

    Heavily 'inspired' by DistDeviate from GalSim
    """

    def __init__(self, function_filename=None, lower=None, upper=None):
        u"""Create an object representing the distribution specified in function_filename"""
        

        self.function_filename = function_filename
        # Basic input checking and setups
        if function_filename is None:
            raise TypeError('You must specify a function_filename for TabulatedPDF!')

        # Load the probability distribution function, pdf(x)
        xarray, pdf = np.loadtxt(function_filename, unpack=True)
        
        if lower==None:
            self.lower = xarray.min()
        else:
            self.lower = lower

        if upper==None:
            self.upper = xarray.max()
        else:
            self.upper = upper

        pdf = pdf[(xarray >= self.lower)*(xarray <= self.upper)]
        xarray = xarray[(xarray >= self.lower)*(xarray <= self.upper)]

        # Set up pdf, so cumsum basically does a cumulative trapz integral
        # On Python 3.4, doing pdf[1:] += pdf[:-1] the last value gets messed up.
        # Writing it this way works.  (Maybe slightly slower though, so if we stop
        # supporting python 3.4, consider switching to the += version.)
        pdf_x = copy.copy(pdf)
        pdf[1:] = pdf[1:] + pdf[:-1]
        pdf[1:] *= np.diff(xarray)
        pdf[0] = 0

        # Check that the probability is nonnegative
        if not np.all(pdf >= 0.):
            raise ValueError('Negative probability found in TabulatedPDF.',function)

        # Compute the cumulative distribution function = int(pdf(x),x)
        cdf = np.cumsum(pdf)

        # Quietly renormalize the probability if it wasn't already normalized
        totalprobability = cdf[-1]
        cdf /= totalprobability

        self.inverse_cdf_interp = interpolate.interp1d(cdf, xarray, kind='linear')
        self.cdf_interp = interpolate.interp1d(xarray, cdf, kind='linear')
        self.pdf_interp = interpolate.interp1d(xarray, pdf_x, kind='linear')

    def __call__(self, x):
        u"""Return the logarithm of the probability density."""
        if x<self.lower:
            return -np.inf
        elif x>self.upper:
            return -np.inf
        return np.log(self.pdf_interp(x))

    def sample(self, n):
        u"""Use interpolation of inverse CDF to give us `n` random samples from a the distribution."""
        if n is None:
            n = 1 
        return self.inverse_cdf_interp(np.random.rand(n))

    def denormalize_from_prior(self, y):
        u"""Get the value for which the cumulated probability is `y`."""
        return self.inverse_cdf_interp(y)

    def __str__(self):
        u"""Tersely describe ourself to a human mathematician."""
        return "Tabulated transform from {0} on range [{1}, {2}]".format(self.function_filename, self.lower, self.upper)

    def truncate(self, lower, upper):
        u"""Return a new distribution whose range is the intersection of ours with [`lower`, `upper`].

        A :class:`ValueError` will be thrown if the arguments supplied do
        not make sense.

        """
        lower = max(self.lower, lower)
        upper = min(self.upper, upper)
        if lower>upper:
            raise ValueError("One of your priors is inconsistent with the range described in the values file")
        return TabulatedPDF(self.function_filename, lower, upper)


class UniformPrior(Prior):

    u"""Statistical distribution in which all values in a range have equal probability of occurring."""
    
    def __init__(self, a, b):
        u"""Create an object which encapsulates a Uniform distribution over the interval [`a`, `b`]."""
        self.a = a
        self.b = b
        rv = scipy.stats.uniform(a, b - a)
        super().__init__(rv)


    def __str__(self):
        u"""Tersely describe ourself to a human mathematician."""
        return "U({}, {})".format(self.a, self.b)


    def truncate(self, lower, upper):
        u"""Return a new Uniform distribution whose range is the intersection of ours with [`lower`, `upper`].

        A :class:`ValueError` will be thrown if the arguments supplied do
        not make sense.

        """
        a = max(lower, self.a)
        b = min(upper, self.b)
        if a>b:
            raise ValueError("One of your priors is inconsistent with the range described in the values file")
        return UniformPrior(a, b)


    def log_pdf_derivative(self, x):
        return 0.0


class GaussianPrior(Prior):

    u"""Encapsulation of a Normal distribution function."""

    def __init__(self, mu, sigma):
        u"""Make a Normal distribution object with mean `mu` and deviation `sigma`."""
        self.mu = mu
        self.sigma = sigma
        rv = scipy.stats.norm(mu, sigma)
        super().__init__(rv)

    def __str__(self):
        u"""Tersely describe ourself to a human mathematician."""
        return "N({}, {} ** 2)".format(self.mu, self.sigma)

    def truncate(self, lower, upper):
        u"""Return a :class:`TruncatedGaussianPrior` object, with our mean and variance and the given `lower` and `upper` limits of non-zero probability."""
        return TruncatedGaussianPrior(self.mu, self.sigma, lower, upper)

    def log_pdf_derivative(self, x):
        return -x * (x - self.mu) / self.sigma**2

    
    
class TruncatedGaussianPrior(Prior):

    u"""A Normal distribution, except that probabilities outside of a specified range are truncated to zero."""

    def __init__(self, mu, sigma, lower, upper):
        u"""Get a concrete object representing a distribution with mode at `mu`, ‘shape width’ `sigma`, and limits of non-zero probability `lower` and `upper`."""
        self.lower = lower
        self.upper = upper
        self.mu = mu
        self.sigma = sigma
        self.a = (lower-mu)/sigma
        self.b = (upper-mu)/sigma
        rv = scipy.stats.truncnorm(self.a, self.b, mu, sigma)
        super(TruncatedGaussianPrior,self).__init__(rv)


    def __str__(self):
        u"""Return a terse description of ourself."""
        return "N({}, {} ** 2)   [{} < x < {}]".format(self.mu, self.sigma, self.lower, self.upper)

    def truncate(self, lower, upper):
        u"""Produce a new :class:`Prior` object representing a Normal distribution whose range of non-zero probability is the intersection of our own range with [`lower`, `upper`]."""
        lower = max(self.lower, lower)
        upper = min(self.upper, upper)
        return TruncatedGaussianPrior(self.mu, self.sigma, lower, upper)

    def log_pdf_derivative(self, x):
        return -x * (x - self.mu) / self.sigma**2

class ExponentialPrior(Prior):

    u"""The Exponential Distribution, zero probability of any value less than zero."""

    def __init__(self, beta):
        u"""Create object representing distribution with ‘width’ `beta`."""
        self.beta = beta
        rv = scipy.stats.expon(scale=beta)
        super().__init__(rv)
        
    def __str__(self):
        u"""Give a terse description of ourself."""
        return "Expon({})".format(self.beta)

    def truncate(self, lower, upper):
        u"""Return a :class:`Prior` object representing the distribution you get when you take the current distribution but set probability to zero everywhere outside the range [`lower`, `upper`], and re-normalize."""
        return TruncatedExponentialPrior(self.beta, lower, upper)

    def log_pdf_derivative(self, x):
        if self.x < 0:
            return np.nan
        return -self.beta




class TruncatedExponentialPrior(Prior):

    u"""Like the Exponential prior, but truncated."""

    def __init__(self, beta, lower, upper):
        u"""Create a distribution with ‘half-life’ `beta`, `lower` bound of non-zero probability, and `upper` bound."""
        self.beta = beta
        if lower<0:
            lower = 0.0
        self.lower = lower
        self.upper = upper
        rv = scipy.stats.truncexpon(b=(upper - lower)/beta, loc=lower, scale=beta)
        super().__init__(rv)
    
    def __str__(self):
        u"""Give a terse description of ourself."""
        return "Expon({})   [{} < x < {}]".format(self.beta, self.lower, self.upper)

    def truncate(self, lower, upper):
        u"""Return a :class:`Prior` like ourself but with range of non-zero probability further restricted to `lower` and `upper` bounds."""
        lower = max(lower, self.lower)
        upper = min(upper, self.upper)
        return TruncatedExponentialPrior(self.beta, lower, upper)

    def log_pdf_derivative(self, x):
        if x < self.lower or x > self.upper:
            return np.nan
        return -self.beta


class TruncatedOneoverxPrior(Prior):
    u"""The 1/x distribution, which is a uniform distribution in ln(x). As ln(x) diverges in both directions, we only provide the truncated option."""

    def __init__(self, lower, upper):
        u"""Create a distribution with 1/x, `lower` bound of non-zero probability, and `upper` bound."""
        if lower<=0:
            lower = np.nextafter(0, 1)
        self.lower = lower
        self.upper = upper
        rv = scipy.stats.loguniform(lower, upper)
        super().__init__(rv)

    def __str__(self):
        u"""Give a terse description of ourself."""
        return "1/x   [{} < x < {}]".format(self.lower, self.upper)

    def truncate(self, lower, upper):
        u"""Return a :class:`Prior` like ourself but with range of non-zero probability further restricted to `lower` and `upper` bounds."""
        lower = max(lower, self.lower)
        upper = min(upper, self.upper)
        return TruncatedOneoverxPrior(lower, upper)
    
    def log_pdf_derivative(self, x):
        if x < self.lower or x > self.upper:
            return np.nan
        return -1/x**2



class DeltaFunctionPrior(Prior):

    u"""Probability distribution with non-zero probability at a single value."""

    # In case this is useful later on
    def __init__(self, x0):
        u"""Create object with atom of probability at `x0`."""
        self.x0 = x0
        super().__init__(None)

    def __call__(self, x):
        u"""The log-density is zero when `x` is `x0`, minus infinity otherwise."""
        if x==self.x0:
            return 0.0
        return -np.inf

    def sample(self, n=None):
        u"""Just return `x0` `n` times."""
        if n is None:
            n = 1
        return np.repeat(self.x0, n)

    def __str__(self):
        u"""Terse description of ourself."""
        return "delta({})".format(self.x0)

    def denormalize_from_prior(self, x):
        u"""Just return `x0`; itʼs the only value with any probability."""
        return self.x0
    
    def log_pdf_derivative(self, x):
        return np.nan
        

