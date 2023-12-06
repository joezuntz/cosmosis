from .jax import tools as jax_tools
import jax.numpy as jnp
from functools import partial
import jax
import numpy as np
from .datablock import names, SectionOptions
from .runtime import FunctionModule



MISSING = "if_you_see_this_there_was_a_mistake_creating_a_gaussian_likelihood"

# we do not vary with respect to the data x location, just the theory y values
# could think about the theory x values - probably don't need those in most cases?
@partial(jax.jit, static_argnums=(0,))
def generate_theory_points(data_x, theory_x, theory_y):
    "Generate theory predicted data points by interpolation into the theory"
    jax.lax.stop_gradient(data_x)
    jax.lax.stop_gradient(theory_x)
    s = jax_tools.InterpolatedUnivariateSpline(theory_x, theory_y)
    return s(data_x)


@jax.jit
def _do_likelihood(theory_x, theory_y, data_x, data_y, inv_cov, log_det):
    # These should all be constant for us
    jax.lax.stop_gradient(theory_x)
    jax.lax.stop_gradient(data_x)
    jax.lax.stop_gradient(data_y)
    jax.lax.stop_gradient(inv_cov)
    jax.lax.stop_gradient(log_det)
    #get data x by interpolation
    x = jnp.atleast_1d(generate_theory_points(data_x, theory_x, theory_y))
    mu = jnp.atleast_1d(data_y)

    #gaussian likelihood
    d = x-mu
    chi2 = jnp.einsum('i,ij,j', d, inv_cov, d)
    like = -0.5*chi2

    norm = -0.5 * log_det
    like += norm

    return like, chi2, x

_do_likelihood_jac = jax.jacrev(_do_likelihood, argnums=[1])


class GaussianLikelihood:
    """
    Gaussian likelihood with a fixed covariance.  
    
    Subclasses must override build_data and build_covariance,
    e.g. to load from file.
    """
    x_section = MISSING
    x_name    = MISSING
    y_section = MISSING
    y_name    = MISSING
    like_name = MISSING

    #Set this to False to load the covariance at 
    #each cosmology instead of once at the start
    constant_covariance = True

    def __init__(self, options):
        self.options=options
        self.data_x, self.data_y = self.build_data()
        self.likelihood_only = options.get_bool('likelihood_only', False)

        if self.constant_covariance:
            self.cov = self.build_covariance()
            self.inv_cov = self.build_inverse_covariance()

            if not self.likelihood_only:
                self.chol = jnp.linalg.cholesky(self.cov)


            # We may want to include the normalization of the likelihood
            # via the log |C| term.
            include_norm = self.options.get_bool("include_norm", False)
            if include_norm:
                # We have no datablock here so we don't want to call any subclass method
                self.log_det_constant = GaussianLikelihood.extract_covariance_log_determinant(self,None)
                print("Including -0.5*|C| normalization in {} likelihood where |C| = {}".format(self.like_name, self.log_det_constant))
            else:
                self.log_det_constant = 0.0

        #Interpolation type, when interpolating into theory vectors
        self.kind = self.options.get_string("kind", "cubic")

        #Allow over-riding where the inputs come from in 
        #the options section
        if options.has_value("x_section"):
            self.x_section = options['x_section']
        if options.has_value("y_section"):
            self.y_section = options['y_section']
        if options.has_value("x_name"):
            self.x_name = options['x_name']
        if options.has_value("y_name"):
            self.y_name = options['y_name']
        if options.has_value("like_name"):
            self.like_name = options['like_name']



    def build_data(self):
        """
        Override the build_data method to read or generate 
        the observed data vector to be used in the likelihood
        """
        raise RuntimeError("Your Gaussian covariance code needs to "
            "over-ride the build_data method so it knows how to "
            "load the observed data")
        #using info in self.options,
        #like filenames etc,
        #build x to which we must interpolate
        #return x, y
    
    def build_covariance(self):
        """
        Override the build_covariance method to read or generate 
        the observed covariance
        """
        raise RuntimeError("Your Gaussian covariance code needs to "
            "over-ride the build_covariance method so it knows how to "
            "load the data covariance (or set constant_covariance=False and "
            "over-ride the extract_covariance method)")

        #using info in self.options,
        #like filenames etc,
        #build covariance

    def build_inverse_covariance(self):
        """
        Override the build_inverse_covariance method to change
        how the inverse is generated from the covariance.

        When the covariance is generated from a suite of simulations,
        for example, the simple inverse is not the best estimate.

        """
        # inverse of symmetric matrix should remain symmetric
        if jnp.allclose(self.cov, self.cov.T):
            return jnp.linalg.pinv(self.cov, hermitian=True)
        return jnp.linalg.inv(self.cov)


    def cleanup(self):
        """
        You can override the cleanup method if you do something 
        unusual to get your data, like open a database or something.
        It is run just once, at the end of the pipeline.
        """
        pass
    
    def extract_covariance(self, block):
        """
        Override this and set constant_covariance=False
        to enable a cosmology-dependent covariance.

        Load the covariance from the block here.
        """
        raise RuntimeError("You need to implement the method "
            "'extract_covariance' if you set constant_covariance=False "
            "in a gaussian likelihood")

    def extract_inverse_covariance(self, block):
        """
        Override this and set constant_covariance=False
        to enable a cosmology-dependent inverse
        covariance matrix.

        By default the inverse is just directly calculated from
        the covariance, but you might have a different method.
        """
        return jnp.linalg.inv(self.cov)

    def extract_covariance_log_determinant(self, block):
        """
        If you are using a varying covariance we have to account
        for the dependence of the covariance matrix on parameters
        in the likelihood.

        Override this method if you have a faster way to get |C|
        rather than just taking the determinant of directly (e.g.
        if you know it is diagonal or block-diagonal).

        Since we know that we must have the inverse covariance,
        whereas the covariance itself is optional, we use the former
        in the default implementation.
        """
        sign, log_inv_det = jnp.linalg.slogdet(self.inv_cov)
        log_det = -log_inv_det
        return log_det

    def do_likelihood(self, block):
        theory_x, theory_y = self.extract_theory_samples(block)
        inv_cov = self.inv_cov
        log_det = self.log_det_constant

        like, chi2, x = _do_likelihood(theory_x, theory_y, self.data_x, self.data_y, inv_cov, log_det)

        r = _do_likelihood_jac(theory_x, theory_y, self.data_x, self.data_y, inv_cov, log_det)
        # dlike_dx = r[0][0]
        # dlike_dy = r[0][1]
        dlike_dy = r[0][0]

        # block[names.likelihoods + "_derivative", f"{self.like_name}_like_by_{self.x_section}.{self.x_name}"] = np.array(dlike_dx)
        # block[names.likelihoods + "_derivative", f"{self.like_name}_like_by_{self.y_section}.{self.y_name}"] = np.array(dlike_dy)
        # block.put_derivative(names.likelihoods, self.like_name+"_like", self.x_section, self.x_name, dlike_dx)
        block.put_derivative(names.likelihoods, self.like_name+"_like", self.y_section, self.y_name, dlike_dy)


        #Now save the resulting likelihood
        block[names.likelihoods, self.like_name+"_LIKE"] = float(like)

        #It can be useful to save the chi^2 as well as the likelihood,
        #especially when the covariance is non-constant.
        block[names.data_vector, self.like_name+"_CHI2"] = float(chi2)
        block[names.data_vector, self.like_name+"_N"] = self.data_y.size

        if self.likelihood_only:
            return

        # Save various other quantities
        block[names.data_vector, self.like_name+"_LOG_DET"] = float(log_det)
        block[names.data_vector, self.like_name+"_NORM"] = -0.5 * float(log_det)

        #And also the predicted data points - the vector of observables 
        # that in a fisher approch we want the derivatives of.
        #and inverse cov mat which also goes into the fisher matrix.
        block[names.data_vector, self.like_name + "_theory"] = np.array(x)
        block[names.data_vector, self.like_name + "_data"] = np.array(self.data_y)
        block[names.data_vector, self.like_name + "_inverse_covariance"] = np.array(inv_cov)


    def simulate_data_vector(self, x):
        "Simulate a data vector by adding a realization of the covariance to the mean"
        #generate a vector of normal deviates
        r = jnp.random.randn(x.size)
        return x + jnp.dot(self.chol, r)


    def extract_theory_samples(self, block):
        "Extract relevant theory from block and get theory at data x values"
        theory_x = block[self.x_section, self.x_name]
        theory_y = block[self.y_section, self.y_name]
        return theory_x, theory_y


    @classmethod
    def build_module(cls):

        def setup(options):
            options = SectionOptions(options)
            likelihoodCalculator = cls(options)
            return likelihoodCalculator

        def execute(block, config):
            likelihoodCalculator = config
            likelihoodCalculator.do_likelihood(block)
            return 0

        def cleanup(config):
            likelihoodCalculator = config
            likelihoodCalculator.cleanup()

        return setup, execute, cleanup

    @classmethod
    def as_module(cls, name):
        setup, execute, cleanup = cls.build_module()
        return FunctionModule(name, setup, execute, cleanup)


# class SingleValueGaussianLikelihood(GaussianLikelihood):
#     """
#     A Gaussian likelihood whos input is a single calculated value
#     not a vector
#     """
#     name = MISSING
#     section = MISSING
#     like_name = MISSING
#     mean = None
#     sigma = None
#     def __init__(self, options):
#         self.options=options

#         #First try getting the value from the class itself
#         mean, sigma = self.build_data()

#         if options.has_value("mean"):
#             mean = options["mean"]
#         if options.has_value("sigma"):
#             sigma = options["sigma"]

#         if sigma is None or mean is None:
#             raise ValueError("Need to specify Gaussian mean/sigma for '{0}' \
#                 either in class definition, build_data method, or in the ini \
#                 file".format(self.like_name))
#         if options.has_value("like_name"):
#             self.like_name = options["like_name"]
#         print('Likelihood "{0}" will be Gaussian {1} +/- {2} '.format(self.like_name, mean, sigma))
#         self.data_y = np.array([mean])
#         self.cov = np.array([[sigma**2]])
#         self.inv_cov = np.array([[sigma**-2]])

#         include_norm = self.options.get_bool("include_norm", False)
#         if include_norm:
#             # We have no datablock here so we don't want to call any subclass method
#             self.log_det_constant = GaussianLikelihood.extract_covariance_log_determinant(self,None)
#             print("Including -0.5*|C| normalization in {} likelihood where log|C| = {}".format(self.like_name, self.log_det_constant))
#         else:
#             self.log_det_constant = 0.0

#         self.likelihood_only = options.get_bool('likelihood_only', False)

#         if not self.likelihood_only:
#             self.chol = sigma


#     def build_data(self):
#         """Sub-classes can over-ride this if they wish, to generate 
#         the data point in a more complex way"""
#         return self.mean, self.sigma

#     def build_covariance(self):
#         """This method is only defined here to satisfy the superclass requirements. 
#         There is no point over-riding it"""
#         raise RuntimeError("Internal cosmosis error in SingleValueGaussianLikelihood")

#     def extract_theory_points(self, block):
#         "Extract relevant theory from block and get theory at data x values"
#         return np.atleast_1d(block[self.section, self.name])


