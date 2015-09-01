import abc
import scipy.interpolate
import scipy.integrate
import scipy.linalg
import numpy as np
from cosmosis.datablock import names, SectionOptions
import traceback 

class GaussianLikelihood(object):
	__metaclass__=abc.ABCMeta
	"""
	Gaussian likelihood with a fixed covariance.  
	
	Subclasses must override build_data and build_covariance,
	e.g. to load from file.
	"""
	x_section = "missing"
	x_name    = "missing"
	y_section = "missing"
	y_name    = "missing"
	like_name = "missing"

	def __init__(self, options):
		self.options=options
		self.data_x, self.data_y = self.build_data()
		self.cov = self.build_covariance()
		self.inv_cov = np.linalg.inv(self.cov)
		self.kind = self.kwargs.get("kind", "cubic")

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



	@abc.abstractmethod
	def build_data(self):
		"""
		Override the build_data method to read or generate 
		the observed data vector to be used in the likelihood
		"""
		#using info in self.options,
		#like filenames etc,
		#build x to which we must interpolate
		#return x, y
		pass
	
	@abc.abstractmethod
	def build_covariance(self):
		"""
		Override the build_data method to read or generate 
		the observed covariance
		"""
		#using info in self.options,
		#like filenames etc,
		#build covariance
		pass

	def cleanup(self):
		"""
		You can override the cleanup method if you do something 
		unusual to get your data, like open a database or something.
		It is run just once, at the end of the pipeline.
		"""
		pass
	

	def do_likelihood(self, block):
		#get data x by interpolation
		x = self.block_theory_points(block)
		mu = self.data_y

		print self.inv_cov
		#gaussian likelihood
		d = x-mu
		like = -0.5*np.einsum('i,ij,j', d, self.inv_cov, d)

		#Now save the resulting likelihood
		block[names.likelihoods, self.like_name+"_LIKE"] = like

		#And also the predicted data points, which we refer to
		#here as the "fisher vector" - the vector of observables we want the derivatives
		#of later, and inverse cov mat which also goes into the fisher matrix.
		#If there is an existing Fisher vector, append to it.
		if block.has_value(names.fisher, 'vector'):
			v = block[names.fisher, 'vector']
			v = np.concatenate((v, x))

			#and the same for the inverse covmat
			M = block[names.fisher, 'inv_cov']
			M = scipy.linalg.block_diag(M, np.atleast_2d(self.inv_cov))
		else:
			#otherwise just use an empty existing vector
			v = x
			M = np.atleast_2d(self.inv_cov)

		#apend the new theory points and save the result
		
		block[names.fisher, 'vector'] = v
		block[names.fisher, 'inv_cov'] = M




	def block_theory_points(self, block):
		"Extract relevant theory from block and get theory at data x values"
		theory_x = block[self.x_section, self.x_name]
		theory_y = block[self.y_section, self.y_name]
		return self.generate_theory_points(theory_x, theory_y)

	def generate_theory_points(self, theory_x, theory_y):
		"Generate theory predicted data points by interpolation into the theory"
		f = scipy.interpolate.interp(theory_x, theory_y, kind=self.kind)
		return np.atleast_1d(f(self.data_x))

	@classmethod
	def build_module(cls):

		def setup(options):
			options = SectionOptions(options)
			likelihoodCalculator = cls(options)
			return likelihoodCalculator

		def execute(block, config):
			likelihoodCalculator = config
			try:
				likelihoodCalculator.do_likelihood(block)
				return 0
			except Exception as error:
				print "Error getting likelihood:"
				print traceback.format_exc()
				return 1

		def cleanup(config):
			likelihoodCalculator = config
			likelihoodCalculator.cleanup()


		return setup, execute, cleanup


class SingleValueGaussianLikelihood(GaussianLikelihood):
	"""
	A Gaussian likelihood whos input is a single calculated value
	not a vector
	"""
	name = "missing"
	section = "missing"
	like_name = "missing"	
	mean = None
	sigma = None
	def __init__(self, options):
		self.options=options

		#First try getting the value from the class itself
		mean, sigma = self.build_data(options)

		if options.has_value("mean"):
			mean = options["mean"]
		if options.has_value("sigma"):
			sigma = options["sigma"]
		print mean, sigma
		if sigma is None or mean is None:
			raise ValueError("Need to specify Gaussian mean/sigma for '{0}' \
				either in class definition, build_data method, or in the ini \
				file".format(self.name))
		if options.has_value("like_name"):
			self.like_name = options["like_name"]
		self.data_y = np.array([mean])
		self.cov = np.array([[sigma**2]])
		self.inv_cov = np.array([[sigma**-2]])

	def build_data(self, options):
		"""Sub-classes can over-ride this if they wish, to generate 
		the data point in a more complex way"""
		return self.mean, self.sigma

	def build_covariance(self, options):
		"""This method is only defined here to satisfy the superclass requirements. 
		There is no point over-riding it"""
		raise RuntimeError("Internal cosmosis error in Gaussian Likelihood")

	def block_theory_points(self, block):
		"Extract relevant theory from block and get theory at data x values"
		return np.atleast_1d(block[self.section, self.name])



class WindowedGaussianLikelihood(GaussianLikelihood):
	def __init__(self, options):
		super(WindowedGaussianLikelihood, self).__init__(options)
		self.windows = self.build_windows()

	@abc.abstractmethod
	def build_windows(self):
		pass

	def generate_theory_points(self, theory_x, theory_y):
		"Generate theory predicted data points using window function"
		f = scipy.interpolate.interp(theory_x, theory_y, kind=self.kind)
		values = []
		for window_x, window_y in self.windows:
			xmin = max(window_x.min(), theory_x.min())
			xmax = min(window_x.max(), theory_x.max())
			w = scipy.interpolate.interp(window_x, window_y, kind=self.kind)
			g = lambda x: w(x)*f(x)
			v = scipy.integrate.romberg(g, xmin, xmax)
			values.append(v)
		return np.atleast_1d(values)
		

