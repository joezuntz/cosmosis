import pydesglue
from numpy import log, pi

HST_H0_MEAN = 0.71
HST_H0_SIGMA = 0.02
HST_NORM = 0.5*log(2*pi*HST_H0_SIGMA**2)

def setup(handle):
	options = pydesglue.DesOptionPackage(handle)
	section = options._default_section
	mean = options.get_float(section, "mean", default=HST_H0_MEAN)
	sigma = options.get_float(section, "sigma", default=HST_H0_SIGMA)
	norm = 0.5*log(2*pi*sigma**2)
	return (mean,sigma,norm)

def execute(handle, config):
	mean,sigma,norm = config
	# Load the value of H0 from the package
	try:
		package = pydesglue.DesDataPackage.from_fits_handle(handle)
		cosmo = pydesglue.section_names.cosmological_parameters
		h0 = package.get_param(cosmo, 'H0')
	
		#compute the likelihood - just a simple Gaussian here
		like = -(h0-mean)**2/sigma**2/2.0 - norm
	
		section = pydesglue.section_names.likelihoods
		package.set_param(section, "HST_LIKE", like)
		package.write_to_fits_handle(handle)
	except KeyboardInterrupt:
		#If the user presses Ctrl-C we respect that
		raise KeyboardInterrupt
	except Exception as E:
		#But any other kind of error is a problem
		print "There was an error calculating the HST H0 likelihood: ", E
		return 1
	return 0
