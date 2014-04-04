from numpy import log, pi

cosmo = cosmosis_py.names.cosmological_parameters_section
likes = cosmosis_py.names.likelihoods_section

HST_H0_MEAN = 0.71
HST_H0_SIGMA = 0.02

def setup(options):
	mean = options.get_float(section, "mean", default=HST_H0_MEAN)
	sigma = options.get_float(section, "sigma", default=HST_H0_SIGMA)
	norm = 0.5*log(2*pi*sigma**2)
	return (mean, sigma, norm)


def execute(block, config):
	# Configuration data, read from ini file above
	mean,sigma,norm = config

	# Get parameters from sampler
	h0 = block[cosmo, 'h0']

	#compute the likelihood - just a simple Gaussian
	like = -(h0-mean)**2/sigma**2/2.0 - norm
	block[likes, 'HST_LIKE'] = like

	#signal that everything went fine
	return 0
