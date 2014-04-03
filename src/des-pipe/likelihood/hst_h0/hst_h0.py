from numpy import log, pi

cosmo_section = "cosmological_parameters"
like_section = "likelihoods"

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
	block[like_section, 'HST_LIKE'] = like

	#signal that everything went fine
	return 0
