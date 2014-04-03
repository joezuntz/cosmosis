from numpy import log, pi
import cosmosis

#Should load these from cosmosis
cosmo_section = "cosmological_parameters"
like_section = "likelihoods"

#Steigman 2008, equation 9
BBN_OMBH2_MEAN = 0.0218
BBN_OMBH2_SIGMA = 0.0011

def setup(options):
	mean = options.get_float(section, "mean", default=BBN_OMBH2_MEAN)
	sigma = options.get_float(section, "sigma", default=BBN_OMBH2_MEAN)
	norm = 0.5*log(2*pi*sigma**2)
	return (mean, sigma, norm)


def execute(block, config):
	# Configuration data, read from ini file above
	mean,sigma,norm = config

	# Get parameters from sampler
	h0 = block[cosmo, 'h0']
	omega_b = block[cosmo, 'omega_b']

	#compute the likelihood - just a simple Gaussian
	ombh2 = omega_b * h0**2
	like = -(ombh2-mean)**2/sigma**2/2.0 - norm
	block[like_section, 'BBN_LIKE'] = like

	#signal that everything went fine
	return 0

