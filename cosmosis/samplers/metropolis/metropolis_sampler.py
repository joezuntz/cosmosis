from .. import ParallelSampler
import numpy as np
from . import metropolis
import os

#We need a global pipeline
#object for MPI to work properly
pipeline=None


METROPOLIS_INI_SECTION = "metropolis"

def posterior(p):
    return pipeline.posterior(p)


class MetropolisSampler(ParallelSampler):
    parallel_output = True
    sampler_outputs = [("like", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline
        self.samples = self.read_ini("samples", int, default=20000)
	self.interrupted = False
	self.num_samples = 0
        #Any other options go here

	#start values from prior
        start = self.define_parameters()
	self.n = len(start)

	try:
                start = self.pipeline.denormalize_vector(start)
	except ValueError:
		return -np.inf

	try:
		covmat = self.load_covariance_matrix()
	except IOError:
		covmat = None
        self.sampler = metropolis.MCMC(start, posterior, covmat, self.pool)

    def execute(self):
        #Run the MCMC  sampler.
        samples = self.sampler.sample(self.n)
	self.num_samples += 1
        for vector, like in samples:
        	self.output.parameters(vector, like)
        self.sampler.tune()

    def is_converged(self):
	 # user has pressed Ctrl-C
        if self.interrupted:
            return True
        if self.num_samples >= self.samples:
            return True
        elif self.num_samples > 0 and self.pool is not None and \
             self.Rconverge is not None:
            return np.all(self.analytics.gelman_rubin() <= self.Rconverge)
        else:
            return False



    def load_covariance_matrix(self):
	covmat_filename = self.ini.get(METROPOLIS_INI_SECTION, "covmat", "").strip()
	if covmat_filename == "":
		return None
	if not os.path.exists(covmat_filename):
		raise ValueError(
		"Covariance matrix %s not found" % covmat_filename)
	covmat = np.loadtxt(covmat_filename)

	if covmat.ndim == 0:
		covmat = covmat.reshape((1, 1))
	elif covmat.ndim == 1:
		covmat = np.diag(covmat ** 2)

	nparams = len(self.pipeline.varied_params)
	if covmat.shape != (nparams, nparams):
		raise ValueError("The covariance matrix was shape (%d x %d), "
				"but there are %d varied parameters." %
				(covmat.shape[0], covmat.shape[1], nparams))

	# normalize covariance matrix	
	#r = np.array([param.width() for param
	#	in self.pipeline.varied_params])
	#for i in xrange(covmat.shape[0]):
	#	covmat[i, :] /= r
	#	covmat[:, i] /= r

	return covmat



    def define_parameters(self):
	priors = []
	for param in self.pipeline.varied_params:
	    prior = param.prior
	    start_value = param.normalize(param.random_point())

	    if prior is None or isinstance(prior, UniformPrior):
	    	# uniform prior
		priors.append(np.random.uniform())
	    elif isinstance(prior, GaussianPrior):
		sd = (prior.sigma2)**0.5
		priors.append(np.random.normal(loc=start_value,scale =sd))
	    elif isinstance(prior, ExponentialPrior):
		priors.append(np.random.exponential(scale=1.0))
	    else:
	    	raise RuntimeError("Unknown prior type in MCMC sampler")
	return priors
