from builtins import zip
from builtins import range
from builtins import str
from .. import ParallelSampler
import numpy as np


def log_probability_function(p):
    return emcee_pipeline.posterior(p)


class EmceeSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("post", float)]

    def config(self):
        global emcee_pipeline
        emcee_pipeline = self.pipeline

        if self.is_master():
            import emcee
            self.emcee = emcee

            # Parameters of the emcee sampler
            self.nwalkers = self.read_ini("walkers", int, 2)
            self.samples = self.read_ini("samples", int, 1000)
            self.nsteps = self.read_ini("nsteps", int, 100)

            assert self.nsteps>0, "You specified nsteps<=0 in the ini file - please set a positive integer"
            assert self.samples>0, "You specified samples<=0 in the ini file - please set a positive integer"

            random_start = self.read_ini("random_start", bool, False)
            start_file = self.read_ini("start_points", str, "")
            covmat_file = self.read_ini("covmat", str, "")
            self.ndim = len(self.pipeline.varied_params)

            #Starting positions and values for the chain
            self.num_samples = 0
            self.prob0 = None
            self.blob0 = None

            if start_file:
                self.p0 = self.load_start(start_file)
                self.output.log_info("Loaded starting position from %s", start_file)
            elif self.distribution_hints.has_cov():
                center = self.start_estimate()
                cov = self.distribution_hints.get_cov()
                self.p0 = self.emcee.utils.sample_ellipsoid(center, cov, size=self.nwalkers)
                self.output.log_info("Generating starting positions from covmat from earlier in pipeline")
            elif covmat_file:
                center = self.start_estimate()
                cov = self.load_covmat(covmat_file)
                self.output.log_info("Generating starting position from covmat in  %s", covmat_file)
                iterations_limit = 100000
                n=0
                p0 = []
                for i in range(iterations_limit):
                    p = self.emcee.utils.sample_ellipsoid(center, cov)[0]
                    if np.isfinite(self.pipeline.prior(p)):
                        p0.append(p)
                    if len(p0)==self.nwalkers:
                        break
                else:
                    raise ValueError("The covmat you used could not generate points inside the prior")
                self.p0 = np.array(p0)
            elif random_start:
                self.p0 = [self.pipeline.randomized_start()
                           for i in range(self.nwalkers)]
                self.output.log_info("Generating random starting positions from within prior")
            else:
                center_norm = self.pipeline.normalize_vector(self.start_estimate())
                sigma_norm=np.repeat(1e-3, center_norm.size)
                p0_norm = self.emcee.utils.sample_ball(center_norm, sigma_norm, size=self.nwalkers)
                p0_norm[p0_norm<=0] = 0.001
                p0_norm[p0_norm>=1] = 0.999
                self.p0 = [self.pipeline.denormalize_vector(p0_norm_i) for p0_norm_i in p0_norm]
                self.output.log_info("Generating starting positions in small ball around starting point")

            #Finally we can create the sampler
            self.ensemble = self.emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                                       log_probability_function,
                                                       pool=self.pool)

    def load_start(self, filename):
        #Load the data and cut to the bits we need.
        #This means you can either just use a test file with
        #starting points, or an emcee output file.
        data = np.genfromtxt(filename, invalid_raise=False)[-self.nwalkers:, :self.ndim]
        if data.shape != (self.nwalkers, self.ndim):
            raise RuntimeError("There are not enough lines or columns "
                               "in the starting point file %s" % filename)
        return list(data)


    def load_covmat(self, covmat_file):
        covmat = np.loadtxt(covmat_file)

        if covmat.ndim == 0:
            covmat = covmat.reshape((1, 1))
        elif covmat.ndim == 1:
            covmat = np.diag(covmat ** 2)

        nparams = len(self.pipeline.varied_params)
        if covmat.shape != (nparams, nparams):
            raise ValueError("The covariance matrix was shape (%d x %d), "
                    "but there are %d varied parameters." %
                    (covmat.shape[0], covmat.shape[1], nparams))
        return covmat


    def output_samples(self, pos, prob, extra_info):
        for p,l,e in zip(pos,prob,extra_info):
            self.output.parameters(p, e, l)

    def execute(self):
        #Run the emcee sampler.
        outputs = []
        for (pos, prob, rstate, extra_info) in self.ensemble.sample(
                self.p0, lnprob0=self.prob0, blobs0=self.blob0,
                iterations=self.nsteps, storechain=False):
            outputs.append((pos.copy(), prob.copy(), extra_info[:]))
    
        for (pos, prob, extra_info) in outputs:
            self.output_samples(pos, prob, extra_info)

        #Set the starting positions for the next chunk of samples
        #to the last ones for this chunk
        self.p0 = pos
        self.prob0 = prob
        self.blob0 = extra_info
        self.num_samples += self.nsteps
        self.output.log_info("Done %d iterations", self.num_samples)
        self.output.final("mean_acceptance_fraction", self.ensemble.acceptance_fraction.mean())

    def is_converged(self):
        return self.num_samples >= self.samples
