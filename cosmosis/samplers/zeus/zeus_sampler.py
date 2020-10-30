from builtins import zip
from builtins import range
from builtins import str
from .. import ParallelSampler, sample_ellipsoid, sample_ball
import numpy as np
import sys


def log_probability_function(p):
    r = zeus_pipeline.run_results(p)
    return r.post, (r.prior, r.extra)


class ZeusSampler(ParallelSampler):
    parallel_output = False
    supports_resume = True
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global zeus_pipeline
        zeus_pipeline = self.pipeline

        if self.is_master():
            # we still import emcee, to use some of its
            # tools
            import emcee
            import zeus
            self.emcee = emcee
            self.zeus = zeus

            if not hasattr(zeus, "EnsembleSampler"):
                raise ImportError("There are two python packages called Zeus, and you"
                    " have the wrong one.  Uninstall zeus and install zeus-mcmc.")


            # Parameters of the zeus sampler
            self.nwalkers = self.read_ini("walkers", int)
            self.samples = self.read_ini("samples", int)
            self.nsteps = self.read_ini("nsteps", int, 50)

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
                self.p0 = sample_ellipsoid(center, cov, size=self.nwalkers)
                self.output.log_info("Generating starting positions from covmat from earlier in pipeline")
            elif covmat_file:
                center = self.start_estimate()
                cov = self.load_covmat(covmat_file)
                self.output.log_info("Generating starting position from covmat in  %s", covmat_file)
                iterations_limit = 100000
                n=0
                p0 = []
                for i in range(iterations_limit):
                    p = sample_ellipsoid(center, cov)[0]
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
                p0_norm = sample_ball(center_norm, sigma_norm, size=self.nwalkers)
                p0_norm[p0_norm<=0] = 0.001
                p0_norm[p0_norm>=1] = 0.999
                self.p0 = [self.pipeline.denormalize_vector(p0_norm_i) for p0_norm_i in p0_norm]
                self.output.log_info("Generating starting positions in small ball around starting point")

            #Finally we can create the sampler
            self.sampler = self.zeus.EnsembleSampler(self.nwalkers, self.ndim,
                                                       log_probability_function,
                                                       pool=self.pool)

    def resume(self):
        if self.output.resumed:
            data = np.genfromtxt(self.output._filename, invalid_raise=False)[:, :self.ndim]
            num_samples = len(data) // self.nwalkers
            self.p0 = data[-self.nwalkers:]
            self.num_samples += num_samples
            if self.num_samples >= self.samples:
                print("You told me to resume the chain - it has already completed (with {} samples), so sampling will end.".format(len(data)))
                print("Increase the 'samples' parameter to keep going.")
            else:
                print("Continuing zeus from existing chain - have {} samples already".format(len(data)))

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


    def execute(self):
        #Run the emcee sampler.
        if self.num_samples == 0:
            print("Begun sampling")
            start = 0
        else:
            start = self.sampler.chain.shape[0]
        self.sampler.run(self.p0, self.nsteps)
        end = self.sampler.chain.shape[0]

        post = self.sampler.get_log_prob()
        chain = self.sampler.get_chain()
        blobs = self.sampler.get_blobs()

            # zeus does not support derived params.  make them nan
        for i in range(start, end):
            for j in range(self.nwalkers):
                prior, extra = blobs[i, j]
                self.output.parameters(chain[i, j], extra, prior, post[i, j])

        #Set the starting positions for the next chunk of samples
        #to the last ones for this chunk
        self.p0 = self.sampler.get_last_sample
        self.num_samples += self.nsteps
        taus = self.zeus.AutoCorrTime(chain)
        print("\nHave {} samples from zeus. Current auto-correlation estimates are:".format(
            self.num_samples*self.nwalkers))
        for par, tau in zip(self.pipeline.varied_params, taus):
            print("   {}:  {:.2f}".format(par, tau))
        sys.stdout.flush()

    def is_converged(self):
        return self.num_samples >= self.samples

