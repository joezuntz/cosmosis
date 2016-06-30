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
            self.ndim = len(self.pipeline.varied_params)

            #Starting positions and values for the chain
            self.num_samples = 0
            self.prob0 = None
            self.blob0 = None

            if start_file:
                self.p0 = self.load_start(start_file)
                self.output.log_info("Loaded starting position from %s", start_file)
            elif random_start:
                self.p0 = [self.pipeline.randomized_start()
                           for i in xrange(self.nwalkers)]
            else:
                center_norm = self.pipeline.normalize_vector(self.pipeline.start_vector())
                sigma_norm=np.repeat(1e-3, center_norm.size)
                p0_norm = self.emcee.utils.sample_ball(center_norm, sigma_norm, size=self.nwalkers)
                self.p0 = [self.pipeline.denormalize_vector(p0_norm_i) for p0_norm_i in p0_norm]

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
