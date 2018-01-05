from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import str
from .. import ParallelSampler
import numpy as np

pipeline = None
def log_probability_function(p):
    global pipeline
    return pipeline.posterior(p)



def sample_ball(p0, std, size=1):
    """
    Produce a ball of walkers around an initial parameter value.

    :param p0: The initial parameter value.
    :param std: The axis-aligned standard deviation.
    :param size: The number of samples to produce.

    Function liberated from emcee utils module.
    http://dan.iel.fm/emcee/current/

    """
    assert(len(p0) == len(std))
    return np.vstack([p0 + std * np.random.normal(size=len(p0))
                      for i in range(size)])


class KombineSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            try:
                import kombine
            except ImportError:
                raise ImportError("The kombine package must be installed to use the kombine sampler\n" +
                            "You can use pip from inside the CosmoSIS environment:\n" +
                            "pip install --user git+git://github.com/bfarr/kombine")
            self.kombine = kombine

            # Parameters of the kombine sampler
            self.nwalkers = self.read_ini("walkers", int, 2)
            self.samples = self.read_ini("samples", int, 1000)
            self.nsteps = self.read_ini("nsteps", int, 100)
            self.update_interval = self.read_ini("update_interval", int, 100)
            start_file = self.read_ini("start_points", str, "")
            random_start = self.read_ini("random_start", bool, False)

            #Starting positions and values for the chain
            self.ndim = len(self.pipeline.varied_params)
            self.num_samples = 0
            self.lnpost0 = None
            self.lnprop0 = None
            self.blob0 = None

            if start_file:
                self.p0 = self.load_start(start_file)
                self.output.log_info("Loaded starting position from %s", start_file)
            elif random_start:
                self.p0 = [self.pipeline.randomized_start()
                           for i in range(self.nwalkers)]
            else:
                center_norm = self.pipeline.normalize_vector(self.pipeline.start_vector())
                sigma_norm=np.repeat(1e-3, center_norm.size)
                p0_norm = sample_ball(center_norm, sigma_norm, size=self.nwalkers)
                self.p0 = [self.pipeline.denormalize_vector(p0_norm_i) for p0_norm_i in p0_norm]

            #Finally we can create the sampler
            self.ensemble = self.kombine.Sampler(self.nwalkers, self.ndim,
                                                 log_probability_function,
                                                 processes=1,
                                                 pool=self.pool)
            self.burned_in = False

    def load_start(self, filename):
        #Load the data and cut to the bits we need.
        #This means you can either just use a test file with
        #starting points, or an output file.
        data = np.genfromtxt(filename, invalid_raise=False)[-self.nwalkers:, :self.ndim]
        if data.shape != (self.nwalkers, self.ndim):
            raise RuntimeError("There are not enough lines or columns "
                               "in the starting point file %s" % filename)
        return list(data)

    def output_samples(self, pos, prob, extra_info):
        for p,l,e in zip(pos,prob,extra_info):
            self.output.parameters(p, e, l)

    def execute(self):
        if not self.burned_in:
            results = self.ensemble.burnin(self.p0, update_interval=self.update_interval)
            try:
                pos, post, prop, extra_info = results
            except ValueError:
                pos, post, prop = results
                extra_info = None
            if self.is_master():
                print("Burn-in phase complete.")
            self.p0 = pos
            self.lnpost0 = post
            self.lnprop0 = prop
            self.blob0 = extra_info
            self.burned_in = True
            return

        outputs = []
        # unlike emcee, kombine only returns extra_info if posterior function provides it
        for results in self.ensemble.sample(
                self.p0, lnpost0=self.lnpost0, lnprop0=self.lnprop0, blob0=self.blob0,
                storechain=False, iterations=self.nsteps, update_interval=self.update_interval):
            try:
                pos, post, prop, extra_info = results
            except ValueError:
                pos, post, prop = results
                extra_info = []
            outputs.append((pos.copy(), post.copy(), prop.copy(), extra_info[:]))

        for (pos, post, prop, extra_info) in outputs:
            self.output_samples(pos, post, extra_info)

        # Set the starting positions for the next chunk of samples
        # to the last ones for this chunk
        self.p0 = pos
        self.lnpost0 = post
        self.prob0 = prop
        self.blob0 = extra_info
        self.num_samples += self.nsteps
        self.output.log_info("Done %d iterations", self.num_samples)

    def is_converged(self):
        return self.num_samples >= self.samples
