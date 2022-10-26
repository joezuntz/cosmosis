import itertools
import numpy as np
from ...output.text_output import TextColumnOutput
from .. import ParallelSampler
from ...runtime import LikelihoodPipeline, ClassModule
from timeit import default_timer
def task(p, return_all=False):
    # print("Computing data vector for:", p)
    r = sampler.pipeline.run_results(p)
    block = r.block

    if block is None:
        return None

    data_vectors = []
    error_vectors = []

    if sampler.keys:
        # user has listed which keys they want
        for sec, key in sampler.keys:
            data_vectors.append(block[sec, key])
        if return_all:
            if sampler.error_keys:
                for sec, key in sampler.error_keys:
                    error_vectors.append(block[sec, key])
            else:
                for d in data_vectors:
                    error_vectors.append(np.ones_like(d))

    else:
        # use all the things found in the data_vector section
        for sec, key in block.keys(section="data_vector"):
            if not key.endwith("_theory"):
                continue
            data_vectors.append(r.block[sec, key])
            if return_all:
                covmat = block[sec, key[:-7] + "_covariance"]
                sigma = covmat.diagonal() ** 0.5
                error_vectors.append(sigma)

    #data_vectors = np.concatenate(data_vectors)

    if return_all:
        #error_vectors = np.concatenate(error_vectors)
        if len(error_vectors) != len(data_vectors):
            raise ValueError("Error and data vectors are different sizes")
        return r.like, data_vectors, error_vectors, r.block
    else:
        return r.like, data_vectors


def log_probability_function(u, tempering):
    try:
        p = sampler.pipeline.denormalize_vector_from_prior(u)
    except ValueError:
        return (-np.inf, (-np.inf, [np.nan for i in range(sampler.pipeline.number_extra)]))
    r = sampler.emu_pipeline.run_results(p)
    return tempering * r.post, (r.prior, r.extra)


class EmulatorModule(ClassModule):
    def __init__(self, options):
        pass

    def set_emulator_info(self, info):
        self.pipeline = info["pipeline"]
        self.fixed_inputs = info["fixed_inputs"]
        self.inputs = [(p.section, p.name) for p in self.pipeline.varied_params]
        self.outputs = info["outputs"]
        self.sizes = info["sizes"]
        self.emulator = None


    def set_emulator(self, emu):
        self.emulator = emu

    def execute(self, block):
        import torch
        p = np.array([block[sec, key] for (sec, key) in self.inputs])
        # These need to be denormalized as that's what the emulator is trained on
        p = np.array([self.pipeline.normalize_vector_to_prior(p)])
        # need to convert to a pytorch tensor
        p = torch.from_numpy(p.astype(np.float32))

        x = self.emulator.predict(p)[0]
        s = 0
        for (sec, key), sz in zip(self.outputs, self.sizes):
            block[sec, key] = x[s : s + sz]
            s += sz

        for (sec, key), val in self.fixed_inputs.items():
            block[sec, key] = val
        return 0


class EmugenSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("tempered_post", float), ("post", float)]

    def config(self):
        global sampler
        sampler = self

        self.converged = False
        keys = self.read_ini("keys", str)
        self.mode = self.read_ini_choices("mode", str, ["gp", "nn"])
        fixed_keys = self.read_ini("fixed_keys", str, "")
        error_keys = self.read_ini("error_keys", str, "")
        self.keys = [k.split(".") for k in keys.split()]
        self.fixed_keys = [k.split(".") for k in fixed_keys.split()]
        self.error_keys = [k.split(".") for k in error_keys.split()]
        self.max_iterations = self.read_ini("iterations", int, 4)
        self.initial_size = self.read_ini("initial_size", int, 9600)
        self.resample_size = self.read_ini("resample_size", int, 4800)
        self.chi2_cut_off = self.read_ini("chi2_cut_off", float)
        self.batch_size = self.read_ini("batch_size", int, 32)
        self.training_iterations = self.read_ini("training_iterations", int)
        self.emcee_walkers = self.read_ini("emcee_walkers", int)
        self.emcee_samples = self.read_ini("emcee_samples", int)
        self.emcee_burn = self.read_ini("emcee_burn", float, 0.3)
        self.emcee_thin = self.read_ini("emcee_thin", int, 1)
        self.last_emulated_module = self.read_ini("last_emulated_module", str)
        self.tempering = self.read_ini("tempering", float)
        self.seed = self.read_ini("seed", int, 0)
        self.ode = self.read_ini("seed", int, 0)
        if self.seed == 0:
            self.seed = None

        self.ndim = len(self.pipeline.varied_params)

        self.emu_pipeline = None
        self.iterations = 0

    def generate_initial_sample(self):
        import scipy.stats

        print("Generating initial sample")
        hypercube = scipy.stats.qmc.LatinHypercube(self.ndim, seed=self.seed)
        unit_sample = hypercube.random(n=self.initial_size)
        sample = np.array(
            [self.pipeline.denormalize_vector_from_prior(p) for p in unit_sample]
        )

        # Generate the likelihood and
        if self.pool:
            sample_results = self.pool.map(task, sample)
        else:
            sample_results = list(map(task, sample))

        # useful to save this with the emulator
        # Train our emulator

        sample_likes = np.array([s[0] for s in sample_results if s is not None])
        sample_data_vectors = np.array([np.concatenate(s[1]) for s in sample_results if s is not None])
        cut = -2 * sample_likes < self.chi2_cut_off

        sample_likes = sample_likes[cut]
        sample_data_vectors = sample_data_vectors[cut]
        sample = sample[cut]
        unit_sample = unit_sample[cut]

        n1 = len(sample_likes)
        print(f"{n1} initial samples had chi^2 < cut-off ({self.chi2_cut_off})")

        self.sample = sample
        self.sample_data_vectors = sample_data_vectors
        self.unit_sample = unit_sample



    def train_emulator(self):
        from .network3 import NNEmulator
        from .gp import GPEmulator

        n_samp, n_in = self.unit_sample.shape
        n_out = self.sample_data_vectors.shape[1]

        # Training emulator from {}
        print(f"Training emulator from {n_in} parameters -> {n_out} data vector points on {n_samp} points.")

        if self.mode == "nn":
            kwargs = {"n_epochs": self.training_iterations, "batch_size": self.batch_size}
            emu_class = NNEmulator
        elif self.mode == "gp":
            kwargs = {}
            emu_class = GPEmulator
        else:
            raise ValueError(f"Unknown training mode {self.mode} - should be gp or nn")

        emu = emu_class(
            n_in, n_out, self.fiducial_data_vector, self.fiducial_errors
        )

        emu.train(self.unit_sample, self.sample_data_vectors, **kwargs)

        self.emulator = emu
        self.emu_module.data.set_emulator(emu)


    def compute_fiducial(self):
        print("Computing fiducial data vector")
        p = self.pipeline.start_vector()
        p_unit = self.pipeline.normalize_vector(p)
        _, data_vectors, errors, block = task(p, return_all=True)
        for x, e in zip(data_vectors, errors):
            if len(x) != len(e):
                raise ValueError("Data vector and error vector different sizes")
        self.data_vector_sizes = [len(x) for x in data_vectors]
        self.fiducial_data_vector = np.concatenate(data_vectors)
        self.fiducial_errors = np.concatenate(errors)

        # find index of the emulation module
        module_names = [m.name for m in self.pipeline.modules]
        emu_index = module_names.index(self.last_emulated_module)
        emu_modules = self.pipeline.modules[emu_index + 1 :]

        fixed_inputs = {(sec,key): block[sec, key] for (sec, key) in self.fixed_keys}

        emu_module = EmulatorModule.as_module("emulator")
        emu_modules.insert(0, emu_module)
        self.emu_module = emu_module

        # Make a secondary pipeline object using the emulator
        print("Setting up emulated pipeline. This will print out the parameters again.")
        self.emu_pipeline = LikelihoodPipeline(
            self.pipeline.options, modules=emu_modules, values=self.pipeline.values_file
        )
        emu_module.data.set_emulator_info({
            "fixed_inputs": fixed_inputs,
            "pipeline": self.pipeline,
            "outputs": self.keys,
            "sizes": self.data_vector_sizes,
        })

    def generate_updated_sample(self):
        print(f"Selecting {self.resample_size} random samples from emcee chain to improve emulator")
        random_index = np.random.choice(np.arange(len(self.chain)), replace=False, size=self.resample_size)
        unit_sample = self.unit_chain[random_index]
        sample = self.chain[random_index]

        print(f"Running real pipeline on new sample")
        # Generate the likelihood and data vectors for the new sample
        if self.pool:
            sample_results = self.pool.map(task, sample)
        else:
            sample_results = list(map(task, sample))

        # append the results of the mcmc to the current sample
        sample_data_vectors = np.array([np.concatenate(s[1]) for s in sample_results if s is not None])
        self.sample = np.append(self.sample, sample, axis=0)
        self.sample_data_vectors = np.append(self.sample_data_vectors, sample_data_vectors, axis=0)
        self.unit_sample = np.append(self.unit_sample, unit_sample, axis=0)

    def get_emcee_start(self):
        # TODO: improve by removing low likelihood samples here
        # we want the last nwalker unique samples with likelihoods
        # that are within nsigma of the best
        return self.unit_sample[-self.emcee_walkers:]        

    def execute(self):
        import emcee
        if self.iterations == 0:
            self.compute_fiducial()
            self.generate_initial_sample()
        else:
            self.generate_updated_sample()

        print(f"Training emulator (iteration {self.iterations} /  {self.max_iterations})")

        self.train_emulator()


        if self.iterations < self.max_iterations - 1:
            print(f"Running emcee with tempering ({self.tempering}) - iteration {self.iterations}")
            tempering = self.tempering
        else:
            print(f"Running final emcee without tempering - iteration {self.iterations}")
            tempering = 1

        # run an mcmc using the current sample
        sampler = emcee.EnsembleSampler(
            self.emcee_walkers,
            self.ndim,
            log_probability_function,
            args=[tempering],
            pool=self.pool,
        )

        if self.emcee_burn < 1:
            burn = int(self.emcee_burn * self.emcee_samples)
        else:
            burn = int(self.emcee_burn)

        start_pos = self.get_emcee_start()

        sampler.run_mcmc(self.unit_sample[-self.emcee_walkers:], self.emcee_samples, progress=True)

        # the chain is in the unit cube
        self.unit_chain = sampler.get_chain(discard=burn, thin=self.emcee_thin, flat=True)
        logp = sampler.get_log_prob(discard=burn, thin=self.emcee_thin, flat=True)
        # derived parameters
        self.blobs = sampler.get_blobs(discard=burn, thin=self.emcee_thin, flat=True)

        self.chain = np.array(
            [self.pipeline.denormalize_vector_from_prior(p) for p in self.unit_chain]
        )


        # We discard the previous chain contents
        self.output.reset_to_chain_start()

        # and then output the latest version of the chain
        for params, tempered_post, extra in zip(self.chain, logp, self.blobs):
            prior, extra = extra
            post = tempered_post / tempering
            self.output.parameters(params, extra, prior, tempered_post, post)



        # Iterate more!
        self.iterations += 1

    def is_converged(self):
        return self.iterations >= self.max_iterations
