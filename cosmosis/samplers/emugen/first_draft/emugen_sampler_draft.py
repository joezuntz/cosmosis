import itertools
import numpy as np
from ...output.text_output import TextColumnOutput
from .. import ParallelSampler
from cosmosis.gaussian_likelihood import GaussianLikelihood
from ...runtime import LikelihoodPipeline, ClassModule
from timeit import default_timer
import h5py as h5
import os
import errno
import matplotlib.pyplot as plt
import types
import copy 
def mkdir(path):
    u"""Ensure that all the components in the `path` exist in the file system.
    """
    # Avoid trying to make an empty path
    if not path.strip():
        return
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno == errno.EEXIST:
            if os.path.isdir(path):
                #error is that dir already exists; fine - no error
                pass
            else:
                #error is that file with name of dir exists already
                raise ValueError("Tried to create dir %s but file with name exists already"%path)
        elif error.errno == errno.ENOTDIR:
            #some part of the path (not the end) already exists as a file 
            raise ValueError("Tried to create dir %s but some part of the path already exists as a file"%path)
        else:
            #Some other kind of error making directory
            raise

SAVE_NONE = 0
SAVE_MODEL = 1
SAVE_ALL = 2

def task(p, return_all=False):
    r = sampler.pipeline.run_results(p)
    block = r.block
    if block is None:
        return None
    data_vectors_theory = []
    data_vectors = []
    error_vectors = []
    data_inv_covariance = []
    if sampler.keys:
        # user has listed which keys they want
        for sec, key in sampler.keys:
            if type(block[sec, key])==float:
                data_vectors_theory.append(np.array([block[sec, key]]))
            else:
                data_vectors_theory.append(block[sec, key])
        if return_all:
            if sampler.error_keys:
                for sec, key in sampler.error_keys:
                    error_vectors.append(block[sec, key])
            else:
                for d in data_vectors_theory:
                    error_vectors.append(np.ones_like(d))

    else:
        # use all the things found in the data_vector section
        for sec, key in block.keys(section="data_vector"):
            if not key.endswith("_theory"):
                continue
            data_vectors_theory.append(r.block[sec, key])
            if return_all:
                covmat = block[sec, key[:-7] + "_covariance"]
                sigma = covmat.diagonal() ** 0.5
                error_vectors.append(sigma)
                data_inv_covariance.append(block[sec, key[:-7] + "_inverse_covariance"])
                data_vectors.append(block[sec, key[:-7] + "_data"])
       
    if return_all:
        if len(error_vectors) != len(data_vectors_theory):
            raise ValueError("Error and data vectors are different sizes")
        return r.like, data_vectors_theory, data_vectors, data_inv_covariance, error_vectors, r.block
    else:
        return r.like, data_vectors_theory, r.prior, r.post


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
        self.nn_model = info["nn_model"]
        self.emulator = None


    def set_emulator(self, emu):
        # specify network
        self.emulator = emu

    def execute(self, block):
        p_dict = {**{str(sec)+'--'+str(key): block[sec, key]  for (sec, key) in self.inputs}}
        x = self.emulator.predict(p_dict)[0]
        if self.outputs==[]:
            block["data_vector", "theory_emulated"] = x 
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
        # properties to emulate or empty for the final datavector
        keys = self.read_ini("keys", str)
        fixed_keys = self.read_ini("fixed_keys", str, "")
        error_keys = self.read_ini("error_keys", str, "")
        self.keys = [k.split(".") for k in keys.split()]
        self.fixed_keys = [k.split(".") for k in fixed_keys.split()]
        self.error_keys = [k.split(".") for k in error_keys.split()]
        save_outputs = self.read_ini("save_outputs", str, "")  
        if save_outputs:
            self.save_outputs_dir = self.read_ini("save_outputs_dir", str, "")
            mkdir(self.save_outputs_dir)
            if save_outputs=="model":
                # only save the final trained model
                self.save_outputs = SAVE_MODEL
            elif save_outputs=="all":  
                # save the following information:
                # * trained model after each itteration
                # * training set and testing set
                # * diagnostic: loss-function vs epoch (within training and on the validation set), best-fit model in each itteration vs data
                self.save_outputs = SAVE_ALL
            else:
                raise ValueError(f"Unknown save_outputs option {save_outputs} - should be model or all or empty")   
        else: 
            self.save_outputs = SAVE_NONE   
            print('WARNING: Nothing is being saved in this run!')      
   
        self.load_emu_filename = self.read_ini("load_emu_filename", str, "")
        self.trained_before = self.read_ini("trained_before", bool, False)


        # number of itterations/cycles of training
        self.max_iterations = self.read_ini("iterations", int, 4)
        # initial size of the training set computed (slow and exact)
        self.initial_size = self.read_ini("initial_size", int, 9600)
        # additional set of computed properties in ech cycle (slow and exact)
        self.resample_size = self.read_ini("resample_size", int, 4800)
        # cutting off data point in the initial training set for chi2 larger than the cut-off
        self.chi2_cut_off = self.read_ini("chi2_cut_off", float)
        # minimal batch sizes in the first itteration, increases like batch_size*(1+itteration)
        # fixed for now within an itteration for all number of training cycles
        self.batch_size = self.read_ini("batch_size", int, 32)
        # number of cycle per itteration 
        self.training_iterations = self.read_ini("training_iterations", int, 5)

        # emcee parameters for sampling within each cycle        
        self.emcee_walkers = self.read_ini("emcee_walkers", int)
        self.emcee_samples = self.read_ini("emcee_samples", int)
        self.emcee_burn = self.read_ini("emcee_burn", float, 0.3)
        self.emcee_thin = self.read_ini("emcee_thin", int, 1)
        if self.emcee_walkers>=self.initial_size:
            raise ValueError('Number of walkers must be smaller than the initial training set!')
        # last emulated module: can be any module before the likelihood 
        # but must agree with the choices of keys!
        self.last_emulated_module = self.read_ini("last_emulated_module", str) 
        tempering = self.read_ini("tempering", float, 0.05)
        self.tempering = np.full(self.max_iterations, tempering)
        self.seed = self.read_ini("seed", int, 0)
        self.ode = self.read_ini("seed", int, 0)
        if self.seed == 0:
            self.seed = None
        self.ndim = len(self.pipeline.varied_params)
        self.emu_pipeline = None
        self.iterations = 0

        tempering_file = self.read_ini("tempering_file", str, "") 
        if tempering_file:
            self.tempering = np.genfromtxt(tempering_file)[:self.max_iterations]
        print('TEMPERING (re-scalling of the likelihood in each itteration): ', self.tempering)    
        # additional options (fixed by now)
        # transformation of the data vector
        self.data_trafo = self.read_ini("data_trafo", str, "log_norm")
        self.n_pca = self.read_ini("n_pca", int, 32)
        # loss function
        self.loss_function = self.read_ini("loss_function", str, "default")
        if self.loss_function.startswith("weighted"):
            assert keys==[], "Weighted loss-function can be applied only to the full data vector"   
        # nn architecture
        self.nn_model = 'MLP' #'ResMLP
        if self.nn_model not in ['MLP', 'ResMLP']:
            raise ValueError(f"Unknown training mode {self.nn_model} - should be MLP or ResMPL")
        

    def generate_initial_sample(self):
        import scipy.stats
        print("Generating initial sample")
        #TODO for Gaussian priors take 3-sigmas or 5-sigmas
        hypercube = scipy.stats.qmc.LatinHypercube(self.ndim, seed=self.seed)
        unit_sample = hypercube.random(n=self.initial_size)
        sample = np.array(
            [self.pipeline.denormalize_vector_from_prior(p) for p in unit_sample]
        )
        print('Parameters de-normalised with their priors')
        print('sample = parameters: ', sample)
        print('Is parallelisation on? ', self.pool)
        # Generate the likelihood and data-vectors
        if self.pool:
            sample_results = self.pool.map(task, sample)
        else:
            sample_results = list(map(task, sample))
        # useful to save this with the emulator

        # Prepare training set for the emulator
        # Maybe don't cutoff in chi2 for the initial training? Especially if nested sampling is used?
        sample_likes = np.array([s[0] for s in sample_results if s is not None])
        sample_data_vectors = np.array([np.concatenate(s[1]) for s in sample_results if s is not None])
        sample_priors = np.array([s[2] for s in sample_results if s is not None])
        sample_posts = np.array([s[3] for s in sample_results if s is not None])
        cut = -2 * sample_likes < self.chi2_cut_off
        print('sample_data_vectors.shape = ', sample_data_vectors.shape)
        sample_likes = sample_likes[cut]
        sample_priors = sample_priors[cut]
        sample_posts = sample_posts[cut]
        sample_data_vectors = sample_data_vectors[cut]
        sample = sample[cut]
        unit_sample = unit_sample[cut]

        n1 = len(sample_likes)
        print('Without tempering:')
        print(f"{n1} initial samples had chi^2 < cut-off ({self.chi2_cut_off})")
        self.initial_size_cut = n1
        self.sample = sample
        self.sample_data_vectors = sample_data_vectors
        self.unit_sample = unit_sample
        self.sample_likes = sample_likes
        self.sample_priors = sample_priors
        self.sample_posts = sample_posts

        self.sample_test = []
        self.sample_data_vectors_test = []
        self.unit_sample_test = []
        self.sample_likes_test = []
        self.sample_priors_test = []
        self.sample_posts_test = []



    def train_emulator(self):
        from .cosmopower import CPEmulator
        n_samp, n_in = self.unit_sample.shape
        n_out = self.sample_data_vectors.shape[1]
        print(f"Training emulator from {n_in} parameters -> {n_out} data vector points on {n_samp} points.")
        kwargs = {"model_filename": f'{self.save_outputs_dir}/emumodel_{self.iterations+1}', 
                  "n_cycles": self.training_iterations, "batch_size": self.batch_size*(self.iterations+1)}
        emu_class = CPEmulator
        model_parameters = [str(param) for param in self.pipeline.varied_params]
        print('MODEL_PARAMETERS: ', model_parameters)
        emu = emu_class(
            model_parameters, np.arange(n_out), self.nn_model, self.iterations+1, self.data_trafo, self.n_pca, self.data, self.inv_cov
        )
        X = {str(param): self.sample[:, i] for i, param in enumerate(self.pipeline.varied_params)}
        #print('emu-train dictionary: ', X)
        self.sample_data_vectors = self.sample_data_vectors
        # train emulator
        emu.train(X, self.sample_data_vectors, **kwargs)
        self.emulator = emu
        # set this emulator as the modelling module in the emulated pipeline
        self.emu_module.data.set_emulator(emu)

    def load_emulator(self):
        from .cosmopower import CPEmulator
        filename = self.load_emu_filename
        emu_class = CPEmulator
        model_parameters = [str(param) for param in self.pipeline.varied_params]
        print('MODEL_PARAMETERS: ', model_parameters)
        emu = emu_class(
            model_parameters, np.ones(10)
        )
        # load model 
        emu.load(filename)
        self.emulator = emu
        # set this emulator as the moduling block in the emulated pipeline
        self.emu_module.data.set_emulator(emu)

    def inject_emulator_into_likemodule(self, module):
        original_setup = module.setup_function

        def setup_wrapper(config):
            # Step 1: Call the original setup to get the instance
            instance = original_setup(config)
            # Step 2: (Monkey) Patch the method
            def emulated_extract_theory_points(self, block):
                data_vector = block["data_vector", "theory_emulated"]
                return data_vector
            instance.extract_theory_points = types.MethodType(emulated_extract_theory_points, instance)
            # Step 3: Return the patched instance
            return instance
        # Replace the setup_function with our wrapper
        module.setup_function = setup_wrapper


    def compute_fiducial_setup_emu_pipeline(self):
        print("Computing fiducial data vector")
        p = self.pipeline.start_vector()
        p_unit = self.pipeline.normalize_vector(p)
        _, data_vectors, self.data, self.inv_cov, errors, block = task(p, return_all=True)
        # sometimes data_vector can include a float, e.g. distances.rs_zdrag
        self.data_vector_sizes = [len(x) for x in data_vectors]
        self.fiducial_data_vector = np.concatenate(data_vectors)
        self.fiducial_errors = np.concatenate(errors)
        print('self.fiducial_data_vector.shape: ', self.fiducial_data_vector.shape)
        # find index of the emulation module
        module_names = [m.name for m in self.pipeline.modules]
        print('module_names: ', module_names)
        is_dynamic = [m.is_dynamic for m in self.pipeline.modules]
        print('is_dynamic: ', is_dynamic)
        last_module = self.pipeline.modules[-1]
        print('last_module: ', last_module)
        emu_index = module_names.index(self.last_emulated_module) if self.last_emulated_module else len(module_names)
        print('emu_index: ', emu_index)
        emu_modules = self.pipeline.modules[emu_index + 1 :]
        fixed_inputs = {(sec,key): block[sec, key] for (sec, key) in self.fixed_keys}
        print('fixed_inputs: ', fixed_inputs)
        fixed_vector = []
        for (sec, key) in self.fixed_keys:
            if type(block[sec, key])==float:
                fixed_vector = np.append(fixed_vector, np.array([block[sec, key]]), axis=0) if len(fixed_vector) else block[sec, key]
            else:
                fixed_vector = np.append(fixed_vector, block[sec, key], axis=0) if len(fixed_vector) else block[sec, key]
        self.fixed_vector = fixed_vector     
        for sec, key in block.keys(section="data_vector"):
            #key: chi2, covariance, data, inv_cov, log_det, n, norm, sim, theory
            if key.endswith("_chi2"):
                chi2_fid = block[sec, key]
            else:
                continue    
        print(f'Fiducial chi2 = {chi2_fid}, while chi2_cutoff = {self.chi2_cut_off}')
        # TODO exclude chi2 = -inf!!!
        emu_module = EmulatorModule.as_module("emulator")
        self.emu_module = emu_module

        if len(emu_modules)!=0:
            # if you're emulating some in-between blocks
            # or properties later interpolated in the likelihood
            emu_modules.insert(0, emu_module)
        else:    
            # if you want to override the extract_theory_points method
            # we want to train the emulator on all scales available and cut later 
            # so that the same model can be used for scale-cuts tests
            like_module = copy.copy(self.pipeline.modules[-1])
            #print([attr for attr in dir(like_module) if not attr.startswith("__")])
            self.inject_emulator_into_likemodule(like_module)
            emu_modules = [emu_module, like_module]
                
        print('emu_module_names: ', [m.name for m in emu_modules])
        is_dynamic = [m.is_dynamic for m in emu_modules]
        print('is_dynamic: ', is_dynamic)
        # Make a secondary pipeline object using the emulator
        print("Setting up emulated pipeline. This will print out the parameters again.")
        print('pipeline options: ', self.pipeline.options)
        self.emu_pipeline = LikelihoodPipeline(
            self.pipeline.options, modules=emu_modules, values=self.pipeline.values_file
        )
        
        emu_module.data.set_emulator_info({
            "fixed_inputs": fixed_inputs,
            "pipeline": self.pipeline,
            "outputs": self.keys,
            "sizes": self.data_vector_sizes,
            "nn_model": self.nn_model,
        })

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Emulated pipeline is initialised!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def generate_updated_sample(self):
        print(f"Selecting {self.resample_size} random samples from emcee chain to improve emulator")
        random_index = np.random.choice(np.arange(len(self.chain)), replace=False, size=self.resample_size)
        unit_sample = self.unit_chain[random_index]
        sample = self.chain[random_index]

        # saving additional datavectors for testing of the final emulated model (20%)
        # must guarantee to not overlap with the training set
        random_index_test = []
        while len(random_index_test) < int(0.2 * self.resample_size):
            candidate = np.random.choice(np.arange(len(self.chain)), replace=False)
            if candidate not in random_index_test:
                random_index_test.append(candidate)
        random_index_test = np.array(random_index_test)
        unit_sample_test = self.unit_chain[random_index_test]
        sample_test = self.chain[random_index_test]
        print(f"Running real original pipeline on new sample")
        # Generate the likelihood and data vectors for the new sample
        if self.pool:
            sample_results = self.pool.map(task, sample)
            sample_results_test = self.pool.map(task, sample_test)
        else:
            sample_results = list(map(task, sample))
            sample_results_test = list(map(task, sample_test))
        # append the results of the mcmc to the current sample
        sample_data_vectors = np.array([np.concatenate(s[1]) for s in sample_results if s is not None])
        self.sample = np.append(self.sample, sample, axis=0) 
        self.sample_data_vectors = np.append(self.sample_data_vectors, sample_data_vectors, axis=0)
        self.unit_sample = np.append(self.unit_sample, unit_sample, axis=0)
        self.sample_likes = np.append(self.sample_likes, [s[0] for s in sample_results if s is not None])
        self.sample_priors = np.append(self.sample_priors, [s[2] for s in sample_results if s is not None])
        self.sample_posts = np.append(self.sample_posts, [s[3] for s in sample_results if s is not None])

        sample_data_vectors_test = np.array([np.concatenate(s[1]) for s in sample_results_test if s is not None])
        self.sample_test = np.append(self.sample_test, sample_test, axis=0) if len(self.sample_test) else sample_test
        self.sample_data_vectors_test = np.append(self.sample_data_vectors_test, sample_data_vectors_test, axis=0) if len(self.sample_data_vectors_test) else sample_data_vectors_test
        self.unit_sample_test = np.append(self.unit_sample_test, unit_sample_test, axis=0) if len(self.unit_sample_test) else unit_sample_test
        self.sample_likes_test = np.append(self.sample_likes_test, [s[0] for s in sample_results_test if s is not None]) if len(self.sample_likes_test) else np.array([s[0] for s in sample_results_test if s is not None])
        self.sample_priors_test = np.append(self.sample_priors_test, [s[2] for s in sample_results_test if s is not None]) if  len(self.sample_priors_test) else np.array([s[2] for s in sample_results_test if s is not None])
        self.sample_posts_test = np.append(self.sample_posts_test, [s[3] for s in sample_results_test if s is not None]) if len(self.sample_posts_test) else np.array([s[3] for s in sample_results_test if s is not None])
        #TODO: save after each itteration for incremental PCA reduction
        if self.save_outputs == SAVE_ALL and self.iterations==(self.max_iterations - 1):
            training_set_dict = {
            **{str(param): self.sample[:, i] for i, param in enumerate(self.pipeline.varied_params)},
            **{str(param)+'--norm': self.unit_sample[:, i] for i, param in enumerate(self.pipeline.varied_params)},
            #'varied_params': {str(param): self.sample[:, i] for i, param in enumerate(self.pipeline.varied_params)},
            #'varied_params_norm': {str(param)+'--norm': self.unit_sample[:, i] for i, param in enumerate(self.pipeline.varied_params)},
            'fixed_keys':  [str(key) for key in self.fixed_keys] if self.fixed_keys else '',
            'fixed_features': self.fixed_vector,
            'output_keys':  [str(key) for key in self.keys] if self.keys else 'data_vector',
            'features_size': self.data_vector_sizes,
            'features': self.sample_data_vectors,
            'chi2': self.sample_likes,
            'priors': self.sample_priors,
            'posts': self.sample_posts
            }
            np.savez(self.save_outputs_dir + '/total_training_set.npz', **training_set_dict)

            testing_set_dict = {
            **{str(param): self.sample_test[:, i] for i, param in enumerate(self.pipeline.varied_params)},
            **{str(param)+'--norm': self.unit_sample_test[:, i] for i, param in enumerate(self.pipeline.varied_params)},
            #'varied_params': {str(param): self.sample_test[:, i] for i, param in enumerate(self.pipeline.varied_params)},
            #'varied_params_norm': {str(param)+'--norm': self.unit_sample_test[:, i] for i, param in enumerate(self.pipeline.varied_params)},
            'fixed_keys':  [str(key) for key in self.fixed_keys] if self.fixed_keys else '',
            'fixed_features': self.fixed_vector,

            'output_keys':  [str(key) for key in self.keys] if self.keys else 'data_vector',
            'features_size': self.data_vector_sizes,
            'features': self.sample_data_vectors_test,
            'chi2': self.sample_likes_test,
            'priors': self.sample_priors_test,
            'posts': self.sample_posts_test
            }
            np.savez(self.save_outputs_dir + '/total_testing_set.npz', **testing_set_dict)

        # delete initial sample for the last sampling cycle:
        #if self.iterations==(self.max_iterations - 1):
        #    print('cut out the initial sample')
        #    self.sample = self.sample[self.initial_size_cut:, :] 
        #    self.sample_data_vectors = self.sample_data_vectors[self.initial_size_cut:, :] 
        #    self.unit_sample = self.unit_sample[self.initial_size_cut:, :]
        #    self.sample_likes = self.sample_likes[self.initial_size_cut:]
        #    self.sample_priors = self.sample_priors[self.initial_size:]
        #    self.sample_posts = self.sample_posts[self.initial_size:]

            

    def get_emcee_start(self):
        # TODO: improve by removing low likelihood samples here
        # we want the last nwalker unique samples with likelihoods
        # that are within nsigma of the best
        return self.unit_sample[-self.emcee_walkers:]        

    def execute(self):
        import emcee


        if  self.trained_before:
                self.compute_fiducial_setup_emu_pipeline()
                print(f"Emulator was trained before! We proceede to the final sampling without tempering.")
                self.load_emulator()
                self.iterations = self.max_iterations - 1
        else:         
            if self.iterations == 0:
                self.compute_fiducial_setup_emu_pipeline()
                self.generate_initial_sample()
                ## for testing :
                #self.iterations = self.max_iterations -1
            else:
                self.generate_updated_sample()

            print(f"Training emulator (iteration {self.iterations+1} /  {self.max_iterations})")
            self.train_emulator()

        '''
        #test a single likelihood computation: 
        #p_ini_norm = self.unit_sample[-3:] 
        p_ini_norm = [[0.5588266,  0.61595163], [0.54974179, 0.61720341],  [0.53829597, 0.61912687] ]
        p_ini = np.array(
            [self.pipeline.denormalize_vector_from_prior(p) for p in p_ini_norm]
        )
        #p_ini = np.array([[0.31176532,0.84638065], [0.30994836, 0.84688136], [0.30765919, 0.84765075]])
        print('test a single likelihood computation: ')
        print('with emulator:')
        for p in p_ini:
            print('p_ini: ', p)
            r = sampler.emu_pipeline.run_results(p)
            #block = r.block
            #for sec, key in block.keys(section="data_vector"):
            #    if not key.endswith("_theory"):
            #        continue
            #    print(r.block[sec, key])
            print('example posterior: ', r.post)
        print('log_probability_function: ')    
        for p in p_ini_norm:    
            log_probability_function(p, 1)

        print('without emulator:')    
        for p in p_ini:
            print('p_ini: ', p)
            r = sampler.pipeline.run_results(p)
            print('example posterior: ', r.post)
        '''
        


        if self.iterations < self.max_iterations - 1:
            print(f"Running emcee with tempering ({self.tempering[self.iterations]}) - iteration {self.iterations+1}")
            tempering = self.tempering[self.iterations]
        else:
            print(f"Running final emcee without tempering (1) - iteration {self.iterations+1}")
            tempering = 1


        # run an mcmc using the current sample
        sampler = emcee.EnsembleSampler(
            self.emcee_walkers,
            self.ndim,
            log_probability_function,
            args=[tempering],
            pool=self.pool,
        )

        start_pos = [self.pipeline.randomized_start()
                           for i in range(self.emcee_walkers)] if self.trained_before else self.get_emcee_start() 
        print('start_pos: ', start_pos)
        sampler.run_mcmc(start_pos, self.emcee_samples, progress=True)
        
        
        if self.emcee_burn < 1:
            burn = int(self.emcee_burn * self.emcee_samples)
        else:
            burn = int(self.emcee_burn)

        # the chain is in the unit cube
        self.unit_chain = sampler.get_chain(discard=burn, thin=self.emcee_thin, flat=True)
        logp = sampler.get_log_prob(discard=burn, thin=self.emcee_thin, flat=True)
        # derived parameters
        self.blobs = sampler.get_blobs(discard=burn, thin=self.emcee_thin, flat=True)

        self.chain = np.array(
            [self.pipeline.denormalize_vector_from_prior(p) for p in self.unit_chain]
        )
        # We discard the previous chain contents
        if self.save_outputs == SAVE_ALL and 0 < self.iterations < self.max_iterations:
            self.output.save_and_reset_to_chain_start(f'_tempering_{self.tempering[self.iterations-1]}_iteration_{self.iterations}')
        else:
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



def extract_theory_points_with_emu(self):
    y = np.zeros(self.data_x.size)
    return y