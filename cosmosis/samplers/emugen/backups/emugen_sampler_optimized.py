"""
EmugenSampler: An emulator-accelerated MCMC sampler for CosmoSIS

This module implements an iterative emulator-based sampler that uses neural networks
to accelerate cosmological parameter estimation. The sampler alternates between:
1. Training neural network emulators on exact pipeline calculations
2. Running MCMC using the fast emulator predictions
3. Improving the emulator with new training data from high-likelihood regions

Authors: CosmoSIS Team
License: BSD 2-Clause
"""

import itertools
import logging
import os
import errno
import types
import copy
from timeit import default_timer
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from ...output.text_output import TextColumnOutput
from .. import ParallelSampler
from cosmosis.gaussian_likelihood import GaussianLikelihood
from ...runtime import LikelihoodPipeline, ClassModule

# Configure logging
logger = logging.getLogger(__name__)

# Save output modes
SAVE_NONE = 0   # Save nothing (not recommended)
SAVE_MODEL = 1  # Save only final trained emulator
SAVE_ALL = 2    # Save all training data, models, and diagnostics


def mkdir(path: str) -> None:
    """Ensure that all the components in the `path` exist in the file system.
    
    Args:
        path: Directory path to create
        
    Raises:
        ValueError: If path conflicts with existing files or directory structure
    """
    # Avoid trying to make an empty path
    if not path.strip():
        return
        
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        if error.errno == errno.EEXIST:
            if not os.path.isdir(path):
                raise ValueError(f"Tried to create directory {path} but file with same name exists")
        elif error.errno == errno.ENOTDIR:
            raise ValueError(f"Tried to create directory {path} but part of path exists as a file")
        else:
            raise ValueError(f"Failed to create directory {path}: {error}")


def task(p: np.ndarray, return_all: bool = False) -> Optional[Union[Tuple, Any]]:
    """Execute pipeline for given parameters and extract relevant data vectors.
    
    This function runs the full cosmological pipeline for a given parameter vector
    and extracts the theory predictions, data vectors, and likelihood information
    needed for emulator training or MCMC sampling.
    
    Args:
        p: Parameter vector to evaluate
        return_all: If True, return additional data for emulator training
                   (covariance matrices, error vectors, etc.)
                   
    Returns:
        If return_all=False: (likelihood, data_vectors_theory, prior, posterior)
        If return_all=True: (likelihood, data_vectors_theory, data_vectors, 
                           inv_covariance, error_vectors, block)
        None if pipeline execution failed
    """
    try:
        r = sampler.pipeline.run_results(p)
        block = r.block
        
        if block is None:
            logger.warning(f"Pipeline execution failed for parameters: {p}")
            return None
            
    except Exception as e:
        logger.error(f"Pipeline execution error for parameters {p}: {e}")
        return None
    
    data_vectors_theory = []
    data_vectors = []
    error_vectors = []
    data_inv_covariance = []
    
    if sampler.keys:
        # User has specified which keys to emulate
        for sec, key in sampler.keys:
            value = block[sec, key]
            # Ensure value is always an array for consistency
            if isinstance(value, (int, float)):
                data_vectors_theory.append(np.array([value]))
            else:
                data_vectors_theory.append(np.asarray(value))
                
        if return_all:
            if sampler.error_keys:
                for sec, key in sampler.error_keys:
                    error_vectors.append(np.asarray(block[sec, key]))
            else:
                # Default to unit errors if none specified
                for d in data_vectors_theory:
                    error_vectors.append(np.ones_like(d))
    else:
        # Use all theory predictions from data_vector section
        for sec, key in block.keys(section="data_vector"):
            if not key.endswith("_theory"):
                continue
                
            data_vectors_theory.append(r.block[sec, key])
            
            if return_all:
                # Extract corresponding data and covariance
                base_key = key[:-7]  # Remove '_theory' suffix
                try:
                    covmat = block[sec, base_key + "_covariance"]
                    sigma = np.sqrt(np.diag(covmat))
                    error_vectors.append(sigma)
                    data_inv_covariance.append(block[sec, base_key + "_inverse_covariance"])
                    data_vectors.append(block[sec, base_key + "_data"])
                except KeyError as e:
                    logger.warning(f"Missing covariance data for {base_key}: {e}")
                    # Use unit errors as fallback
                    error_vectors.append(np.ones_like(data_vectors_theory[-1]))
       
    if return_all:
        if len(error_vectors) != len(data_vectors_theory):
            raise ValueError(f"Mismatch: {len(error_vectors)} error vectors vs "
                           f"{len(data_vectors_theory)} theory vectors")
        return r.like, data_vectors_theory, data_vectors, data_inv_covariance, error_vectors, r.block
    else:
        return r.like, data_vectors_theory, r.prior, r.post


def log_probability_function(u: np.ndarray, tempering: float) -> Tuple[float, Tuple[float, List[float]]]:
    """Log probability function for emcee sampler using emulated pipeline.
    
    Args:
        u: Parameter vector in unit hypercube [0,1]^ndim
        tempering: Tempering factor to apply to posterior (0 < tempering <= 1)
                  Lower values flatten the likelihood for better exploration
                  
    Returns:
        Tuple of (tempered_posterior, (prior, extra_parameters))
        Returns (-inf, (-inf, [nan, ...])) if parameters are outside prior bounds
    """
    try:
        # Transform from unit hypercube to physical parameter space
        p = sampler.pipeline.denormalize_vector_from_prior(u)
    except ValueError:
        # Parameters outside prior bounds
        n_extra = sampler.pipeline.number_extra
        return (-np.inf, (-np.inf, [np.nan] * n_extra))
    
    try:
        # Run emulated pipeline
        r = sampler.emu_pipeline.run_results(p)
        return tempering * r.post, (r.prior, r.extra)
    except Exception as e:
        logger.error(f"Emulated pipeline execution failed for parameters {p}: {e}")
        n_extra = sampler.pipeline.number_extra
        return (-np.inf, (-np.inf, [np.nan] * n_extra))


class EmulatorModule(ClassModule):
    """CosmoSIS module that replaces pipeline calculations with emulator predictions.
    
    This module intercepts the pipeline execution and replaces slow calculations
    with fast neural network predictions. It handles parameter transformation,
    emulator prediction, and output formatting.
    """
    
    def __init__(self, options: Dict[str, Any]) -> None:
        """Initialize emulator module.
        
        Args:
            options: Configuration options (currently unused)
        """
        super().__init__(options)
        self.emulator = None
        self.pipeline = None
        self.fixed_inputs = {}
        self.inputs = []
        self.outputs = []
        self.sizes = []
        self.nn_model = None

    def set_emulator_info(self, info: Dict[str, Any]) -> None:
        """Set emulator configuration information.
        
        Args:
            info: Dictionary containing:
                - pipeline: Original CosmoSIS pipeline object
                - fixed_inputs: Parameters that don't vary during emulation
                - outputs: List of (section, key) tuples for emulator outputs
                - sizes: Sizes of each output vector
                - nn_model: Neural network model type
        """
        self.pipeline = info["pipeline"]
        self.fixed_inputs = info["fixed_inputs"]
        self.inputs = [(p.section, p.name) for p in self.pipeline.varied_params]
        self.outputs = info["outputs"]
        self.sizes = info["sizes"]
        self.nn_model = info["nn_model"]

    def set_emulator(self, emu: Any) -> None:
        """Set the trained emulator object.
        
        Args:
            emu: Trained emulator object with predict() method
        """
        self.emulator = emu

    def execute(self, block: Any) -> int:
        """Execute emulator prediction and populate data block.
        
        Args:
            block: CosmoSIS data block to populate with predictions
            
        Returns:
            0 on success (CosmoSIS convention)
            
        Raises:
            RuntimeError: If emulator is not set or prediction fails
        """
        if self.emulator is None:
            raise RuntimeError("Emulator not set - call set_emulator() first")
        
        try:
            # Prepare input dictionary for emulator
            # Use '--' separator to avoid conflicts with parameter names
            p_dict = {f"{sec}--{key}": block[sec, key] for (sec, key) in self.inputs}
            
            # Get emulator prediction
            prediction = self.emulator.predict(p_dict)[0]
            
            # Populate outputs
            if not self.outputs:  # Empty outputs means full data vector
                block["data_vector", "theory_emulated"] = prediction
            else:
                # Split prediction into specified output components
                start_idx = 0
                for (sec, key), size in zip(self.outputs, self.sizes):
                    block[sec, key] = prediction[start_idx:start_idx + size]
                    start_idx += size

            # Set fixed inputs that don't vary during emulation
            for (sec, key), val in self.fixed_inputs.items():
                block[sec, key] = val
                
            return 0
            
        except Exception as e:
            logger.error(f"Emulator execution failed: {e}")
            raise RuntimeError(f"Emulator prediction failed: {e}")


class EmugenSampler(ParallelSampler):
    """Emulator-accelerated MCMC sampler for CosmoSIS.
    
    This sampler uses neural network emulators to speed up cosmological parameter
    estimation by 10-1000x. It works by:
    
    1. Generating an initial training set with exact (slow) pipeline calculations
    2. Training a neural network emulator on this data
    3. Running MCMC using the fast emulator for likelihood evaluation
    4. Iteratively improving the emulator with new training data
    5. Final sampling with the best emulator
    
    The sampler supports:
    - Partial pipeline emulation (specify last_emulated_module)
    - Custom data vector components (specify keys)
    - Likelihood tempering for better exploration
    - Comprehensive diagnostics and model saving
    - Reuse of pre-trained emulators
    
    Attributes:
        parallel_output: Whether to use parallel output (always False)
        sampler_outputs: Output columns for chains
    """
    
    parallel_output = False
    sampler_outputs = [("prior", float), ("tempered_post", float), ("post", float)]

    def config(self) -> None:
        """Configure the emulator sampler from ini file parameters.
        
        This method reads all configuration parameters from the ini file and
        sets up the sampler for training and MCMC sampling. It validates
        parameters and sets up the global sampler reference needed by helper functions.
        
        Raises:
            ValueError: If configuration parameters are invalid or inconsistent
        """
        global sampler
        sampler = self
        self.converged = False
        
        # Parse emulation target settings
        keys = self.read_ini("keys", str, "")
        fixed_keys = self.read_ini("fixed_keys", str, "")
        error_keys = self.read_ini("error_keys", str, "")
        
        # Convert space-separated strings to lists of (section, key) tuples
        self.keys = [k.split(".") for k in keys.split()] if keys else []
        self.fixed_keys = [k.split(".") for k in fixed_keys.split()] if fixed_keys else []
        self.error_keys = [k.split(".") for k in error_keys.split()] if error_keys else []
        
        # Configure output saving
        self._configure_output_saving()
        
        # Configure emulator loading
        self.load_emu_filename = self.read_ini("load_emu_filename", str, "")
        self.trained_before = self.read_ini("trained_before", bool, False)
        
        if self.trained_before and not self.load_emu_filename:
            raise ValueError("trained_before=true requires load_emu_filename to be specified")
        
        # Configure training parameters
        self._configure_training_parameters()
        
        # Configure MCMC parameters
        self._configure_mcmc_parameters()
        
        # Configure advanced options
        self._configure_advanced_options()
        
        # Initialize state
        self.ndim = len(self.pipeline.varied_params)
        self.emu_pipeline = None
        self.iterations = 0
        
        logger.info(f"EmugenSampler configured with {self.ndim} parameters, "
                   f"{self.max_iterations} iterations, initial training size {self.initial_size}")
    
    def _configure_output_saving(self) -> None:
        """Configure output saving options."""
        save_outputs = self.read_ini("save_outputs", str, "")
        
        if save_outputs:
            self.save_outputs_dir = self.read_ini("save_outputs_dir", str, "")
            if not self.save_outputs_dir:
                raise ValueError("save_outputs_dir must be specified when save_outputs is set")
                
            mkdir(self.save_outputs_dir)
            
            if save_outputs == "model":
                self.save_outputs = SAVE_MODEL
            elif save_outputs == "all":
                self.save_outputs = SAVE_ALL
            else:
                raise ValueError(f"Unknown save_outputs option '{save_outputs}' - "
                               "should be 'model', 'all', or empty")
        else:
            self.save_outputs = SAVE_NONE
            logger.warning("No outputs will be saved (save_outputs not specified)")
    
    def _configure_training_parameters(self) -> None:
        """Configure neural network training parameters."""
        self.max_iterations = self.read_ini("iterations", int, 4)
        self.initial_size = self.read_ini("initial_size", int, 9600)
        self.resample_size = self.read_ini("resample_size", int, 4800)
        self.chi2_cut_off = self.read_ini("chi2_cut_off", float)
        self.batch_size = self.read_ini("batch_size", int, 32)
        self.training_iterations = self.read_ini("training_iterations", int, 5)
        
        # Validate training parameters
        if self.max_iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.initial_size < 10:
            raise ValueError("initial_size must be >= 10")
        if self.resample_size < 1:
            raise ValueError("resample_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.training_iterations < 1:
            raise ValueError("training_iterations must be >= 1")
    
    def _configure_mcmc_parameters(self) -> None:
        """Configure MCMC sampling parameters."""
        self.emcee_walkers = self.read_ini("emcee_walkers", int)
        self.emcee_samples = self.read_ini("emcee_samples", int)
        self.emcee_burn = self.read_ini("emcee_burn", float, 0.3)
        self.emcee_thin = self.read_ini("emcee_thin", int, 1)
        
        # Validate MCMC parameters
        if self.emcee_walkers >= self.initial_size:
            raise ValueError("emcee_walkers must be < initial_size for proper initialization")
        if self.emcee_walkers < 2 * self.ndim:
            logger.warning(f"emcee_walkers ({self.emcee_walkers}) < 2*ndim ({2*self.ndim}) "
                          "may lead to poor sampling")
        if self.emcee_samples < 100:
            logger.warning(f"emcee_samples ({self.emcee_samples}) is very small")
        if not (0 <= self.emcee_burn <= 1) and self.emcee_burn >= self.emcee_samples:
            raise ValueError("emcee_burn must be fraction in [0,1] or integer < emcee_samples")
        if self.emcee_thin < 1:
            raise ValueError("emcee_thin must be >= 1")
    
    def _configure_advanced_options(self) -> None:
        """Configure advanced and experimental options."""
        # Pipeline emulation settings
        self.last_emulated_module = self.read_ini("last_emulated_module", str, "")
        
        # Tempering settings
        tempering = self.read_ini("tempering", float, 0.05)
        self.tempering = np.full(self.max_iterations, tempering)
        
        tempering_file = self.read_ini("tempering_file", str, "")
        if tempering_file:
            try:
                custom_tempering = np.genfromtxt(tempering_file)
                if len(custom_tempering) < self.max_iterations:
                    logger.warning(f"Tempering file has {len(custom_tempering)} values but "
                                 f"{self.max_iterations} iterations requested")
                self.tempering = custom_tempering[:self.max_iterations]
            except Exception as e:
                raise ValueError(f"Failed to read tempering file {tempering_file}: {e}")
        
        logger.info(f"Tempering schedule: {self.tempering}")
        
        # Random seed
        self.seed = self.read_ini("seed", int, 0)
        if self.seed == 0:
            self.seed = None  # Use random seed
        
        # Neural network options (currently fixed)
        self.data_trafo = self.read_ini("data_trafo", str, "log_norm")
        self.n_pca = self.read_ini("n_pca", int, 32)
        self.loss_function = self.read_ini("loss_function", str, "default")
        
        if self.loss_function.startswith("weighted") and self.keys:
            raise ValueError("Weighted loss function can only be used with full data vector "
                           "(empty keys parameter)")
        
        # Neural network architecture
        self.nn_model = 'MLP'  # Could be made configurable in future
        if self.nn_model not in ['MLP', 'ResMLP']:
            raise ValueError(f"Unknown nn_model '{self.nn_model}' - should be 'MLP' or 'ResMLP'")

    def generate_initial_sample(self) -> None:
        """Generate initial training sample using Latin Hypercube sampling.
        
        This method creates the initial training dataset by:
        1. Sampling parameters from priors using Latin Hypercube
        2. Running the full pipeline for each parameter set
        3. Filtering out poor fits based on chi2 cutoff
        4. Storing results for emulator training
        """
        import scipy.stats
        
        logger.info("Generating initial training sample")
        
        # Generate Latin Hypercube sample in unit cube
        hypercube = scipy.stats.qmc.LatinHypercube(self.ndim, seed=self.seed)
        unit_sample = hypercube.random(n=self.initial_size)
        
        # Transform to physical parameter space
        sample = np.array([
            self.pipeline.denormalize_vector_from_prior(p) for p in unit_sample
        ])
        
        logger.info(f"Generated {len(sample)} parameter combinations")
        logger.info(f"Parallelization enabled: {self.pool is not None}")
        
        # Run pipeline for all samples
        start_time = default_timer()
        if self.pool:
            sample_results = self.pool.map(task, sample)
        else:
            sample_results = list(map(task, sample))
        
        end_time = default_timer()
        logger.info(f"Initial sample evaluation took {end_time - start_time:.1f} seconds")
        
        # Process results and apply chi2 cutoff
        self._process_initial_results(sample_results, sample, unit_sample)
        
        # Initialize test sample storage
        self._initialize_test_storage()

    def _process_initial_results(self, sample_results: List, sample: np.ndarray, 
                                unit_sample: np.ndarray) -> None:
        """Process initial sample results and apply chi2 filtering."""
        # Extract successful results
        valid_results = [s for s in sample_results if s is not None]
        
        if not valid_results:
            raise RuntimeError("No valid pipeline evaluations in initial sample")
        
        # Extract components
        sample_likes = np.array([s[0] for s in valid_results])
        sample_data_vectors = np.array([np.concatenate(s[1]) for s in valid_results])
        sample_priors = np.array([s[2] for s in valid_results])
        sample_posts = np.array([s[3] for s in valid_results])
        
        # Apply chi2 cutoff
        chi2_values = -2 * sample_likes
        cut = chi2_values < self.chi2_cut_off
        
        logger.info(f"Data vector shape: {sample_data_vectors.shape}")
        logger.info(f"Chi2 range: [{chi2_values.min():.1f}, {chi2_values.max():.1f}]")
        
        # Filter arrays
        self.sample_likes = sample_likes[cut]
        self.sample_priors = sample_priors[cut]
        self.sample_posts = sample_posts[cut]
        self.sample_data_vectors = sample_data_vectors[cut]
        
        # Filter corresponding parameter arrays
        valid_indices = np.arange(len(sample_results))[np.array([s is not None for s in sample_results])]
        filtered_indices = valid_indices[cut]
        self.sample = sample[filtered_indices]
        self.unit_sample = unit_sample[filtered_indices]
        
        n_kept = len(self.sample_likes)
        n_total = len(sample_results)
        logger.info(f"Kept {n_kept}/{n_total} samples after chi2 < {self.chi2_cut_off} filter")
        
        if n_kept < self.emcee_walkers:
            raise RuntimeError(f"Only {n_kept} samples passed chi2 filter, but need "
                             f"at least {self.emcee_walkers} for MCMC walkers")
        
        self.initial_size_cut = n_kept

    def _initialize_test_storage(self) -> None:
        """Initialize storage for test samples."""
        self.sample_test = np.array([]).reshape(0, self.ndim)
        self.sample_data_vectors_test = np.array([]).reshape(0, self.sample_data_vectors.shape[1])
        self.unit_sample_test = np.array([]).reshape(0, self.ndim)
        self.sample_likes_test = np.array([])
        self.sample_priors_test = np.array([])
        self.sample_posts_test = np.array([])

    def train_emulator(self) -> None:
        """Train neural network emulator on current training data.
        
        This method:
        1. Imports the emulator class
        2. Sets up training parameters
        3. Trains the neural network
        4. Updates the emulated pipeline
        """
        from .cosmopower import CPEmulator
        
        n_samp, n_in = self.unit_sample.shape
        n_out = self.sample_data_vectors.shape[1]
        
        logger.info(f"Training emulator: {n_in} parameters -> {n_out} outputs "
                   f"using {n_samp} training points")
        
        # Set up training parameters
        model_filename = f'{self.save_outputs_dir}/emumodel_{self.iterations+1}'
        kwargs = {
            "model_filename": model_filename,
            "n_cycles": self.training_iterations,
            "batch_size": self.batch_size * (self.iterations + 1)
        }
        
        # Create emulator instance
        model_parameters = [str(param) for param in self.pipeline.varied_params]
        logger.info(f"Model parameters: {model_parameters}")
        
        emu = CPEmulator(
            model_parameters, 
            np.arange(n_out), 
            self.nn_model, 
            self.iterations + 1,
            self.data_trafo, 
            self.n_pca, 
            self.data, 
            self.inv_cov
        )
        
        # Prepare training data
        X = {str(param): self.sample[:, i] 
             for i, param in enumerate(self.pipeline.varied_params)}
        
        # Train emulator
        start_time = default_timer()
        emu.train(X, self.sample_data_vectors, **kwargs)
        end_time = default_timer()
        
        logger.info(f"Emulator training took {end_time - start_time:.1f} seconds")
        
        # Store emulator and update pipeline
        self.emulator = emu
        self.emu_module.data.set_emulator(emu)

    def load_emulator(self) -> None:
        """Load pre-trained emulator from file.
        
        This method loads a previously trained emulator, useful for
        parameter studies or continuing interrupted runs.
        """
        from .cosmopower import CPEmulator
        
        if not os.path.exists(self.load_emu_filename):
            raise FileNotFoundError(f"Emulator file not found: {self.load_emu_filename}")
        
        logger.info(f"Loading pre-trained emulator from {self.load_emu_filename}")
        
        model_parameters = [str(param) for param in self.pipeline.varied_params]
        emu = CPEmulator(model_parameters, np.ones(10))  # Dummy output size
        
        # Load trained model
        emu.load(self.load_emu_filename)
        
        self.emulator = emu
        self.emu_module.data.set_emulator(emu)
        
        logger.info("Pre-trained emulator loaded successfully")

    def inject_emulator_into_likemodule(self, module: Any) -> None:
        """Inject emulator into likelihood module by monkey-patching.
        
        This method modifies the likelihood module to use emulator predictions
        instead of extracting theory points from the pipeline.
        
        Args:
            module: Likelihood module to modify
        """
        original_setup = module.setup_function

        def setup_wrapper(config):
            # Call original setup to get instance
            instance = original_setup(config)
            
            # Monkey-patch the extract_theory_points method
            def emulated_extract_theory_points(self, block):
                return block["data_vector", "theory_emulated"]
            
            instance.extract_theory_points = types.MethodType(
                emulated_extract_theory_points, instance
            )
            
            return instance

        # Replace the setup function
        module.setup_function = setup_wrapper

    def compute_fiducial_setup_emu_pipeline(self) -> None:
        """Compute fiducial data vector and set up emulated pipeline.
        
        This method:
        1. Computes fiducial model at parameter center
        2. Determines pipeline structure for emulation
        3. Sets up emulated pipeline with emulator module
        4. Configures fixed inputs and outputs
        """
        logger.info("Computing fiducial data vector and setting up emulated pipeline")
        
        # Get fiducial parameter vector
        p = self.pipeline.start_vector()
        p_unit = self.pipeline.normalize_vector(p)
        
        # Run full pipeline to get fiducial results
        _, data_vectors, self.data, self.inv_cov, errors, block = task(p, return_all=True)
        
        # Process data vectors
        self.data_vector_sizes = [len(x) for x in data_vectors]
        self.fiducial_data_vector = np.concatenate(data_vectors)
        self.fiducial_errors = np.concatenate(errors)
        
        logger.info(f"Fiducial data vector shape: {self.fiducial_data_vector.shape}")
        
        # Determine emulation structure
        self._setup_emulation_structure(block)
        
        # Create emulated pipeline
        self._create_emulated_pipeline()
        
        logger.info("Emulated pipeline setup complete")

    def _setup_emulation_structure(self, block: Any) -> None:
        """Set up the structure for emulation based on pipeline modules."""
        # Get module information
        module_names = [m.name for m in self.pipeline.modules]
        logger.info(f"Pipeline modules: {module_names}")
        
        # Find emulation cutoff point
        if self.last_emulated_module:
            try:
                emu_index = module_names.index(self.last_emulated_module)
            except ValueError:
                raise ValueError(f"Module '{self.last_emulated_module}' not found in pipeline")
        else:
            emu_index = len(module_names)  # Emulate entire pipeline
        
        logger.info(f"Emulation cutoff at module index: {emu_index}")
        
        # Get modules to include in emulated pipeline
        emu_modules = self.pipeline.modules[emu_index + 1:]
        
        # Set up fixed inputs
        fixed_inputs = {(sec, key): block[sec, key] for (sec, key) in self.fixed_keys}
        logger.info(f"Fixed inputs: {list(fixed_inputs.keys())}")
        
        # Build fixed vector for emulator
        self.fixed_vector = self._build_fixed_vector(block)
        
        # Get fiducial chi2 for diagnostics
        self._extract_fiducial_chi2(block)
        
        # Create emulator module
        emu_module = EmulatorModule.as_module("emulator")
        self.emu_module = emu_module
        
        # Configure emulated modules
        if emu_modules:
            # Partial pipeline emulation
            emu_modules.insert(0, emu_module)
        else:
            # Full pipeline emulation - modify likelihood module
            like_module = copy.copy(self.pipeline.modules[-1])
            self.inject_emulator_into_likemodule(like_module)
            emu_modules = [emu_module, like_module]
        
        self.emu_modules = emu_modules
        self.fixed_inputs = fixed_inputs

    def _build_fixed_vector(self, block: Any) -> np.ndarray:
        """Build vector of fixed parameters for emulator."""
        fixed_vector = []
        
        for (sec, key) in self.fixed_keys:
            value = block[sec, key]
            if isinstance(value, (int, float)):
                fixed_vector.append(value)
            else:
                fixed_vector.extend(np.asarray(value).flatten())
        
        return np.array(fixed_vector) if fixed_vector else np.array([])

    def _extract_fiducial_chi2(self, block: Any) -> None:
        """Extract fiducial chi2 for diagnostics."""
        chi2_fid = None
        for sec, key in block.keys(section="data_vector"):
            if key.endswith("_chi2"):
                chi2_fid = block[sec, key]
                break
        
        if chi2_fid is not None:
            logger.info(f"Fiducial chi2: {chi2_fid:.2f}, cutoff: {self.chi2_cut_off}")
        else:
            logger.warning("No chi2 value found in fiducial evaluation")

    def _create_emulated_pipeline(self) -> None:
        """Create the emulated pipeline object."""
        logger.info("Creating emulated pipeline")
        
        self.emu_pipeline = LikelihoodPipeline(
            self.pipeline.options,
            modules=self.emu_modules,
            values=self.pipeline.values_file
        )
        
        # Configure emulator module
        self.emu_module.data.set_emulator_info({
            "fixed_inputs": self.fixed_inputs,
            "pipeline": self.pipeline,
            "outputs": self.keys,
            "sizes": self.data_vector_sizes,
            "nn_model": self.nn_model,
        })

    def generate_updated_sample(self) -> None:
        """Generate additional training samples from MCMC chain.
        
        This method selects high-likelihood samples from the current MCMC chain,
        evaluates them with the exact pipeline, and adds them to the training set.
        """
        logger.info(f"Selecting {self.resample_size} samples from MCMC chain for training")
        
        # Select random samples from chain
        chain_length = len(self.chain)
        random_indices = np.random.choice(
            chain_length, size=self.resample_size, replace=False
        )
        
        unit_sample = self.unit_chain[random_indices]
        sample = self.chain[random_indices]
        
        # Select test samples (20% of resample size)
        test_size = max(1, int(0.2 * self.resample_size))
        test_indices = self._select_non_overlapping_indices(chain_length, random_indices, test_size)
        
        unit_sample_test = self.unit_chain[test_indices]
        sample_test = self.chain[test_indices]
        
        logger.info(f"Running exact pipeline on {len(sample)} training + {len(sample_test)} test samples")
        
        # Evaluate samples with exact pipeline
        start_time = default_timer()
        if self.pool:
            sample_results = self.pool.map(task, sample)
            sample_results_test = self.pool.map(task, sample_test)
        else:
            sample_results = list(map(task, sample))
            sample_results_test = list(map(task, sample_test))
        
        end_time = default_timer()
        logger.info(f"Sample evaluation took {end_time - start_time:.1f} seconds")
        
        # Update training and test sets
        self._update_training_set(sample_results, sample, unit_sample)
        self._update_test_set(sample_results_test, sample_test, unit_sample_test)
        
        # Save datasets if requested
        if self.save_outputs == SAVE_ALL and self.iterations == (self.max_iterations - 1):
            self._save_datasets()

    def _select_non_overlapping_indices(self, chain_length: int, 
                                      excluded_indices: np.ndarray, 
                                      n_select: int) -> np.ndarray:
        """Select indices that don't overlap with excluded set."""
        available_indices = np.setdiff1d(np.arange(chain_length), excluded_indices)
        
        if len(available_indices) < n_select:
            logger.warning(f"Only {len(available_indices)} non-overlapping indices available, "
                          f"requested {n_select}")
            n_select = len(available_indices)
        
        return np.random.choice(available_indices, size=n_select, replace=False)

    def _update_training_set(self, sample_results: List, sample: np.ndarray, 
                           unit_sample: np.ndarray) -> None:
        """Update training set with new samples."""
        valid_results = [s for s in sample_results if s is not None]
        
        if not valid_results:
            logger.warning("No valid results in training sample update")
            return
        
        # Extract new data
        new_data_vectors = np.array([np.concatenate(s[1]) for s in valid_results])
        new_likes = np.array([s[0] for s in valid_results])
        new_priors = np.array([s[2] for s in valid_results])
        new_posts = np.array([s[3] for s in valid_results])
        
        # Filter for valid indices
        valid_mask = np.array([s is not None for s in sample_results])
        valid_sample = sample[valid_mask]
        valid_unit_sample = unit_sample[valid_mask]
        
        # Append to existing arrays
        self.sample = np.vstack([self.sample, valid_sample])
        self.sample_data_vectors = np.vstack([self.sample_data_vectors, new_data_vectors])
        self.unit_sample = np.vstack([self.unit_sample, valid_unit_sample])
        self.sample_likes = np.concatenate([self.sample_likes, new_likes])
        self.sample_priors = np.concatenate([self.sample_priors, new_priors])
        self.sample_posts = np.concatenate([self.sample_posts, new_posts])

    def _update_test_set(self, sample_results_test: List, sample_test: np.ndarray,
                        unit_sample_test: np.ndarray) -> None:
        """Update test set with new samples."""
        valid_results = [s for s in sample_results_test if s is not None]
        
        if not valid_results:
            logger.warning("No valid results in test sample update")
            return
        
        # Extract new test data
        new_data_vectors = np.array([np.concatenate(s[1]) for s in valid_results])
        new_likes = np.array([s[0] for s in valid_results])
        new_priors = np.array([s[2] for s in valid_results])
        new_posts = np.array([s[3] for s in valid_results])
        
        # Filter for valid indices
        valid_mask = np.array([s is not None for s in sample_results_test])
        valid_sample = sample_test[valid_mask]
        valid_unit_sample = unit_sample_test[valid_mask]
        
        # Append to test arrays
        self.sample_test = np.vstack([self.sample_test, valid_sample])
        self.sample_data_vectors_test = np.vstack([self.sample_data_vectors_test, new_data_vectors])
        self.unit_sample_test = np.vstack([self.unit_sample_test, valid_unit_sample])
        self.sample_likes_test = np.concatenate([self.sample_likes_test, new_likes])
        self.sample_priors_test = np.concatenate([self.sample_priors_test, new_priors])
        self.sample_posts_test = np.concatenate([self.sample_posts_test, new_posts])

    def _save_datasets(self) -> None:
        """Save training and test datasets."""
        logger.info("Saving training and test datasets")
        
        # Training set
        training_dict = self._build_dataset_dict(
            self.sample, self.unit_sample, self.sample_data_vectors,
            self.sample_likes, self.sample_priors, self.sample_posts
        )
        np.savez(f'{self.save_outputs_dir}/total_training_set.npz', **training_dict)
        
        # Test set
        test_dict = self._build_dataset_dict(
            self.sample_test, self.unit_sample_test, self.sample_data_vectors_test,
            self.sample_likes_test, self.sample_priors_test, self.sample_posts_test
        )
        np.savez(f'{self.save_outputs_dir}/total_testing_set.npz', **test_dict)

    def _build_dataset_dict(self, sample: np.ndarray, unit_sample: np.ndarray,
                           data_vectors: np.ndarray, likes: np.ndarray,
                           priors: np.ndarray, posts: np.ndarray) -> Dict[str, Any]:
        """Build dataset dictionary for saving."""
        param_names = [str(param) for param in self.pipeline.varied_params]
        
        dataset_dict = {}
        
        # Add parameter arrays
        for i, param in enumerate(param_names):
            dataset_dict[param] = sample[:, i]
            dataset_dict[f"{param}--norm"] = unit_sample[:, i]
        
        # Add metadata and results
        dataset_dict.update({
            'fixed_keys': [str(key) for key in self.fixed_keys] if self.fixed_keys else '',
            'fixed_features': self.fixed_vector,
            'output_keys': [str(key) for key in self.keys] if self.keys else 'data_vector',
            'features_size': self.data_vector_sizes,
            'features': data_vectors,
            'chi2': likes,
            'priors': priors,
            'posts': posts
        })
        
        return dataset_dict

    def get_emcee_start(self) -> np.ndarray:
        """Get starting positions for MCMC walkers.
        
        Returns:
            Array of starting positions in unit hypercube
            
        TODO: Improve by selecting high-likelihood samples and adding noise
        """
        if len(self.unit_sample) < self.emcee_walkers:
            raise RuntimeError(f"Not enough training samples ({len(self.unit_sample)}) "
                             f"for {self.emcee_walkers} walkers")
        
        # For now, just take the last N samples
        # TODO: Select best samples and add small random perturbations
        return self.unit_sample[-self.emcee_walkers:]

    def execute(self) -> None:
        """Execute one iteration of the emulator sampler.
        
        This method performs one complete iteration:
        1. Training (if not using pre-trained emulator)
        2. MCMC sampling with current emulator
        3. Output processing and chain storage
        """
        import emcee
        
        # Handle pre-trained emulator case
        if self.trained_before:
            self.compute_fiducial_setup_emu_pipeline()
            logger.info("Using pre-trained emulator, proceeding to final sampling")
            self.load_emulator()
            self.iterations = self.max_iterations - 1
        else:
            # Normal training workflow
            if self.iterations == 0:
                # First iteration: setup and initial training
                self.compute_fiducial_setup_emu_pipeline()
                self.generate_initial_sample()
            else:
                # Subsequent iterations: update training set
                self.generate_updated_sample()
            
            # Train emulator
            logger.info(f"Training emulator (iteration {self.iterations + 1}/{self.max_iterations})")
            self.train_emulator()
        
        # Set up MCMC sampling
        tempering = self._get_current_tempering()
        
        # Create emcee sampler
        emcee_sampler = emcee.EnsembleSampler(
            self.emcee_walkers,
            self.ndim,
            log_probability_function,
            args=[tempering],
            pool=self.pool,
        )
        
        # Get starting positions
        if self.trained_before:
            start_pos = [self.pipeline.randomized_start() for _ in range(self.emcee_walkers)]
        else:
            start_pos = self.get_emcee_start()
        
        logger.info(f"Starting MCMC with {len(start_pos)} walkers")
        
        # Run MCMC
        start_time = default_timer()
        emcee_sampler.run_mcmc(start_pos, self.emcee_samples, progress=True)
        end_time = default_timer()
        
        logger.info(f"MCMC sampling took {end_time - start_time:.1f} seconds")
        
        # Process MCMC results
        self._process_mcmc_results(emcee_sampler, tempering)
        
        # Increment iteration counter
        self.iterations += 1

    def _get_current_tempering(self) -> float:
        """Get tempering factor for current iteration."""
        if self.iterations < self.max_iterations - 1:
            tempering = self.tempering[self.iterations]
            logger.info(f"Running MCMC with tempering {tempering} (iteration {self.iterations + 1})")
        else:
            tempering = 1.0
            logger.info(f"Running final MCMC without tempering (iteration {self.iterations + 1})")
        
        return tempering

    def _process_mcmc_results(self, emcee_sampler: Any, tempering: float) -> None:
        """Process MCMC results and update output chains."""
        # Calculate burn-in
        if self.emcee_burn < 1:
            burn = int(self.emcee_burn * self.emcee_samples)
        else:
            burn = int(self.emcee_burn)
        
        # Extract chains
        self.unit_chain = emcee_sampler.get_chain(discard=burn, thin=self.emcee_thin, flat=True)
        logp = emcee_sampler.get_log_prob(discard=burn, thin=self.emcee_thin, flat=True)
        self.blobs = emcee_sampler.get_blobs(discard=burn, thin=self.emcee_thin, flat=True)
        
        # Transform to physical parameters
        self.chain = np.array([
            self.pipeline.denormalize_vector_from_prior(p) for p in self.unit_chain
        ])
        
        # Handle output file management
        if self.save_outputs == SAVE_ALL and 0 < self.iterations < self.max_iterations:
            suffix = f'_tempering_{self.tempering[self.iterations-1]}_iteration_{self.iterations}'
            self.output.save_and_reset_to_chain_start(suffix)
        else:
            self.output.reset_to_chain_start()
        
        # Output chain points
        for params, tempered_post, extra in zip(self.chain, logp, self.blobs):
            prior, extra = extra
            post = tempered_post / tempering
            self.output.parameters(params, extra, prior, tempered_post, post)
        
        logger.info(f"Generated {len(self.chain)} chain samples")

    def is_converged(self) -> bool:
        """Check if sampler has completed all iterations.
        
        Returns:
            True if all iterations are complete
        """
        return self.iterations >= self.max_iterations


# Legacy function for compatibility
def extract_theory_points_with_emu(self) -> np.ndarray:
    """Legacy function for emulator theory point extraction.
    
    This function is kept for backward compatibility but should not be used
    in new code. Use the EmulatorModule class instead.
    
    Returns:
        Zero array (placeholder implementation)
    """
    logger.warning("extract_theory_points_with_emu is deprecated - use EmulatorModule instead")
    return np.zeros(self.data_x.size)
