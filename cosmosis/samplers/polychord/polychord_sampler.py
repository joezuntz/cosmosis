#coding: utf-8
from __future__ import print_function
from builtins import str
from builtins import range
from .. import ParallelSampler
import ctypes as ct
import os
import cosmosis
import numpy as np
import sys

prior_type = ct.CFUNCTYPE(None, 
    ct.POINTER(ct.c_double),  #hypercube
    ct.POINTER(ct.c_double),  #physical
    ct.c_int,   #ndim
)

loglike_type = ct.CFUNCTYPE(ct.c_double, 
    ct.POINTER(ct.c_double),  #physical
    ct.c_int,   #ndim
    ct.POINTER(ct.c_double),  #derived
    ct.c_int,   #nderived
)


dumper_type = ct.CFUNCTYPE(None, #void
    ct.c_int,  #ndead
    ct.c_int,  #nlive
    ct.c_int,  #npars
    ct.POINTER(ct.c_double),   #live
    ct.POINTER(ct.c_double),   #dead
    ct.POINTER(ct.c_double),   #logweights
    ct.c_double,   #logZ
    ct.c_double,   #logZerr
)


polychord_args = [
    loglike_type,           #loglike,
    prior_type,             #prior,
    dumper_type,            #dumper,
    ct.c_int,               #nlive
    ct.c_int,               #nrepeats
    ct.c_int,               #nprior
    ct.c_bool,              #do_clustering
    ct.c_int,               #feedback
    ct.c_double,            #precision_criterion
    ct.c_double,            #logzero
    ct.c_int,               #max_ndead
    ct.c_double,            #boost_posterior
    ct.c_bool,              #posteriors
    ct.c_bool,              #equals
    ct.c_bool,              #cluster_posteriors
    ct.c_bool,              #write_resume 
    ct.c_bool,              #write_paramnames
    ct.c_bool,              #read_resume
    ct.c_bool,              #write_stats
    ct.c_bool,              #write_live
    ct.c_bool,              #write_dead
    ct.c_bool,              #write_prior
    ct.c_double,            #compression_factor
    ct.c_int,               #nDims
    ct.c_int,               #nDerived 
    ct.c_char_p,            #base_dir
    ct.c_char_p,            #file_root
    ct.c_int,               #nGrade
    ct.POINTER(ct.c_double),#grade_frac
    ct.POINTER(ct.c_int),   #grade_dims
    ct.c_int,               #n_nlives
    ct.POINTER(ct.c_double),#loglikes
    ct.POINTER(ct.c_int),   #nlives
    ct.c_int,               #seed
]


POLYCHORD_SECTION='polychord'


class PolyChordSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("post", float), ("weight", float)]
    supports_smp=False
    understands_fast_subspaces = True

    def config(self):
        if self.pool:
            libname = "libchord_mpi.so"
        else:
            libname = "libchord.so"

        dirname = os.path.split(__file__)[0]
        libname = os.path.join(dirname, "polychord_src", libname)
            
        try:
            libchord = ct.cdll.LoadLibrary(libname)
        except Exception as error:
            sys.stderr.write("PolyChord could not be loaded.\n")
            sys.stderr.write("This may mean an MPI compiler was not found to compile it,\n")
            sys.stderr.write("or that some other error occurred.  More info below.\n")
            sys.stderr.write(str(error)+'\n')
            sys.exit(1)

        self._run = libchord.polychord_c_interface
        self._run.restype=None
        self._run.argtypes = polychord_args
        self.converged=False

        self.ndim = len(self.pipeline.varied_params)
        self.nderived = len(self.pipeline.extra_saves)

        #Required options
        self.live_points    = self.read_ini("live_points", int, 100)

        #Output and feedback options
        self.feedback               = self.read_ini("feedback", int, 1)
        self.resume                 = self.read_ini("resume", bool, False)
        self.polychord_outfile_root = self.read_ini("polychord_outfile_root", str, "")
        self.compression_factor     = self.read_ini("compression_factor", float, np.exp(-1))

        #General run options
        self.max_iterations = self.read_ini("max_iterations", int, -1)
        self.num_repeats = self.read_ini("num_repeats", int, 0)
        self.nprior = self.read_ini("nprior", int, self.live_points*10)
        self.random_seed = self.read_ini("random_seed", int, -1)
        self.tolerance   = self.read_ini("tolerance", float, 0.1)
        self.log_zero    = self.read_ini("log_zero", float, -1e6)

        self.fast_fraction    = self.read_ini("fast_fraction", float, 0.5)

        if self.output:
            def dumper(ndead, nlive, npars, live, dead, logweights, log_z, log_z_err):
                print("Saving %d samples" % ndead)
                self.output_params(ndead, nlive, npars, live, dead, logweights, log_z, log_z_err)
            self.wrapped_output_logger = dumper_type(dumper)
        else:
            def dumper(ndead, nlive, npars, live, dead, logweights, log_z, log_z_err):
                pass
            self.wrapped_output_logger = dumper_type(dumper)

        def prior(cube, theta, ndim):
            cube_vector = np.array([cube[i] for i in range(ndim)])
            if self.pipeline.do_fast_slow:
                cube_vector = self.reorder_slow_fast(cube_vector)
            try:
                theta_vector = self.pipeline.denormalize_vector_from_prior(cube_vector) 
            except ValueError:
                # Polychord sometimes seems to propose outside the prior.
                # Just give terrible parameters when that happens.
                theta_vector = np.repeat(-np.inf, ndim)
            for i in range(ndim):
                theta[i] = theta_vector[i]

        self.wrapped_prior = prior_type(prior)

        def likelihood(theta, ndim, phi, nderived):
            theta_vector = np.array([theta[i] for i in range(ndim)])
            if np.any(~np.isfinite(theta_vector)):
                return -np.inf

            try:
                like, phi_vector = self.pipeline.likelihood(theta_vector)
                for i in range(nderived):
                    phi[i] = phi_vector[i]
            except KeyboardInterrupt:
                raise sys.exit(1)

            return like
        self.wrapped_likelihood = loglike_type(likelihood)

    def reorder_slow_fast(self, x):
        y = np.zeros_like(x)
        ns = self.pipeline.n_slow_params
        y[self.pipeline.slow_param_indices] = x[0:ns]
        y[self.pipeline.fast_param_indices] = x[ns:]
        return y



    def worker(self):
        self.sample()

    def execute(self):
        self.log_z = 0.0
        self.log_z_err = 0.0

        self.sample()

        self.output.final("log_z", self.log_z)
        self.output.final("log_z_error", self.log_z_err)

    def sample(self):

        n_grade = 2 if self.pipeline.do_fast_slow else 1        
        grade_dims = (ct.c_int*n_grade)()
        grade_frac = (ct.c_double*n_grade)()

        if self.pipeline.do_fast_slow:
            grade_dims[0] = self.pipeline.n_slow_params
            grade_dims[1] = self.pipeline.n_fast_params
            grade_frac[0] = 1 - self.fast_fraction
            grade_frac[1] = self.fast_fraction
            print("Telling Polychord to spend fraction {} if its time in the fast subspace (adjust with fast_fraction option)".format(self.fast_fraction))
        else:
            grade_dims[0] = self.pipeline.nvaried
            grade_frac[0] = 1.0


        n_nlives = 0
        loglikes = (ct.c_double*n_nlives)()
        nlives = (ct.c_int*n_nlives)()

        polychord_outfile_root = self.polychord_outfile_root.encode('ascii')

        if self.num_repeats == 0:
            num_repeats = 3 * grade_dims[0]
            print("Polychord num_repeats = {}  (3 * n_slow_params [{}])".format(num_repeats, grade_dims[0]))
        else:
            num_repeats = self.num_repeats
            print("Polychord num_repeats = {}  (from parameter file)".format(num_repeats))

        self._run(
                self.wrapped_likelihood,      #loglike,
                self.wrapped_prior,           #prior,
                self.wrapped_output_logger,   #dumper,
                self.live_points,             #nlive
                num_repeats,                  #nrepeats
                self.nprior,                  #nprior
                True,                         #do_clustering
                self.feedback,                #feedback
                self.tolerance,               #precision_criterion
                self.log_zero,                #logzero
                self.max_iterations,          #max_ndead
                0.,                           #boost_posterior
                polychord_outfile_root,  #posteriors
                polychord_outfile_root,  #equals
                ct.c_bool,                    #cluster_posteriors
                self.resume,                  #write_resume 
                False,                        #write_paramnames
                self.resume,                  #read_resume
                polychord_outfile_root,  #write_stats
                polychord_outfile_root,  #write_live
                polychord_outfile_root,  #write_dead
                polychord_outfile_root,  #write_prior
                self.compression_factor,      #compression_factor
                self.ndim,                    #nDims
                self.nderived,                #nDerived 
                "".encode('ascii'),           #base_dir
                polychord_outfile_root,  #file_root
                n_grade,                      #nGrade
                grade_frac,                   #grade_frac
                grade_dims,                   #grade_dims
                n_nlives,                     #n_nlives
                loglikes,                     #loglikes
                nlives,                       #nlives
                self.random_seed,             #seed
                )

        self.converged = True

    def output_params(self, ndead, nlive, npars, live, dead, logweights, log_z, log_z_err):
        self.log_z = log_z
        self.log_z_err = log_z_err
        data = np.array([dead[i] for i in range(npars*ndead)]).reshape((ndead, npars))
        logw = np.array([logweights[i] for i in range(ndead)])
        for row, w in zip(data,logw):
            params = row[:self.ndim]
            extra_vals = row[self.ndim:self.ndim+self.nderived]
            birth_like = row[self.ndim+self.nderived]
            like = row[self.ndim+self.nderived+1]
            importance = np.exp(w)
            self.output.parameters(params, extra_vals, like, importance)
        self.output.final("nsample", ndead)
        self.output.flush()

    def is_converged(self):
        return self.converged
