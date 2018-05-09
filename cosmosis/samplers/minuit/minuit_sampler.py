from __future__ import print_function
from builtins import zip
from builtins import str
from .. import ParallelSampler
import numpy as np
import ctypes as ct
import os
import sys
import collections

MINUIT_INI_SECTION = "minuit"

loglike_type = ct.CFUNCTYPE(
    ct.c_double, #likelihood
    ct.POINTER(ct.c_double),  #parameter cube
)

class MinuitOptionsStruct(ct.Structure):
    _fields_ = [
        ("max_evals", ct.c_int),
        ("strategy", ct.c_int),
        ("algorithm", ct.c_int),
        ("save_cov", ct.c_char_p),
        ("tolerance", ct.c_double),
        ("width_estimate", ct.c_double),
        ("do_master_output", ct.c_int),
    ]

libname=os.path.join(os.path.split(__file__)[0],"minuit_wrapper.so")

class MinuitSampler(ParallelSampler):
    needs_output = False
    needs_parallel_output = False
    libminuit = None
    
    def config(self):
        self.converged = False
        if MinuitSampler.libminuit is None:
            if not os.path.exists(libname):
                raise ValueError("The CosmoSIS minuit2 wrapper was not compiled.  If you installed CosmoSIS manually you need the library minuit2 to use this sampler. See the wiki for more details.")
            MinuitSampler.libminuit = ct.cdll.LoadLibrary(libname)
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.save_dir = self.read_ini("save_dir", str, "")
        self.save_cov = self.read_ini("save_cov", str, "")
        self.output_ini = self.read_ini("output_ini", str, "")
        self.verbose = self.read_ini("verbose", bool, False)
        self.iterations = 0

        #Minuit options
        self.strategy = self.read_ini("strategy", str, "medium").lower()
        self.algorithm = self.read_ini("algorithm", str, "migrad").lower()

        self.width_estimate = self.read_ini("width_estimate", float, 0.05)
        self.tolerance = self.read_ini("tolerance", float, 50.0)
        self.neval = 0
        self.param_vector = self.pipeline.start_vector()  #initial value

        strategy = {
            "fast":0,
            "medium":1,
            "safe":2,
            "0":0,
            "1":1,
            "2":2,
        }.get(self.strategy)
        if strategy is None:
            raise ValueError("Minuit 'strategy' parameter must be one of fast, medium, safe (default=medium) not '%s'"%self.strategy)
        self.strategy = strategy

        algorithm = {
            "migrad":0,
            "fallback":1,
            "combined":1,
            "simplex":2,
            # "scan":3,
        }.get(self.algorithm)
        if algorithm is None:
            raise ValueError("Minuit 'algorithm' parameter must be one of migrad, fallback, simplex (default=migrad) not '%s'"%strategy)
        self.algorithm = algorithm


        self._run = MinuitSampler.libminuit.cosmosis_minuit2_wrapper
        self._run.restype = ct.c_int
        vec = ct.POINTER(ct.c_double)
        self.ndim = len(self.pipeline.varied_params)

        cube_type = ct.c_double*self.ndim

        @loglike_type
        def wrapped_likelihood(cube_p):
            vector = np.frombuffer(cube_type.from_address(ct.addressof(cube_p.contents)))
            try:
                like, extra = self.pipeline.posterior(vector)
            except KeyboardInterrupt:
                sys.exit(1)
            self.iterations += 1
            if self.verbose:
                print(self.iterations, like, "   ",  "    ".join(str(v) for v in vector))
            return -like

        self.wrapped_likelihood = wrapped_likelihood


    def execute(self):
        #Run an iteration of minuit
        param_vector, param_names, like, data, status, made_cov, cov_vector = self.sample()


        #update the current parameters
        self.param_vector = param_vector.copy()
        self.neval += status

        if status == 0:
            print() 
            print("SUCCESS: Minuit has converged!")
            print()
            self.save_results(param_vector, param_names, like, data)
            self.converged = True
            self.distribution_hints.set_peak(param_vector)

            if made_cov:
                cov_matrix = cov_vector.reshape((self.ndim, self.ndim))
                self.distribution_hints.set_cov(cov_matrix)
        elif self.neval > self.maxiter:
            print()
            print("MINUIT has failed to converge properly in the max number of iterations.  Sorry.")
            print("Saving the best fitting parameters of the ones we trid, though beware: these are probably not the best-fit")
            print()
            self.save_results(param_vector, param_names, like, data)
            #we actually just use self.converged to indicate that the 
            #sampler should stop now
            self.converged = True

        else:
            print()
            print("Minuit did not converge this run; trying again")
            print("until we run out of iterations.")
            print()
            


    def save_results(self, param_vector, param_names, like, data):
        section = None

        if self.pool is not None:
            print()
            print("Note that the # of function calls printed above is not the total count for all cores, just for one core.")
            print()

        if self.output_ini:
          self.pipeline.create_ini(param_vector, self.output_ini)


        for name, value in zip(param_names, param_vector):
            sec,name=name.split('--')
            if section!=sec:
                print()
                print("[%s]" % sec)
                section=sec
            print("%s = %g" % (name,value))
        print()
        print("Likelihood = ", like)

        if self.save_dir:
            print("Saving best-fit model cosmology to ", self.save_dir)
            data.save_to_directory(self.save_dir, clobber=True)

        self.converged = True

    def sample(self):

        param_names = [str(p) for p in self.pipeline.varied_params]
        param_names_array = (ct.c_char_p * len(param_names))()
        param_names_array[:] = [p.encode('ascii') for p in param_names]

        master = 1 if self.pool is None or self.pool.is_master() else 0
        
        param_max = self.pipeline.max_vector()
        param_min = self.pipeline.min_vector()

        param_vector = self.param_vector.copy()

        cov_vector = np.zeros(self.ndim*self.ndim)
        made_cov = ct.c_int()

        options = MinuitOptionsStruct(
            #allow for more loops
            max_evals=self.maxiter-self.neval, strategy=self.strategy, 
            algorithm=self.algorithm, save_cov=self.save_cov.encode('ascii'),
            tolerance=self.tolerance, width_estimate=self.width_estimate,
            do_master_output = master
            )

        status = self._run(self.ndim, 
            param_vector.ctypes.data_as(ct.POINTER(ct.c_double)), 
            param_min.ctypes.data_as(ct.POINTER(ct.c_double)), 
            param_max.ctypes.data_as(ct.POINTER(ct.c_double)), 
            self.wrapped_likelihood, 
            param_names_array,
            cov_vector.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.byref(made_cov),
            options
            )

        #Run the pipeline one last time ourselves, so we can save the 
        #likelihood and cosmology
        like, _, data = self.pipeline.likelihood(param_vector, return_data=True)
        return param_vector, param_names, like, data, status, made_cov, cov_vector

    def worker(self):
        self.sample()


    def is_converged(self):
        return self.converged
