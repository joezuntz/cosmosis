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
        fisher = self.read_ini("fisher", str, "")
        self.width_estimate = self.read_ini("width_estimate", float, 0.05)
        self.tolerance = self.read_ini("tolerance", float, 0.01)
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

        self.do_fisher = int(bool(fisher))
        self.fisher_file = fisher

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
                print self.iterations, like, "   ",  "    ".join(str(v) for v in vector)
            return -like

        self.wrapped_likelihood = wrapped_likelihood


    def execute(self):
        param_vector, param_names, like, data, status = self.sample()

        #update the current parameters
        self.param_vector = param_vector.copy()


        if status == 0:
            print 
            print "Minuit suceeeded this iteration so we will stop now."
            self.save_results(status, param_vector, param_names, like, data)
            self.converged = True
        else:
            self.neval += status


        if (self.neval > self.maxiter) and (status>0):
            print
            print "Reached max number of evalations Stopping now."
            print "Never converged but saving the best we did."
            print
            self.save_results(status, param_vector, param_names, like, data)
        else:
            print
            print "Minuit did not converge this iteration; running again"
            print "until we run out of iterations."
            print


    def save_results(self, status, param_vector, param_names, like, data):
        section = None

        if self.pool is not None:
            print
            print "Note that the # of function calls printed above is not the total count for all cores, just for one core."
            print

        if self.output_ini:
          self.pipeline.create_ini(param_vector, self.output_ini)


        for name, value in zip(param_names, param_vector):
            sec,name=name.split('--')
            if section!=sec:
                print
                print "[%s]" % sec
                section=sec
            print "%s = %g" % (name,value)
        print
        print "Likelihood = ", like
        if status:
            print "MINUIT reports that it failed to converge properly.  Sorry."
            print "status = ", status
            print

        if self.save_dir:
            print "Saving best-fit model cosmology to ", self.save_dir
            data.save_to_directory(self.save_dir, clobber=True)

        self.converged = True

    def sample(self):

        param_names = [str(p) for p in self.pipeline.varied_params]
        param_names_array = (ct.c_char_p * len(param_names))()
        param_names_array[:] = param_names

        master = 1 if self.pool is None or self.pool.is_master() else 0
        
        param_max = self.pipeline.max_vector()
        param_min = self.pipeline.min_vector()

        param_vector = self.param_vector.copy()

        options = MinuitOptionsStruct(
            #allow for more loops
            max_evals=self.maxiter-self.neval, strategy=self.strategy, 
            algorithm=self.algorithm, save_cov=self.save_cov, 
            tolerance=self.tolerance, width_estimate=self.width_estimate,
            do_master_output = master
            )

        status = self._run(self.ndim, 
            param_vector.ctypes.data_as(ct.POINTER(ct.c_double)), 
            param_min.ctypes.data_as(ct.POINTER(ct.c_double)), 
            param_max.ctypes.data_as(ct.POINTER(ct.c_double)), 
            self.wrapped_likelihood, 
            param_names_array,
            options
            )

        #Run the pipeline one last time ourselves, so we can save the 
        #likelihood and cosmology
        like, _, data = self.pipeline.likelihood(param_vector, return_data=True)
        return param_vector, param_names, like, data, status

    def worker(self):
        self.sample()


    def is_converged(self):
        return self.converged
