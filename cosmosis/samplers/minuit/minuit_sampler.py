from .. import Sampler
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

libname=os.path.join(os.path.split(__file__)[0],"minuit_wrapper.so")

class MinuitSampler(Sampler):
    libminuit = None
    
    def config(self):
        self.converged = False
        if MinuitSampler.libminuit is None:
            MinuitSampler.libminuit = ct.cdll.LoadLibrary(libname)
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.save_dir = self.read_ini("save_dir", str, "")
        self.verbose = self.read_ini("verbose", bool, False)
        self._run = MinuitSampler.libminuit.cosmosis_minuit2_wrapper
        self._run.restype = ct.c_int
        self._run.argtypes = [ct.c_int, ct.POINTER(ct.c_double), loglike_type, ct.c_uint]
        self.ndim = len(self.pipeline.varied_params)

    def execute(self):
        cube_type = ct.c_double*self.ndim

        @loglike_type
        def wrapped_likelihood(cube_p):
            cube_vector = np.frombuffer(cube_type.from_address(ct.addressof(cube_p.contents)))
            vector = self.pipeline.denormalize_vector(cube_vector)
            try:
                like, extra = self.pipeline.likelihood(vector)
            except KeyboardInterrupt:
                sys.exit(1)
            if self.verbose:
                print like, "    ".join(str(v) for v in vector)
            return -like

        param_names = [str(p) for p in self.pipeline.varied_params]
        param_names_array = (ct.c_char_p * len(param_names))()
        param_names_array[:] = param_names


        param_vector = self.pipeline.normalize_vector(self.pipeline.start_vector())
        status = self._run(self.ndim, 
            param_vector.ctypes.data_as(ct.POINTER(ct.c_double)), 
            wrapped_likelihood, 
            self.maxiter,
            param_names_array
            )

        #Run the pipeline one last time ourselves, so we can save the 
        #likelihood and cosmology
        max_like = self.pipeline.denormalize_vector(param_vector)
        like, _, data = self.pipeline.likelihood(max_like, return_data=True)

        print "The values above (which are printed from within MINUIT) refer to normalized parameters"
        print "(in the range 0-1).  The actual parameter values are:"
        section = None
        for name, value in zip(param_names, max_like):
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
            print "Saving best-fit results to ", self.save_dir
            data.save_to_directory(self.save_dir, clobber=True)

        self.converged = True

    def is_converged(self):
        return self.converged
