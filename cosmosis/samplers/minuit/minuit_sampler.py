from .. import Sampler
import numpy as np
import ctypes as ct
import os

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
        self._run = MinuitSampler.libminuit.cosmosis_minuit2_wrapper
        self._run.restype = ct.c_int
        self._run.argtypes = [ct.c_int, ct.POINTER(ct.c_double), loglike_type]
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
                import sys; sys.exit(1)
            print vector, like
            return -like

        param_vector = self.pipeline.normalize_vector(self.pipeline.start_vector())
        self._run(self.ndim, param_vector.ctypes.data_as(ct.POINTER(ct.c_double)), wrapped_likelihood)

        print "Actual values: "
        print self.pipeline.denormalize_vector(param_vector)

        self.converged = True

    def is_converged(self):
        return self.converged
