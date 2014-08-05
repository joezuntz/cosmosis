from .. import Sampler
import numpy as np
import ctypes as ct
import os

MINUIT_INI_SECTION = "minuit"

pipeline=None
libminuit=None

loglike_type = ct.CFUNCTYPE(
    ct.c_double, #likelihood
    ct.POINTER(ct.c_double),  #parameter cube
)

libname=os.path.join(os.path.split(__file__)[0],"minuit_wrapper.so")

@loglike_type
def wrapped_likelihood(cube_p):
    ndim = len(pipeline.varied_params)
    cube_vector = np.array([cube_p[i] for i in xrange(ndim)])
    vector = pipeline.denormalize_vector(cube_vector)
    try:
        like, extra = pipeline.likelihood(vector)
    except KeyboardInterrupt:
        raise sys.exit(1)
    print vector, like
    return -like

class MinuitSampler(Sampler):

    def config(self):
        self.converged = False
        global libminuit
        if libminuit is None:
            libminuit = ct.cdll.LoadLibrary(libname)
            self._run = libminuit.cosmosis_minuit2_wrapper
            self._run.restype=ct.c_int
            self._run.argtypes = [ct.c_int, ct.POINTER(ct.c_double), loglike_type]
        self.ndim = len(self.pipeline.varied_params)

    def execute(self):
        global pipeline
        pipeline=self.pipeline
        starts = (ct.c_double * self.ndim)()
        start_vector = self.pipeline.normalize_vector(self.pipeline.start_vector())

        for i in xrange(self.ndim):
            starts[i] = start_vector[i]
        self._run(self.ndim, starts, wrapped_likelihood)

        #starts now filled in with final values
        opt = [starts[i] for i in xrange(self.ndim)]
        opt = self.pipeline.denormalize_vector(opt)

        print "Actual values: "
        print opt

        self.converged = True

    def is_converged(self):
        return self.converged
