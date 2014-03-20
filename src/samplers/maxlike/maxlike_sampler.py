import numpy as np
import scipy.optimize
import scipy.misc


MAXLIKE_INI_SECTION = "maxlike"


def MaxlikeSampler(Sampler):
    
    def config(self):
        self.tolerance = self.ini.getfloat(MAXLIKE_INI_SECTION,"tolerance",0.0)

        self.start_vector = np.array([param.normalize(param.start)
                                 from self.pipeline.varied_params])
        
    def execute(self):
        def likefn(p):
            p = self.pipeline.denormalize_vector(p_in)
            like, extra = pipeline.likelihood(p)
            return -like

        opt_norm = scipy.optimize.fmin(likefn, 
                                       self.start_vector,
                                       xtol=self.tolerance)
        opt = pipeline.denormalize_vector(opt_norm)

        # for now print optimzied values
        sys.stdout.write(opt)
