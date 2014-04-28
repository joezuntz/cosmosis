from .. import Sampler
import numpy as np


MAXLIKE_INI_SECTION = "maxlike"


class MaxlikeSampler(Sampler):

    def config(self):
        self.tolerance = self.ini.getfloat(MAXLIKE_INI_SECTION,
                                           "tolerance", 1e-3)
        self.maxiter = self.ini.getint(MAXLIKE_INI_SECTION,
                                       "maxiter", 1000)
        self.output_ini = self.ini.get(MAXLIKE_INI_SECTION,
                                       "output_ini", "")

        self.converged = False

    def execute(self):
        import scipy.optimize

        def likefn(p_in):
            #Check the normalization
            if np.any(p_in<0) or np.any(p_in>1):
                return np.inf
            p = self.pipeline.denormalize_vector(p_in)
            like, extra = self.pipeline.likelihood(p)
            self.output.log_debug("%s  like=%le"%('   '.join(str(x) for x in p),like))
            return -like

        #starting position in the normalized space
        start_vector = self.pipeline.normalize_vector(self.pipeline.start_vector())


        opt_norm = scipy.optimize.fmin(likefn,
                                       start_vector,
                                       ftol=self.tolerance,
                                       disp=False,
                                       maxiter=self.maxiter)

        opt = self.pipeline.denormalize_vector(opt_norm)
        
        like, extra = self.pipeline.likelihood(opt)

        #Some output - first log the parameters to the screen.
        #It's not really a warning - that's just a level name
        self.output.log_warning("Best fit:\n%s"%'   '.join(str(x) for x in opt))

        #Next save them to the proper table file
        self.output.parameters(opt, extra)

        #And finally, if requested, create a new ini file for the
        #best fit.
        if self.output_ini:
          self.pipeline.create_ini(opt, self.output_ini)

        self.converged = True

    def is_converged(self):
        return self.converged
