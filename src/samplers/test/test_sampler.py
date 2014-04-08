from sampler import Sampler
import numpy as np
import os

RUNTIME_INI_SECTION = "runtime"

class TestSampler(Sampler):
    def config(self):
        self.converged = False
        self.fatal_errors = self.ini.getboolean(RUNTIME_INI_SECTION, "fatal-errors", default=False)

    def execute(self):
        # load initial parameter values
        p = np.array([param.start for param in self.pipeline.varied_params])
        try:
            prior  = self.pipeline.prior(p)
            like, extra, data = self.pipeline.likelihood(p, return_data=True)

            print "Prior      = ", prior
            print "Likelihood = ", like
            print "Posterior  = ", like+prior
        except Exception as e:
            if self.fatal_errors: raise
            print "(Could not get a likelihood) Error:"+str(e)
        try:
            if self.ini.has_option(RUNTIME_INI_SECTION, "save_dir"):
                output_data_dir = self.ini.get(RUNTIME_INI_SECTION, "save_dir")
                if output_data_dir:
                    data.save_to_directory(output_data_dir, clobber=True)
        except Exception as e:
            if self.fatal_errors: raise
            print "Could not save output."
        self.converged = True

    def is_converged(self):
        return self.converged
