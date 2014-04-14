from sampler import Sampler
import numpy as np

TEST_INI_SECTION = "test"


class TestSampler(Sampler):
    needs_output = False

    def config(self):
        self.converged = False
        self.fatal_errors = self.ini.getboolean(TEST_INI_SECTION,
                                                "fatal-errors",
                                                default=False)
        self.save_dir = self.ini.get(TEST_INI_SECTION,
                                           "save_dir",
                                           default=False)


    def execute(self):
        # load initial parameter values
        p = np.array([param.start for param in self.pipeline.varied_params])
        try:
            prior = self.pipeline.prior(p)
            like, extra, data = self.pipeline.likelihood(p, return_data=True)

            print "Prior      = ", prior
            print "Likelihood = ", like
            print "Posterior  = ", like+prior
        except Exception as e:
            if self.fatal_errors:
                raise
            print "(Could not get a likelihood) Error:"+str(e)
        try:
            if self.save_dir:
                data.save_to_directory(self.save_dir, clobber=True)
        except Exception as e:
            if self.fatal_errors:
                raise
            print "Could not save output."
        self.converged = True

    def is_converged(self):
        return self.converged
