from sampler import Sampler
import numpy as np

class TestSampler(Sampler):
    def config(self):
        self.converged = False

    def execute(self):
        # load 
        p = np.array([param.start for param in self.pipeline.varied_params])
        try:
            prior  = self.pipeline.prior(p)
            like, extra = self.pipeline.likelihood(p)
            print "Prior      = ", prior
            print "Likelihood = ", like
            print "Posterior  = ", like+prior
        except Exception as e:
            print "(Could not get a likelihood) Error:"+str(e)
        self.converged = True

    def is_converged(self):
        return self.converged
