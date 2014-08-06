from .. import Sampler
import numpy as np

TEST_INI_SECTION = "test"


class TestSampler(Sampler):
    needs_output = False

    def config(self):
        self.converged = False
        self.fatal_errors = self.ini.getboolean(TEST_INI_SECTION,
                                                "fatal_errors",
                                                default=False)
        #for backward compatibility we add a version with the hyphen
        self.fatal_errors = self.fatal_errors or (
                            self.ini.getboolean(TEST_INI_SECTION,
                                                "fatal-errors",
                                                default=False)
            )
        self.save_dir = self.ini.get(TEST_INI_SECTION,
                                     "save_dir",
                                     default=False)


    def execute(self):
        # load initial parameter values
        p = np.array([param.start for param in self.pipeline.varied_params])
    
        # try to print likelihood if it exists
        data=None
        try:
            prior = self.pipeline.prior(p)
            like, extra, data = self.pipeline.likelihood(p, return_data=True)
            if self.pipeline.likelihood_names:
                print "Prior      = ", prior
                print "Likelihood = ", like
                print "Posterior  = ", like+prior
        except Exception as e:
            if self.fatal_errors:
                raise
            print "(Could not get a likelihood) Error:"+str(e)
        if not self.pipeline.likelihood_names:
            print
            print "There was no likelihood as you did not ask for any"
            print "Fill in the parameter 'likelihoods ='"
            print "In the ini file [pipeline] section if you want some"
            print
        elif (like==-np.inf) and (data is not None):
            found_likelihoods = [k[1] for k in data.keys() if k[0]=="likelihoods"]
            print
            print "The log-likelihood was -infinity!"
            print "This means one of three things:"
            print "1)  A likelihood code returned -inf because it is broken"
            print "2)  The parameters you chose are really really bad"
            print "3)  You made a typo filling in the likelihoods in the ini file"
            print "In case the answer is 3, you asked for these likelihoods:"
            print "  ",  ", ".join([k+"_like" for k in self.pipeline.likelihood_names])
            print "And the pipeline calculated these:"
            print "  ",  ", ".join(found_likelihoods)
            print

        try:
            if self.save_dir:
                if data is not None:
                    data.save_to_directory(self.save_dir, clobber=True)
                else:
                    print "(There was an error so no output to save)"
        except Exception as e:
            if self.fatal_errors:
                raise
            print "Could not save output."
        self.converged = True

    def is_converged(self):
        return self.converged
