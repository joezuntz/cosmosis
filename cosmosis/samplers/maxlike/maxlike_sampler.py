from builtins import str
from .. import Sampler
import numpy as np


class MaxlikeSampler(Sampler):
    sampler_outputs = [("like", float)]

    def config(self):
        self.tolerance = self.read_ini("tolerance", float, 1e-3)
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.output_ini = self.read_ini("output_ini", str, "")
        self.output_cov = self.read_ini("output_covmat", str, "")
        self.method = self.read_ini("method",str,"Nelder-Mead")
        self.max_posterior = self.read_ini("max_posterior", bool, False)

        self.converged = False

    def execute(self):
        import scipy.optimize

        def likefn(p_in):
            #Check the normalization
            if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
                return np.inf
            p = self.pipeline.denormalize_vector(p_in)
            if self.max_posterior:
                like, extra = self.pipeline.posterior(p)
                self.output.log_debug("%s  post=%le"%('   '.join(str(x) for x in p),like))
            else:
                like, extra = self.pipeline.likelihood(p)
                self.output.log_debug("%s  like=%le"%('   '.join(str(x) for x in p),like))
            return -like

        #starting position in the normalized space
        start_vector = self.pipeline.normalize_vector(self.pipeline.start_vector())
        bounds = [(0.0, 1.0) for p in self.pipeline.varied_params]


        result = scipy.optimize.minimize(likefn, start_vector, method=self.method, 
          jac=False, tol=self.tolerance,  #bounds=bounds, 
          options={'maxiter':self.maxiter, 'disp':True})

        opt_norm = result.x
        opt = self.pipeline.denormalize_vector(opt_norm)
        

        #Some output - first log the parameters to the screen.
        #It's not really a warning - that's just a level name
        if self.max_posterior:
            like, extra = self.pipeline.posterior(opt)
            self.output.log_warning("Best fit:\n%s"%'   '.join(str(x) for x in opt))
            self.output.log_warning("Posterior: {}\n".format(like))
        else:
            like, extra = self.pipeline.likelihood(opt)
            self.output.log_warning("Best fit:\n%s"%'   '.join(str(x) for x in opt))
            self.output.log_warning("Likelihood: {}\n".format(like))

        #Next save them to the proper table file
        self.output.parameters(opt, extra, like)

        #If requested, create a new ini file for the
        #best fit.
        if self.output_ini:
          self.pipeline.create_ini(opt, self.output_ini)

        self.distribution_hints.set_peak(opt)          

        #Also if requested, approximate the covariance matrix with the 
        #inverse of the Hessian matrix.
        #For a gaussian likelihood this is exact.
        covmat = None
        if hasattr(result, 'hess_inv'):
            if self.method == "L-BFGS-B":
                covmat = self.pipeline.denormalize_matrix(result.hess_inv.todense())
            else:
                covmat = self.pipeline.denormalize_matrix(result.hess_inv)
        elif hasattr(result, 'hess'):
            covmat = self.pipeline.denormalize_matrix(np.linalg.inv(result.hess_inv))

        if covmat is None:
            if self.output_cov:
               self.output.log_error("Sorry - the optimization method you chose does not return a covariance (or Hessian) matrix")
        else:
            if self.output_cov:
                np.savetxt(self.output_cov, covmat)
            self.distribution_hints.set_cov(covmat)

        self.converged = True

    def is_converged(self):
        return self.converged
