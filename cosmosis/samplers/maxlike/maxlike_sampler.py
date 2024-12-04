from .. import Sampler
from ...runtime import logs
import numpy as np
import scipy.optimize


class MaxlikeSampler(Sampler):
    sampler_outputs = [("prior", float), ("like", float), ("post", float)]

    def config(self):
        self.tolerance = self.read_ini("tolerance", float, 1e-3)
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.output_ini = self.read_ini("output_ini", str, "")
        self.output_cov = self.read_ini("output_covmat", str, "")
        self.method = self.read_ini("method",str,"Nelder-Mead")
        self.max_posterior = self.read_ini("max_posterior", bool, False)
        self.repeats = self.read_ini("repeats", int, 1)
        start_method = self.read_ini("start_method", str, "")

        if self.repeats > 1:
            start_method_is_random = start_method in ["prior", "cov", "chain"]
            have_previous_peak = self.distribution_hints.has_peak()
            have_previous_cov = self.distribution_hints.has_cov()

            if have_previous_peak and not have_previous_cov:
                raise ValueError("You have set repeats>1 in maxlike, and have chained multiple samplers together. "
                                 "But the previous sampler(s) did not save a covariance matrix. "
                                 "So we have no way to generate multiple starting points. "
                                 )
            elif not start_method_is_random:
                raise ValueError(f"You set the repeats parameter in the max-like sampler to {self.repeats}"
                                 " but to you this you must also set a start method to choose a new starting point"
                                 " each repeat.  You can set start_method to 'prior' to sample from the prior,"
                                 " or set it to 'cov' or 'chain' and also set start_input to a file to load one"
                                 " either a covariance matrix or a chain of samples."
                                 )

        if self.max_posterior:
            logs.overview("------------------------------------------------")
            logs.overview("NOTE: Running optimizer in **max-posterior** mode:")
            logs.overview("NOTE: Will maximize the combined likelihood and prior")
            logs.overview("------------------------------------------------")
        else:
            logs.overview("--------------------------------------------------")
            logs.overview("NOTE: Running optimizer in **max-like** mode:")
            logs.overview("NOTE: not including the prior, just the likelihood.")
            logs.overview("NOTE: Set the parameter max_posterior=T to change this.")
            logs.overview("NOTE: This won't matter unless you set some non-flat")
            logs.overview("NOTE: priors in a separate priors file.")
            logs.overview("--------------------------------------------------")

        self.repeat_index = 0
        self.best_fit_results = []

    def save_final_outputs(self):

        # This can happen if the user ctrl'c's the run before any results are saved
        if not self.best_fit_results:
            return

        # Sort the repeated best-fit estimates by increasing posterior or likelihood
        if self.max_posterior:
            self.best_fit_results.sort(key=lambda x: x.post)
        else:
            self.best_fit_results.sort(key=lambda x: x.like)

        # Save all the results to the main chain output file
        for results in self.best_fit_results:
            self.output.parameters(results.vector, results.extra, results.prior, results.like, results.post)


        # Find the overall best fit. We use this to create a new ini file and/or covmat, if requested,
        # and to pass on to the next sampler in any chain
        results = self.best_fit_results[-1]

        # If requested, create a new ini file for the best fit.
        if self.output_ini:
          self.pipeline.create_ini(results.vector, self.output_ini)

        # This info is useful for the next sampler in a chain
        self.distribution_hints.set_peak(results.vector, results.post)

        # Also if requested, approximate the covariance matrix with the
        # inverse of the Hessian matrix.
        # For a gaussian posterior this is exact.
        if results.covmat is None:
            if self.output_cov:
               logs.error("Sorry - the optimization method you chose does not return a covariance (or Hessian) matrix")
        else:
            if self.output_cov:
                np.savetxt(self.output_cov, results.covmat)

            # This info is also useful for the next sampler in a chain
            self.distribution_hints.set_cov(results.covmat)


    def run_optimizer(self, likefn, start_vector):
        bounds = [(0.0, 1.0) for p in self.pipeline.varied_params]

        if self.method.lower() == "bobyqa":
            # use the specific pybobyqa minimizer
            import pybobyqa

            # this works with bounds in the form of a tuple of two arrays
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            kw = {
                "seek_global_minimum": True,
                "bounds": (lower,upper),
                "print_progress": logs.is_enabled_for(logs.NOISY),
                "rhobeg": 0.1,
                "rhoend": self.tolerance,
            }
            optimizer_result = pybobyqa.solve(likefn, start_vector, **kw)
            opt_norm = optimizer_result.x
        else:
            # Use scipy mainimizer instead
            optimizer_result = scipy.optimize.minimize(likefn, start_vector, method=self.method,
            jac=False, tol=self.tolerance,
            options={'maxiter':self.maxiter, 'disp':True})

            opt_norm = optimizer_result.x

        opt = self.pipeline.denormalize_vector(opt_norm)
        
        # Generate a results object containing the best-fit parameters,
        # their likelihood, posterior, prior, and derived parameters
        results = self.pipeline.run_results(opt)

        # Log some info to the screen
        logs.overview("\n\nOptimization run {} of {}".format(self.repeat_index+1, self.repeats))
        if self.max_posterior:
            logs.overview("Best fit (by posterior):\n%s"%'   '.join(str(x) for x in opt))
        else:
            logs.overview("Best fit (by likelihood):\n%s"%'   '.join(str(x) for x in opt))
        logs.overview("Posterior: {}".format(results.post))
        logs.overview("Likelihood: {}\n".format(results.like))

        # Also attach any covariance matrix saved by the sampler to the results
        # object, in an ad-hoc way
        if hasattr(optimizer_result, 'hess_inv'):
            if self.method == "L-BFGS-B":
                results.covmat = self.pipeline.denormalize_matrix(optimizer_result.hess_inv.todense())
            else:
                results.covmat = self.pipeline.denormalize_matrix(optimizer_result.hess_inv)
        elif hasattr(optimizer_result, 'hess'):
            results.covmat = self.pipeline.denormalize_matrix(np.linalg.inv(optimizer_result.hess))
        else:
            results.covmat = None

        return results

    def execute(self):

        # The likelihood function we wish to optimize.
        # Not that we are minimizing the negative of the likelihood/posterior
        def likefn(p_in):
            # Check the normalization
            if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
                return np.inf
            p = self.pipeline.denormalize_vector(p_in)
            r = self.pipeline.run_results(p)
            if self.max_posterior:
                return -r.post
            else:
                return -r.like

        # Choose a starting point. If we are repeating our optimization we will need
        # multiple starting points. Otherwise we use a fixed one
        repeating = self.repeats > 1

        if repeating:
            # If we are taking a random starting point then there is a chance it will randomly
            # be invalid. So we should try a 20 times to get a valid one.
            for _ in range(20):
                start_vector_denorm = self.start_estimate(prefer_random=True)
                start_vector = self.pipeline.normalize_vector(start_vector_denorm)
                if np.isfinite(likefn(start_vector)):
                    break
            else:
                raise RuntimeError("Could not find a valid random starting point for maxlike in 20 tries")
        else:
            # Otherwise we just use the one starting point. It will be from a previous
            # chained sampler, if there is one, or otherwise just what is in the values file
            start_vector_denorm = self.start_estimate()
            start_vector = self.pipeline.normalize_vector(start_vector_denorm)

            # If that is invalid we will raise an error
            if not np.isfinite(likefn(start_vector)):
                raise RuntimeError("Invalid starting point for maxlike")

        logs.overview(f"Starting from point: {start_vector_denorm}")

        try:
            results = self.run_optimizer(likefn, start_vector)
        except KeyboardInterrupt:
            # If we get a ctrl-c we should save the current best fit
            # and then exit.
            # Otherwise report what we are doing. It should be basically instant
            # so it shouldn't annoy anyone
            logs.overview("Keyboard interrupt - saving current best fit, if any finished")
            self.save_final_outputs()
            raise

        # We will only save all the results at the end, because then we can sort them
        # by likelihood or posterior. So for now just save this one.
        self.best_fit_results.append(results)

        # In the final iteration we save everything. # We probably want to do this
        # on a keyboard interrupt too.
        self.repeat_index += 1

        if self.repeat_index == self.repeats:
            self.save_final_outputs()

    def is_converged(self):
        return self.repeat_index >= self.repeats
