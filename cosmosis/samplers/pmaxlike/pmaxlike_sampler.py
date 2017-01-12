from .. import ParallelSampler
import numpy as np



neval = 0
nfail = 0
def minus_log_posterior(p_in):
    global neval, nfail
    if maxlike_sampler.pool:
        rank = maxlike_sampler.pool.rank
    else:
        rank = 0
    if (not np.all(p_in>=0)) or (not np.all(p_in<=1.0)):
        nfail += 1
        pstr = '   '.join(str(x) for x in p_in)
        msg1 = "[Proc {}] Posterior = NaN for out-of-bounds: (normalized) point = {}".format(rank, pstr)
        msg2 = "[Proc {}] fails= {}".format(rank, nfail)
        maxlike_sampler.output.log_error(msg1)
        maxlike_sampler.output.log_error(msg2)
        return np.inf
    neval +=1 
    p = maxlike_sampler.pipeline.denormalize_vector(p_in)
    post, extra = maxlike_sampler.pipeline.posterior(p)
    pstr = '   '.join(str(x) for x in p)
    msg = "[Proc {} (evals={})] Posterior = {} for {}".format(rank, neval, post, pstr)
    maxlike_sampler.output.log_warning(msg)
    return -post



def posterior_and_gradient(p_in):
    pstr = '   '.join(str(x) for x in p_in)
    msg = "Calculating gradient about (normalized) point {}".format(pstr)
    maxlike_sampler.output.log_warning(msg)
    points = [p_in]
    n = len(p_in)
    for i in xrange(n):
        p = p_in.copy()
        p[i] += maxlike_sampler.epsilon
        points.append(p)

    if maxlike_sampler.pool:
        results = maxlike_sampler.pool.map(minus_log_posterior, points)
    else:
        results = map(minus_log_posterior, points)

    post=results[0]
    grad=np.array([(results[i+1]-post)/maxlike_sampler.epsilon  for i in xrange(n)])
    return post, grad




class PMaxlikeSampler(ParallelSampler):
    sampler_outputs = [("like", float)]

    def config(self):
        self.tolerance = self.read_ini("tolerance", float, 1e-3)
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.output_ini = self.read_ini("output_ini", str, "")
        self.output_cov = self.read_ini("output_covmat", str, "")
        self.epsilon = self.read_ini("gradient_epsilon",float,1e-9)
        self.converged = False
        global maxlike_sampler
        maxlike_sampler = self

    def execute(self):
        import scipy.optimize


        #starting position in the normalized space
        start_vector = self.pipeline.normalize_vector(self.pipeline.start_vector())
        bounds = [(0.0, 1.0) for p in self.pipeline.varied_params]


        result = scipy.optimize.minimize(posterior_and_gradient, start_vector, method='CG',
          jac=True, tol=self.tolerance,  #bounds=bounds, 
          options={'maxiter':self.maxiter, 'disp':True})

        opt_norm = result.x
        opt = self.pipeline.denormalize_vector(opt_norm)
        
        like, extra = self.pipeline.likelihood(opt)

        #Some output - first log the parameters to the screen.
        #It's not really a warning - that's just a level name
        self.output.log_warning("Best fit:\n%s"%'   '.join(str(x) for x in opt))
        self.output.log_warning("Best likelihood: %f\n", like)

        #Next save them to the proper table file
        self.output.parameters(opt, extra, like)

        #If requested, create a new ini file for the
        #best fit.
        if self.output_ini:
          self.pipeline.create_ini(opt, self.output_ini)

        #Also if requested, approximate the covariance matrix with the 
        #inverse of the Hessian matrix.
        #For a gaussian likelihood this is exact.
        if self.output_cov:
            if hasattr(result, 'hess_inv'):
                covmat = self.pipeline.denormalize_matrix(result.hess_inv)
                np.savetxt(self.output_cov, covmat)
            elif hasattr(result, 'hess'):
                covmat = self.pipeline.denormalize_matrix(np.linalg.inv(result.hess_inv))
                np.savetxt(self.output_cov, covmat)
            else:
                self.output.log_error("Sorry - the optimization method you chose does not return a covariance (or Hessian) matrix")

        self.converged = True

    def is_converged(self):
        return self.converged
