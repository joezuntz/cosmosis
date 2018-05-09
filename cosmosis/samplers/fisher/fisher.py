from __future__ import print_function
from builtins import map
from builtins import range
from builtins import object
import numpy as np
import pdb

'''
NOTE: Need to recompute the spectra between each iteration (EUGH).

Steps:
    - For a given parameter, choose 4 even variations withing 1 sigma contour (i.e. this will probably be an input)
    - Calculate Fisher as dClobs/dparm
        - Calculate derivative with 5-point stencil method http://en.wikipedia.org/wiki/Five-point_stencil
    - Find the new 1 sigma contour according to this Fisher
    - Iterate until contours converge to some tolerance -- 0.5% as default

Inputs:
    - function to calculate vector of Clobs
Outpus:
    - Converged Fisher matrix

Write to parallelise
will be passed a pool with map function can pool.map(params) will give list of results
make list of params, run all at once, will parallelise nicely
be aware that some of the workers in pool might fail, return e.g. None
can have all paramers normalised to [0,1]

compute_vector(p) runs an entire pipeline on a set of parameters (i.e. needs to be done five times here) and returns vector, covmat where vector is Cl_*** and covmat is the necessary **INVERSE** covmat

can do pool.map(compute_vector, [p1,p2 etc]) where p1 is an array of the parameter values

len(start_vector) will give you the number of parameters we are varying, which is taken from the cosmosis values.ini file

'''

class FisherParameterError(Exception):
    def __init__(self, parameter_index):
        message = "Fisher Matrix likelihood function returned None for parameter: {}".format(parameter_index)
        super(Exception,self).__init__(self, message)
        self.parameter_index = parameter_index


class Fisher(object):
    def __init__(self, compute_vector, start_vector, step_size, tolerance, maxiter, pool=None):
        
        self.compute_vector = compute_vector
        self.maxiter = maxiter
        self.step_size = step_size
        self.start_params = start_vector
        self.current_params = start_vector
        self.nparams = start_vector.shape[0]
        self.iterations = 0
        self.pool = pool

    def converged(self):
        crit = (abs(self.new_onesigma - self.old_onesigma).max() < self.threshold)
        return crit

    def converge_fisher_matrix(self):
        
        self.new_Fmatrix = compute_fisher_matrix(self.compute_vector,
            self.start_vector, self.start_covmat)
        
        self.old_onesigma = compute_one_sigma(new_Fmatrix)

        while True:
            self.iterations+=1
            self.old_onesigma = self.new_onesigma
            self.current_params = self.choose_new_params(self.new_Fmatrix)

            self.new_Fmatrix = self.compute_fisher_matrix()

            self.new_onesigma = compute_one_sigma(self.new_Fmatrix)

            if self.converged():
                print('Fisher has converged!')
                return new_Fmatrix

            if self.iterations > self.maxiter:
                print("Run out of iterations.")
                print("Done %d, max allowed %d" % (self.iterations, self.maxiter))
                return None

    def compute_derivatives(self):
        derivatives = []
        points = []

        #To improve parallelization we first gather all the data points
        #we use in all the dimensions
        for p in range(self.nparams):
            points +=  self.five_points_stencil_points(p)
        print("Calculating derivatives using {} total models".format(len(points)))
        if self.pool is None:
            results = list(map(self.compute_vector, points))
        else:
            results = self.pool.map(self.compute_vector, points)

        #Now get out the results that correspond to each dimension
        for p in range(self.nparams):
            results_p = results[4*p:4*(p+1)]
            derivative, inv_cov = self.five_point_stencil_deriv(results_p, p)
            derivatives.append(derivative)
        derivatives = np.array(derivatives)
        return derivatives, inv_cov


    def compute_fisher_matrix(self):
        derivatives, inv_cov = self.compute_derivatives()

        if not np.allclose(inv_cov, inv_cov.T):
            print("WARNING: The inverse covariance matrix produced by your pipeline")
            print("         is not symmetric. This probably indicates a mistake somewhere.")
            print("         If you are only using cosmosis-standard-library likelihoods please ")
            print("         open an issue about this on the cosmosis site.")
        fisher_matrix = np.einsum("il,lk,jk->ij", derivatives, inv_cov, derivatives)
        return fisher_matrix

    def five_points_stencil_points(self, param_index):
        delta = np.zeros(self.nparams)
        delta[param_index] = 1.0
        points = [self.current_params + x*delta for x in 
            [2*self.step_size, 
             1*self.step_size, 
            -1*self.step_size, 
            -2*self.step_size]
        ]
        return points        

    def five_point_stencil_deriv(self, results, param_index):
        for r in results:
            if r is None:
                raise FisherParameterError(param_index)
        obs = [r[0] for r in results]
        inv_cov = results[0][1]
        deriv = (-obs[0] + 8*obs[1] - 8*obs[2] + obs[3])/(12*self.step_size)
        return deriv, inv_cov

    def compute_one_sigma(Fmatrix):
        sigma = np.sqrt(np.linalg.inv(Fmatrix))
        return sigma

class NumDiffToolsFisher(Fisher):
    def compute_derivatives(self):
        import numdifftools as nd
        def wrapper(param_vector):
            print("Running pipeline:", param_vector)
            return self.compute_vector(param_vector)[0]
        jacobian_calculator = nd.Jacobian(wrapper, step=self.step_size)
        derivatives = jacobian_calculator(self.current_params)
        _, inv_cov = self.compute_vector(self.current_params)
        print(derivatives.shape, inv_cov.shape)
        return derivatives.T, inv_cov
    


def test():
    def theory_prediction(x):
        #same number of data points as parameters here
        theory = 2*x + 2
        inv_cov = np.diag(np.ones_like(x)**-1)
        return theory, inv_cov

    best_fit_params = np.array([0.1, 1.0, 2.0, 4.0])
    fisher_calculator = Fisher(theory_prediction, best_fit_params, 0.01, 0.0, 1)
    F = fisher_calculator.compute_fisher_matrix()
    print(F)
    return F

if __name__ == '__main__':
    test()