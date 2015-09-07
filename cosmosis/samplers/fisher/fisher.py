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

class Fisher(object):
  def __init__(self,
               compute_vector, inv_covmat, step_size, ell_arr
               tolerance, maxiter,
               pool=None):
    
    self.inverse_covmat = inv_covmat
    self.compute_vector = compute_vector
    self.maxiter = maxiter
    self.step_size = step_size
    self.ell_arr = ell_arr
    self.start_params = start_params
    self.current_params = start_params
    self.nparams = start_vector.shape[0]
    self.iterations = 0

  def converged(self):
    crit = (abs(self.new_onesigma - self.old_onesigma).max() < self.threshold)
    return crit

  def converge_fisher_matrix(self):
    
    self.new_Fmatrix = compute_fisher_matrix(self.compute_vector,
                                             self.start_vector,
                                             self.start_covmat)
    
    self.old_onesigma = compute_one_sigma(new_Fmatrix)

    while True:
      self.iterations+=1
      self.old_onesigma = self.new_onesigma
      self.current_params = self.choose_new_params(self.new_Fmatrix)
      #self.current_cl = self.pool.map(self.compute_vector,
      #                                 self.current_params)

      self.new_Fmatrix = compute_fisher_matrix()

      self.new_onesigma = compute_one_sigma(self.new_Fmatrix)

      if self.converged():
        print 'Fisher has converged!'
        return new_Fmatrix

      if self.iterations > self.maxiter:
        print "Run out of iterations."
        print "Done %d, max allowed %d" % (self.iterations, self.maxiter)
        return None

  def compute_fisher_matrix():

    dCl = compute_five_point_stencil_deriv()

    delta_ell = np.diff(self.ell_arr)
    delta_ell = np.concatenate([[delta_ell[0]], delta_ell])
    Fmatrix = (2.*ell_arr+1.)*delta_ell*dCl[0]*self.inverse_covmat*dCl[1]
    print 'ToDo: choose the axis!'
    pdb.set_trace()
    return Fmatrix

  def compute_five_point_stencil_deriv():
    # ToDo: would be nicer to compute the different steps as a pool too,
    # but this would require a pool within a pool?
    Cl_plustwo = self.pool.map(self.compute_vector,
                               self.current_params + 2*self.step_size)
    Cl_plusone = self.pool.map(self.compute_vector,
                               self.current_params + 1*self.step_size)
    Cl_minustwo = self.pool.map(self.compute_vector,
                                self.current_params - 2*self.step_size)
    Cl_minusone = self.pool.map(self.compute_vector,
                                self.current_params - 1*self.step_size)

    deriv = (-Cl_plustwo + 8*Cl_plusone
             - 8*Cl_minusone + Cl_minustwo)/(12*self.step_size)

    return deriv

  def compute_one_sigma(Fmatrix):
    sigma = np.sqrt(np.linalg.inv(Fmatrix))

    return sigma
