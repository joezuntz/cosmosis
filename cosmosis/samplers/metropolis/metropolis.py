from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
import numpy as np


class Proposal(object):
    def __init__(self, cholesky, scaling=2.4, exponential_probability=0.333333):
        self.iteration = 0
        self.ndim = len(cholesky)
        self.cholesky = cholesky 
        rotation = np.identity(self.ndim)
        self.proposal_axes = np.dot(self.cholesky, rotation)
        self.scaling = scaling
        self.exponential_probability = exponential_probability

    def proposal_distance(self, ndim, scaling):
        #from CosmoMC
        if np.random.uniform()<self.exponential_probability:
            r = np.random.exponential()
        else:
            n = min(ndim,2)
            r = (np.random.normal(size=n)**2).mean()**0.5
        return r * scaling

    def randomize_rotation(self):
        #After CosmoMC, we randomly rotate our proposal axes
        #to avoid doubling back on ourselves
        rotation = random_rotation_matrix(self.ndim)
        #All our proposals are done along the axes of our covariance
        #matrix
        self.proposal_axes = np.dot(self.cholesky, rotation)

    def propose(self, p):
        #Once we have cycled through our axes, re-randomize them
        i = self.iteration%self.ndim
        if i==0:
            self.randomize_rotation()
        self.iteration += 1
        #otherwise, propose along our defined axes
        return p + self.proposal_distance(self.ndim,self.scaling) * self.proposal_axes[:,i]



class FastSlowProposal(Proposal):
    def __init__(self, covariance, fast_indices, slow_indices, oversampling, scaling=2.4, exponential_probability=0.3333):
        self.ordering = np.concatenate([slow_indices, fast_indices])
        self.inverse_ordering = invert_ordering(self.ordering)
        self.nslow = len(slow_indices)
        self.oversampling = oversampling
        self.iteration = 0
        self.slow_iteration = 0
        reordered_covariance = covariance[:,self.ordering][self.ordering]
        reordered_cholesky = np.linalg.cholesky(reordered_covariance)

        #For the fast subspace we just use the original vanilla proposal.
        #The slow proposal must be a little different  - not a square matrix
        self.fast_proposal = Proposal(reordered_cholesky[self.nslow:, self.nslow:], exponential_probability=exponential_probability)
        self.slow_matrix = reordered_cholesky[:,:self.nslow]

        self.slow_rotation = np.identity(self.nslow)
        self.scaling = scaling
        self.exponential_probability = exponential_probability



    def propose(self, p):
        p = p[self.ordering]
        if self.iteration%(self.oversampling+1)==0:
            q = self.propose_slow(p)
        else:
            q = self.propose_fast(p)
        self.iteration += 1
        return q[self.inverse_ordering]

    def propose_fast(self, p):
        q = np.zeros_like(p)
        q[:self.nslow] = p[:self.nslow]
        q[self.nslow:] += self.fast_proposal.propose(p[self.nslow:])
        return q
        
    def propose_slow(self, p):
        i = self.slow_iteration%self.nslow
        if i==0:
            self.randomize_rotation()   

        #Following the notation in Lewis (2013)
        delta_s = self.slow_rotation[i] * self.proposal_distance(self.nslow,self.scaling)
        self.slow_iteration += 1

        q = p + np.dot(self.slow_matrix, delta_s)
        return q
        

    def randomize_rotation(self):
        self.slow_rotation = random_rotation_matrix(self.nslow)



class MCMC(object):
    def __init__(self, start, posterior, covariance, quiet=False, 
        tuning_frequency=-1, tuning_grace=np.inf, tuning_end=np.inf, 
        scaling=2.4,
        exponential_probability=0.33333):
        """

        """
        #Set up basic variables
        self.posterior = posterior
        self.p = np.array(start)
        self.ndim = len(self.p)
        self.quiet=quiet
        #Run the pipeline for the first time, on the 
        #starting point
        self.Lp = self.posterior(self.p)

        #Proposal
        self.covariance = covariance
        cholesky = np.linalg.cholesky(covariance)
        self.scaling = scaling
        self.exponential_probability = exponential_probability
        self.proposal = Proposal(cholesky, scaling=scaling, exponential_probability=exponential_probability)

        #For adaptive sampling
        self.last_covariance_estimate = covariance.copy()       
        self.covariance_estimate = covariance.copy()
        self.S_estimate = np.zeros((self.ndim,self.ndim))
        #self.covariance_estimate.copy()
        self.mean_estimate = start.copy()
        self.tuning_frequency = tuning_frequency
        self.tuning_grace = tuning_grace
        self.tuning_end = tuning_end

        #Set up instance variables storing samples, etc.
        self.samples = []
        self.iterations = 0
        self.accepted = 0
        self.accepted_since_tuning = 0
        self.iterations_since_tuning = 0


    def sample(self, n):
        samples = []
        blobs = []
        #Take n sample mcmc steps
        for i in range(n):
            #this function is designed to be called
            #multiple times.  keep track of overall iteration number
            self.iterations += 1
            self.iterations_since_tuning += 1
            # proposal point and its likelihood
            q = self.proposal.propose(self.p)
            #assume two proposal subsets for now
            Lq = self.posterior(q)
            if not self.quiet:
                print("  ".join(str(x) for x in q))
            #acceptance test
            delta = Lq[0] - self.Lp[0]
            if  (delta>0) or (delta >= np.log(np.random.uniform())):
                #update if accepted
                self.Lp = Lq
                self.p = q
                self.accepted += 1
                self.accepted_since_tuning += 1
                if not self.quiet:
                    print("[Accept delta={:.3g}]\n".format(delta))
            elif not self.quiet:
                print("[Reject delta={:.3g}]\n".format(delta))

            #store next point
            self.update_covariance_estimate()
            if self.should_tune_now():
                self.tune()
            samples.append((self.p, self.Lp[0], self.Lp[1]))
        return samples

    def should_tune_now(self):
        return (    
            self.tuning_frequency>0
        and self.iterations>self.tuning_grace 
        and self.iterations%self.tuning_frequency==0
        and self.iterations<self.tuning_end
        )


    def update_covariance_estimate(self):
        n = self.iterations
        delta = (self.p - self.mean_estimate)
        self.mean_estimate += delta / n
        self.S_estimate  += np.outer(delta, delta)
        self.covariance_estimate = self.S_estimate / n

    def set_fast_slow(self, fast_indices, slow_indices, oversampling):
        self.fast_indices = fast_indices
        self.slow_indices = slow_indices
        self.oversampling = oversampling
        self.proposal = FastSlowProposal(self.covariance, fast_indices, slow_indices, oversampling, scaling=self.scaling, exponential_probability=self.exponential_probability)

    def tune(self):

        print("Cov 00 = ", self.covariance_estimate[0,0])
        f = (self.covariance_estimate.diagonal()**0.5-self.last_covariance_estimate.diagonal()**0.5)/self.last_covariance_estimate.diagonal()**0.5
        i = abs(f).argmax()
        print("Largerst parameter sigma fractional change = {:.1f}% for param {}".format(100*f[i], i))
        self.last_covariance_estimate = self.covariance_estimate.copy()

        print("Accepted since last tuning: {}%".format((100.*self.accepted_since_tuning)/self.iterations_since_tuning))
        self.accepted_since_tuning = 0
        self.iterations_since_tuning = 0


        if isinstance(self.proposal, FastSlowProposal):
            print("Tuning fast/slow sampler proposal.")
            self.proposal = FastSlowProposal(self.covariance_estimate, 
                self.fast_indices, self.slow_indices, self.oversampling, scaling=self.scaling, exponential_probability=self.exponential_probability)
        elif isinstance(self.proposal, Proposal):
            print("Tuning standard sampler proposal.")
            cholesky = np.linalg.cholesky(self.covariance_estimate)
            self.proposal = Proposal(cholesky, scaling=self.scaling, exponential_probability=self.exponential_probability)
        else:
            #unknown proposal type
            pass


#I've been copying this algorithm out of CosmoMC
#for the last decade.  Every time I need a new one
#I think to myself that this time I'll heed the warning
#in the CosmoMC code that, quote:
#!this is most certainly not the world's most efficient or 
#robust random rotation generator"
#unquote, and that I'll try and dig out a better one.
#And every time I spend about an hour looking, before
#coming back to this and translating it.
def random_rotation_matrix(n):
    R=np.identity(n)
    for j in range(n):
        while True:
            v = np.random.normal(size=n)
            for i in range(j):
                v -= R[i,:] * np.dot(v,R[i,:])
            L = np.dot(v,v)
            if (L>1e-3): break
        R[j,:] = v/L**0.5
    return R



def submatrix(M, x):
    """If x is an array of integer row/col numbers and M a matrix,
    extract the submatrix which is the all x'th rows and cols.
    i.e. A = submatrix(M,x) => A_ij = M_{x_i}{x_j}
    """
    return M[np.ix_(x,x)]


def invert_ordering(ordering):
    n = len(ordering)
    inverse = np.zeros_like(ordering)
    for i,j in enumerate(ordering):
        inverse[j] = i
    return inverse
