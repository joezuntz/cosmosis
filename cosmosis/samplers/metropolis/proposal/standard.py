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
