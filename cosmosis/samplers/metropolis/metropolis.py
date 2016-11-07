import numpy as np

class MCMC(object):
	def __init__(self, start, posterior, covariance, quiet=False):
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

		#Set up the covariance and initial
		#proposal axes
		covariance = covariance
		self.cholesky = np.linalg.cholesky(covariance)
		rotation = np.identity(self.ndim)
		self.proposal_axes = np.dot(self.cholesky, rotation)

		self.last_subset_index = 0



		#Set up instance variables storing samples, etc.
		self.samples = []
		self.iterations = 0
		self.accepted = 0
		self.proposal_scale = 2.4
		self.subsets = None

	def set_subsets(self, subsets):
		#If there are sub-spaces defined that split the space 
		#into fast and slow parameters, then find the subsets here.
		#Each entry in the input parameter "subsets" should be the pair
		#(indices,oversampling)
		self.subsets = subsets
		if subsets:
			self.nsubsets = len(subsets)
			self.subset_axes = []
			self.subset_cholesky = []
			self.subset_iterations = []
			self.subset_oversampling = []
			self.subsets = []
			self.current_subset_index = 0
			for s,oversampling in subsets:
				chol = self.submatrix(self.cholesky, s)
				axes = np.dot(chol, np.identity(len(s)))
				self.subset_cholesky.append(chol)
				self.subset_axes.append(axes)
				self.subset_iterations.append(0)
				self.subset_oversampling.append(oversampling)
				self.subsets.append(s)


	@staticmethod
	def submatrix(M, x):
		"""If x is an array of integer row/col numbers and M a matrix,
		extract the submatrix which is the all x'th rows and cols.
		i.e. A = submatrix(M,x) => A_ij = M_{x_i}{x_j}
		"""
		return M[np.ix_(x,x)]

	def sample(self, n):
		samples = []
		blobs = []
		#Take n sample mcmc steps
		for i in xrange(n):
			#this function is designed to be called
			#multiple times.  keep track of overall iteration number
			self.iterations += 1
			# proposal point and its likelihood
			q = self.propose()
			if not self.quiet:
				print "  ".join(str(x) for x in q)
			#assume two proposal subsets for now
			Lq = self.posterior(q)
			if not self.quiet:
				print
			#acceptance test
			if  Lq[0] >= self.Lp[0] or  (Lq[0] - self.Lp[0]) >= np.log(np.random.uniform()):
				#update if accepted
				self.Lp = Lq
				self.p = q
				self.accepted += 1
			#store next point
			samples.append((self.p, self.Lp[0], self.Lp[1]))
		return samples

	def randomize_rotation(self):
		#After CosmoMC, we randomly rotate our proposal axes
		#to avoid doubling back on ourselves
		rotation = random_rotation_matrix(self.ndim)
		#All our proposals are done along the axes of our covariance
		#matrix
		self.proposal_axes = np.dot(self.cholesky, rotation)

	def tune(self):
		#Not yet implemented!
		pass

	def propose(self):
		#Different method if we are using subsets
		if self.subsets:
			return self.propose_subset()
		else:
			return self.propose_vanilla()

	def propose_vanilla(self):
		#Once we have cycled through our axes, re-randomize them
		i = self.iterations%self.ndim
		if i==0:
			self.randomize_rotation()
		#otherwise, propose along our defined axes
		return self.p + proposal_distance(self.ndim) * self.proposal_scale \
				* self.proposal_axes[:,i]


	def propose_subset(self):
		#
		subset_index = self.current_subset_index
		n = len(self.subsets[subset_index])
		print "Proposing in parameter subset {}: {}".format(subset_index, self.subsets[subset_index])
		q = self.propose_subset_index(subset_index)
		self.subset_iterations[subset_index] += 1

		#if we have done one sample in each axis in the subset then 
		#randomize its ordering
		i_s = self.subset_iterations[subset_index]%n
		if i_s==0:
			self.randomize_rotation_subset(subset_index)

		self.last_subset_index = subset_index
		#If we have sufficiently oversampled this subset move on to the
		#next subset
		if self.subset_iterations[subset_index]%self.subset_oversampling[subset_index]==0:
			self.current_subset_index = (self.current_subset_index+1)%self.nsubsets

		return q

	def propose_subset_index(self, subset_index):
		s = self.subsets[subset_index]
		i_s = self.subset_iterations[subset_index]%len(s)
		axes = self.subset_axes[subset_index]
		q = self.p.copy()
		q[s] += proposal_distance(len(s)) * self.proposal_scale * axes[:,i_s]
		return q

	def randomize_rotation_subset(self, subset_index):
		s = self.subsets[subset_index]
		rotation = random_rotation_matrix(len(s))
		cholesky = self.subset_cholesky[subset_index]
		self.subset_axes[subset_index] = np.dot(cholesky, rotation)

def proposal_distance(ndim):
	# See http://cosmologist.info/notes/CosmoMC.pdf
	#for more details on this
	n = min(ndim,2)
	r = (np.random.normal(size=n)**2).mean()**0.5
	return r

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
    for j in xrange(n):
        while True:
            v = np.random.normal(size=n)
            for i in xrange(j):
                v -= R[i,:] * np.dot(v,R[i,:])
            L = np.dot(v,v)
            if (L>1e-3): break
        R[j,:] = v/L**0.5
    return R
