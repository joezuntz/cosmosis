import numpy as np

class MCMC(object):
	def __init__(self, start,posterior, covariance):
		#Set up basic variables
		self.posterior = posterior
		self.p = np.array(start)
		self.ndim = len(self.p)
		#Run the pipeline for the first time, on the 
		#starting point
		self.Lp = self.posterior(self.p)

		#Set up the covariance and initial
		#proposal axes
		self.covariance = covariance
		self.cholesky = np.linalg.cholesky(covariance)
		self.rotation = np.identity(self.ndim)
		self.proposal_axes = np.dot(self.cholesky, self.rotation)

		#Set up instance variables storing samples, etc.
		self.samples = []
		self.iterations = 0
		self.accepted = 0
		self.proposal_scale = 2.4

	def sample(self, n):
		samples = []
		#Take n sample mcmc steps
		for i in xrange(n):
			#this function is designed to be called
			#multiple times.  keep track of overall iteration number
			self.iterations += 1
			# proposal point and its likelihood
			q = self.propose()
			Lq = self.posterior(q)
			print "  ".join(str(x) for x in q), Lq[0]
			#acceptance test
			if  Lq[0] >= self.Lp[0] or  (Lq[0] - self.Lp[0]) >= np.log(np.random.uniform()):
				#update if accepted
				self.Lp = Lq
				self.p = q
				self.accepted += 1
			#store next point
			samples.append((self.p, self.Lp[0]))
		return samples

	def randomize_rotation(self):
		#After CosmoMC, we randomly rotate our proposal axes
		#to avoid doubling back on ourselves
		self.rotation = random_rotation_matrix(self.ndim)
		#All our proposals are done along the axes of our covariance
		#matrix
		self.proposal_axes = np.dot(self.cholesky, self.rotation)

	def tune(self):
		#Not yet implemented!
		pass

	def propose(self):
		#Once we have cycled through our axes, re-randomize them
		i = self.iterations%self.ndim
		if i==0:
			self.randomize_rotation()
		#otherwise, propose along our defined axes
		return self.p + proposal_distance(self.ndim) * self.proposal_scale \
				* self.proposal_axes[:,i]

def proposal_distance(ndim):
	# See http://cosmologist.info/notes/CosmoMC.pdf
	#for more details on this
	r = (np.random.normal(size=ndim)**2).mean()**0.5
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
