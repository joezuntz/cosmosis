"""

This sampler uses the notation of Allison & Dunkley 2013/2014

"""
import collections
import numpy as np

class NestedSampler(object):
	def __init__(self, likelihood_function, starting_points, pool=None, factor=1.06, cache_size=None):
		""" The starting points are presumed to be drawn from the prior.  That is up to the user. """
		self.likelihood_function = likelihood_function
		self.pool = pool
		self.i = 0
		self.f2 = factor**2
		if np.isscalar(starting_points[0]):
			self.np=1
		else:
			self.np = len(starting_points[0])
		# We could do this in parallel
		likes = self.map(self.likelihood_function, starting_points)
		order = np.argsort(likes)

		self.points = list(starting_points[order])
		self.likes = [likes[o] for o in order]
		self.point_cache = []
		if cache_size is None:
			if pool is None:
				self.cache_size = 10
			else:
				self.cache_size = len(pool) - 1
		else:
			self.cache_size = cache_size



	def map(self, function, tasks):
		if self.pool:
			return self.pool.map(function, tasks)
		else:
			return map(function, tasks)

	def compute_bounding_ellipse(self):
		if self.np==1:
			points = np.array(self.points).squeeze()
		else:
			points = self.points
		self.mu = np.mean(points)
		self.C = np.cov(points)
		if self.np==1:
			self.iC = 1.0/self.C
			self.R = 1.0
			self.D = self.C
		else:
			self.iC = np.linalg.inv(self.C)
			eigenvalues, eigenvectors = np.linalg.eig(self.C)
			self.R = np.array(eigenvectors)
			self.D = np.diag(eigenvalues)
		self.k = self.f2 * np.max([np.dot(p-self.mu, np.dot(self.iC, p-self.mu)) for p in self.points])



	def generate_sample(self):
		w = np.random.randn(self.np)
		w_abs = np.sqrt(np.dot(w,w))
		u = np.random.uniform()
		z = (w/w_abs) * u ** (1.0/self.np)
		T = np.sqrt(self.k) * np.dot(self.R, np.dot(self.D, self.R))
		y = self.mu + np.dot(T,z)
		return y

	def refill_cache(self):
		self.compute_bounding_ellipse()		
		samples = [self.generate_sample() for i in xrange(self.cache_size)]
		likes = self.map(self.likelihood_function, samples)
		# print "REFILL"
		# print samples
		# print likes
		self.point_cache = zip(samples, likes)

	def get_sample(self):
		if len(self.point_cache)==0:
			self.refill_cache()
		return self.point_cache.pop()


	def iterate(self):
		#The least likely point in the active set is the first one
		#since they are sorted
		least_likely = 0
		L_i = self.likes[least_likely]
		L = -np.inf
		#We can re-write this so we have a collection of valid points ready
		#for if the first one fails.  Then we can keep a stack ready
		while L < L_i:
			p, L = self.get_sample()
		self.point_cache=[]
		i = np.searchsorted(self.likes, L)
		self.points.insert(i, p)
		self.likes.insert(i, L)
		p_out = self.points.pop(0)
		L_out = self.likes.pop(0)
		self.i+=1
		return p_out, L_out

	def sample(self, n):
		P = np.zeros((n, self.np))
		L = np.zeros(n)
		for i in xrange(n):
			P[i], L[i] = self.iterate()
		L = np.exp(L)
		i = np.arange(n,dtype=float)
		N = len(self.points)
		X = np.exp(-i/N)
		w = -np.concatenate(([X[1]-X[0]], 0.5*(X[2:] - X[:-2]), [X[-1]-X[-2]]))
		L_active = np.exp(self.likes)
		Z = (L*w).sum() + L_active.mean()*X[-1]

		#weights of the samples
		x = list(P) + self.points
		#print L_active, X[-1], Z, N
		p = np.concatenate([L*w/Z, L_active*X[-1]/(N*Z)])
		return Z, x, p
		
		

