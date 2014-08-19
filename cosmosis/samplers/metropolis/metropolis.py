import numpy as np

class MCMC(object):
	def __init__(self, start,posterior, covariance=None, pool=None):
		self.pool = pool
		self.posterior = posterior
		self.p = np.array(start)
		self.Lp = self.posterior(self.p)
		self.covariance = covariance
		self.samples = []

	def sample(self, n):
		samples = []
		for i in xrange(n):
			qparam = self.propose(i)
			q = np.copy(self.p)
			q[i] = qparam
			Lq = self.posterior(q)
			if  Lq[0] >= self.Lp[0] or  (Lq[0] - self.Lp[0]) >= np.log(np.random.uniform()):
				self.Lp = Lq
				self.p = q
		samples.append((self.p, self.Lp[0]))
		return samples

	def tune(self):
		pass

	def propose(self,i):
		return (np.random.normal(loc = self.p[i],scale = (self.covariance[i][i])**0.5))


		
