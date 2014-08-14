class MCMC(object):
	def __init__(self, start, limits, posterior, covariance=None, pool=None):
		self.pool = pool
		self.posterior = posterior
		self.p = np.array(start)
		self.Lp = self.posterior(self.p)
		self.limits = np.array(limits)
		self.covariance = covariance
		self.samples = []

	def sample(self, n):
		samples = []
		for i in xrange(n):
			q = self.propose()
			Lq = self.posterior(q)
			if Lq>self.Lp or Lq-self.Lp>np.log(np.random.uniform()):
				self.Lp = Lq
				self.p = q
			samples.append((self.p, self.Lp))
		self.samples.extend(samples)
		return samples

	def tune(self):
		pass

	def propose(self):
		pass