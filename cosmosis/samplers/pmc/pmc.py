from numpy import pi, dot, exp, einsum
import numpy as np


class PopulationMonteCarlo(object):
	"""
	A Population Monte Carlo (PMC) sampler,
	which combines expectation-maximization and
	importance sampling
	
	This code follows the notation and methodolgy in
	http://arxiv.org/pdf/0903.0837v1.pdf


	"""
	def __init__(self, posterior, n, start, sigma, pool=None, quiet=False):
		"""
		posterior: the posterior function
		n: number of components to use in the mixture
		start: estimated mean of the distribution
		sigmas: estimated std. devs. of the distribution
		pool (optional): an MPI or multiprocessing worker pool

		"""
		self.posterior = posterior
		mu = np.random.multivariate_normal(start, sigma, size=n)
		self.components = [GasussianComponent(1.0/n, m, sigma) for m in mu]
		self.pool = pool
		self.quiet=quiet #not currently used

	def sample(self, n):
		"Draw a sample from the Gaussian mixture and update the mixture"
		self.kill_count = n*1./len(self.components)/50.
		self.kill_alpha = 0.002
		self.kill = [False for c in self.components]
		#draw sample from current mixture
		x = self.draw(n)

		#calculate likelihoods
		if self.pool is None:
			samples = map(self.posterior, x)
		else:
			samples = pool.map(self.posterior, x)

		post = np.array([s[0] for s in samples])
		extra = [s[1] for s in samples]

		#update components
		weights = self.update_components(x, np.exp(post))

		return x, post, extra, weights

	def draw(self, n):
		"Draw a sample from the Gaussian mixture"
		A = [m.alpha for m in self.components]
		A = np.array(A)
		A/=A.sum()
		#Components to draw from
		N = np.arange(len(self.components))
		C = np.random.choice(N, size=n, replace=True, p=A)
		for i in N:
			if np.sum(C==i)<self.kill_count:
				self.kill[i] = True

		x = np.array([self.components[c].sample() for c in C])
		return x

	def update_components(self, x, post):
		"Equations 13-16 of arxiv.org 0903.0837v1"

		#x #n_sample*n_dim
		Aphi = np.array([m.alpha*m.phi(x) for m in self.components]) #n-component * n_sample
		w = post/Aphi.sum(0) #n_sample
		w_norm = w/w.sum()  #n_sample
		A = [m.alpha for m in self.components]
		#rho_top = einsum('i,ij->ij', A, phi)  #n_component * n_sample
		rho_bottom = Aphi.sum(0) #n_sample
		rho = [rho_t/rho_bottom for rho_t in Aphi]



		for d,(m,rho_d) in enumerate(zip(self.components, rho)):
			try:
				m.update(w_norm, x, rho_d, self.kill_alpha)
			except np.linalg.LinAlgError as error:
				print "Removing component", d, error.message
				self.kill[d] = True

		self.components = [c for c,kill in zip(self.components,self.kill) if not kill]

		return w



class GasussianComponent(object):
	"""
	A single Gaussian component of the mixture model.

	Could implememnt equations in the appendix of
	http://arxiv.org/pdf/0903.0837v1.pdf
	on another class with the same interface
	to implement the student's t distribution,
	which should sample distributions with heavier
	tails more efficiently.

	Anyone have a Masters student in need of a quick
	project?
	"""
	def __init__(self,alpha, mu, sigma):
		self.set(alpha, mu, sigma)

	def set(self, alpha, mu, sigma):
		"Set the parameters of this distribution component"
		self.alpha = alpha
		self.mu = mu
		ndim = len(self.mu)
		self.sigma = sigma
		self.sigma_inv = np.linalg.inv(self.sigma)
		self.A = (2*pi)**(-ndim/2.0) * np.linalg.det(self.sigma)**-0.5

	def update(self, w_norm, x, rho_d, kill_alpha):
		"Update the parameters according to the samples and rho values"
		alpha = dot(w_norm, rho_d)  #scalar
		if not alpha>kill_alpha:
			raise np.linalg.LinAlgError("alpha = %f"%kill_alpha)
		mu = einsum('i,ij,i->j',w_norm, x, rho_d) / alpha  #scalar
		delta = x-mu  #n_sample * n_dim
		sigma = einsum('i,ij,ik,i->jk',w_norm, delta, delta, rho_d) / alpha  #n_dim * n_dim
		self.set(alpha, mu, sigma)

	def phi(self, x):
		"Evaluate the distribution"
		d = (x-self.mu) #n_sample * n_dim
		chi2 = einsum('ij,jk,ik->i',d,self.sigma_inv,d)
		#result size n_sample
		return self.A * exp(-0.5*chi2)

	def sample(self):
		"Draw a sample from the distribution"
		return np.random.multivariate_normal(self.mu, self.sigma)





def test_like(p):
	x,y=p
	a = (x+y)/np.sqrt(2)
	b = (x-y)/np.sqrt(2)
	return 0.2*exp(-0.5*(a**2+b**2/25.0)) + 0.8*exp(-0.5*(a**2/25.0+b**2))


def test():
	pmc = PopulationMonteCarlo(test_like, 2, [0.0, 0.0], [1.0, 1.0])
	for i in xrange(50):
		s = pmc.sample(4000)
	import pylab
	x = s[:,0]
	y = s[:,1]
	# pylab.plot(x[:,0], x[:,1], '.')
	xmin=x.min()
	xmax=x.max()
	ymin=y.min()
	ymax=y.max()
	Y,X=np.mgrid[ymin:ymax:100j,xmin:xmax:100j]
	r = np.array(zip(X.flatten(),Y.flatten()))
	P = np.array([test_like((xi,yi) ) for (xi,yi) in r])
	P = P.reshape(X.shape)
	for c in pmc.components:
		pylab.contour(X, Y, c.phi(r).reshape(X.shape), 3)
	pylab.contourf(X,Y,P,50, cmap='Reds')
	pylab.plot(x, y, 'k,')
	pylab.show()


if __name__ == '__main__':
	test()
