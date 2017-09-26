"""
This is an implementation of the "Snake" sampler from
Mikkelsen et al
http://arxiv.org/pdf/1211.3126.pdf

"""
from __future__ import print_function

from builtins import zip
from builtins import map
from builtins import range
from builtins import object
import numpy as np
import random
import heapq
import sys

class Snake(object):
	def __init__(self, posterior, origin, spacing, threshold, pool=None):
		self.posterior=posterior
		self.origin=np.array(origin)
		self.spacing=np.array(spacing)
		self.ndim = len(origin)
		self.threshold = threshold
		self.best_fit=tuple([0 for i in range(self.ndim)])
		#Temporarily set the pool to None because we have
		#to evaluate the starting point
		self.pool = None
		self.best_fit_like = self.evaluate([self.best_fit])[1][0]
		self.has_blobs = hasattr(self.best_fit_like, "__len__")
		if self.has_blobs:
			self.best_fit_like, _ = self.best_fit_like
		self.likelihoods = {self.best_fit:self.best_fit_like}
		self.internal = set()
		self.surface = [(-self.best_fit_like,self.best_fit)]
		self.best_like_ever = self.best_fit_like
		self.iterations = 0
		self.pool=pool

	def has_adjacent(self, p):
		return self.adjacent_points(p, True)

	def adjacent_points(self, p, has=False):
		p = np.array(p, dtype=int)
		points = []
		#Loop through each dimension, checking
		#if adjacent points in each direction have been evaluated.
		for i in range(self.ndim):
			dx = np.zeros(self.ndim, dtype=int)
			dx[i] = 1
			# Check point in the +ve direction
			q = tuple(p+dx)
			if q not in self.likelihoods:
				if has: return True
				points.append(q)
			# Check point in the -ve direction
			q = tuple(p-dx)
			if q not in self.likelihoods:
				if has: return True
				points.append(q)
		if has: return False
		return points

	def find_best_fit(self):
		go_on = True
		while go_on:
			best_fit_like, best_fit = heapq.nsmallest(1, self.surface)[0]
			if self.has_adjacent(best_fit):
				go_on=False
			else:
				heapq.heappop(self.surface)
		self.best_fit = best_fit
		self.best_fit_like = -best_fit_like
		if self.best_fit_like>self.best_like_ever:
			self.best_like_ever = self.best_fit_like

	def converged(self):
		return (self.best_like_ever - self.best_fit_like > self.threshold)


	def iterate(self):
		self.iterations += 1
		#take the adjacent points from the best fit
		adjacent = self.adjacent_points(self.best_fit)
		#PARALLEL: take more random choices here
		if self.pool is None:
			n = 1
		else:
			n = min(self.pool.size, len(adjacent))

		P = random.sample(adjacent,n)

		#PARALLEL: do the full evaluation here
		X, outputs = self.evaluate(P)
		if self.has_blobs:
			likelihoods = [L[0] for L in outputs]
			blobs = [L[1] for L in outputs]
		else:
			likelihoods = [L for L in outputs]
			blobs = [None for i in range(len(outputs))]

		for (p,L) in zip(P, likelihoods):
			self.likelihoods[p] = L

		#We cannot combine this with the loop
		#above since we then we will not have filled
		#in the likelihoods for all the points nearby
		for (p,L) in zip(P, likelihoods):
			if self.has_adjacent(p):
				heapq.heappush(self.surface, (-L,p))
		
		if not self.has_adjacent(self.best_fit):
			#This point should now move off the surface
			#and into the interior.
			#This should remove the best fit
			self.surface.remove((-self.best_fit_like, self.best_fit))
			heapq.heapify(self.surface)
		#Definitely overkill to do this every time,
		#but better to be safe and as a nice by-product
		#it gives us a chance to do a convergence check
		#cleanly
		self.find_best_fit()
		return X, likelihoods, blobs

	def evaluate(self, indices):
		points = [self.origin + np.array(index)*self.spacing for index in indices]
		if self.pool is None:
			output = list(map(self.posterior, points))
		else:
			output = self.pool.map(self.posterior, points)
		return points, output


def test_like(x):
	L =  -0.5 * ((x[0]-0.5)**2 + (x[1]-0.8)**2)/0.02**2
	return L

def test():
	snake = Snake(test_like, [0.0, 0.0], [0.01, 0.01], None)
	while not snake.converged():
		snake.iterate()

	for p, L in list(snake.likelihoods.items()):
		print(p[0], p[1], L)
	sys.stderr.write("Ran for %d iterations\n"%snake.iterations)
if __name__ == '__main__':
	test()