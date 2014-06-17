import nested_sampler
import numpy as np


def like_1d(x):
	mu = 1.0
	sigma = 0.1
	return float(-0.5 * (x-mu)**2 / sigma**2)

def like_2d(x):
	mu = np.array([1.0, 2.0])
	sigma = np.array([0.1, 0.2])
	return -0.5 * ((x-mu)**2 / sigma**2).sum()


starting_points = np.random.uniform(0.0, 2.0, 500)
nest = nested_sampler.NestedSampler(like_1d, starting_points)
Z, x, p = nest.sample(10000)