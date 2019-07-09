import numpy as np

def setup(options):
    return {}

def execute(block, config):
    p1 = block['parameters', 'p1']
    p2 = block['parameters', 'p2']
    like = -(p1**2 + p2**2)/2.
    block['parameters', 'p3'] = p1 + p2
    block['likelihoods', 'test_like'] = like
    block['data_vector', 'test_theory'] = np.array([p1,p2])
    block['data_vector', 'test_inverse_covariance'] = np.array([[1., 0.,],[0., 1.,]])
    return 0
