import numpy as np

def setup(options):
    return {}

def execute(block, config):
    p3 = block['parameters', 'p3']
    p4 = block['parameters', 'p4']
    like = -(p3**2 + p4**2)/2.
    block['likelihoods', 'test3_like'] = like
    return 0
