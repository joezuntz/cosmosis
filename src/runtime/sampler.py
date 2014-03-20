import config

class Sampler(object):
    def __init__(self, ini, pipeline):
        self.ini = ini
        self.pipeline = pipeline

    def config(self):
        ''' Set up sampler (could instead use __init__) '''
        pass

    def execute(self):
        ''' Run one (self-determined) iteration of sampler.  Should be enough to test convergence '''
        raise NotImplementedError

    def is_converged(self):
        return False
