from .. import Sampler
from .fisher import compute_fisher_matrix
import numpy as np

def compute_fisher_vector(p):
    # use normalized parameters - fisherPipeline is a global
    # variable that will be set during config (because it has
    # to be picklable)
    try:
        x = fisherPipeline.denormalize_vector(p)
    except ValueError:
        print "Parameter vector outside limits: %r" % p
        return None

    #Run the pipeline, generating a data block
    data = fisherPipeline.run_parameters(x)

    #If the pipeline failed, return "None"
    #This might happen if the parameters stray into
    #a bad region.
    if data is None:
        return None

    #Get out the fisher vector.  Failing on this is definitely an error
    #since if the pipeline finishes it must have a fisher vector if it
    #has been acceptably designed.
    try:
        data['fisher', 'vector']
    except BlockError:
        raise ValueError("The pipeline you write for the Fisher sampler should save a vector 'data' in a section called 'fisher'")

    #Might be only length-one, conceivably, so convert to a vector
    vector = np.atleast_1d(vector)

    #Return numpy vector
    return vector

class SingleProcessPool(object):
    def map(self, function, tasks):
        return map(function, tasks)

class FisherSampler(ParallelSampler):
    sampler_outputs = []

    def config(self):
        #Save the pipeline as a global variable so it
        #works okay with MPI
        global fisherPipeline
        fisherPipeline = self.pipeline

        self.covariance = self.read_ini("covariance",str)
        self.converged = False

    def execute(self):
        #Load the starting point and covariance matrix
        #in the normalized space
        start_vector = self.pipeline.normalize_vector(self.pipeline.start_vector())
        covmat = self.pipeline.normalize_matrix(np.loadtxt(self.covariance))

        pool = self.pool if self.pool else SingleProcessPool()
        fisher_matrix = compute_fisher_matrix(compute_fisher_vector, start_vector, 
            covmat, pool)

        fisher_matrix = self.pipeline.denormalize_matrix(fisher_matrix)

        self.converged = True

    def is_converged(self):
        return self.converged
