from .. import ParallelSampler
from .fisher import compute_fisher_matrix
from ...datablock import BlockError
import numpy as np

def compute_fisher_vector(p):
    # use normalized parameters - fisherPipeline is a global
    # variable because it has to be picklable)
    try:
        x = fisherPipeline.denormalize_vector(p)
    except ValueError:
        print "Parameter vector outside limits: %r" % p
        return None
    print x
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
        vector = data['data_vector', 'vector']
        inv_cov = data['data_vector', 'inv_covariance']
    except BlockError:
        raise ValueError("The pipeline you write for the Fisher sampler should save a vector 'vector' and inverse covmat 'inv_covariance' in a section called 'fisher'")

    #Might be only length-one, conceivably, so convert to a vector
    vector = np.atleast_1d(vector)
    inv_cov = np.atleast_2d(inv_cov)

    #Return numpy vector
    return vector, inv_cov

class SingleProcessPool(object):
    def map(self, function, tasks):
        return map(function, tasks)

class FisherSampler(ParallelSampler):
    sampler_outputs = []
    parallel_output = False

    def config(self):
        #Save the pipeline as a global variable so it
        #works okay with MPI
        global fisherPipeline
        fisherPipeline = self.pipeline
        self.step_size = self.read_ini("step_size", float, 0.01)
        self.tolerance = self.read_ini("tolerance", float, 0.01)
        self.maxiter = self.read_ini("maxiter", int, 10)

        self.converged = False

    def execute(self):
        #Load the starting point and covariance matrix
        #in the normalized space
        start_vector = self.pipeline.start_vector()
        for i,x in enumerate(start_vector):
            self.output.metadata("mu_{0}".format(i), x)
        start_vector = self.pipeline.normalize_vector(start_vector)

        #calculate the fisher matrix.
        #right now just a single step
        fisher_calc = fisher.Fisher(compute_fisher_vector, start_vector, 
            self.step_size, self.tolerance, self.maxiter, pool=self.pool)

        fisher_matrix = fisher_calc.compute_fisher_matrix()

        self.converged = True

        if self.converged:
            for row in fisher_matrix:
                self.output.parameters(row)

    def is_converged(self):
        return self.converged
