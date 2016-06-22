from .. import ParallelSampler
from . import fisher
from ...datablock import BlockError
import numpy as np
import scipy.linalg
from ...runtime import prior

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
    v = []
    M = []
    for like_name in fisherPipeline.likelihood_names:
        v.append(data["data_vector", like_name + "_theory"])
        M.append(data["data_vector", like_name + "_inverse_covariance"])

    v = np.concatenate(v)
    M = scipy.linalg.block_diag(*M)

    #Might be only length-one, conceivably, so convert to a vector
    v = np.atleast_1d(v)
    M = np.atleast_2d(M)

    #Return numpy vector
    return v, M

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
        self.use_numdifftools = self.read_ini("use_numdifftools", bool, False)

        if self.output:
            for p in self.pipeline.extra_saves:
                name = '%s--%s'%p
                print "NOTE: You set extra_output to include parameter %s in the parameter file" % name
                print "      But the Fisher Sampler cannot do that, so this will be ignored."
                self.output.del_column(name)

        self.converged = False

    def compute_prior_matrix(self):
        #We include the priors as an additional matrix term
        #This is just added to the fisher matrix
        n = len(self.pipeline.varied_params)
        P = np.zeros((n,n))
        for i, param in enumerate(self.pipeline.varied_params):
            if isinstance(param.prior, prior.GaussianPrior):
                print "Applying additional prior sigma = {0} to {1}".format(param.prior.sigma2**0.5, param)
                print "This will be assumed to be centered at the parameter center regardless of what the ini file says"
                print 
                P[i,i] = 1./param.prior.sigma2
            elif isinstance(param.prior, prior.ExponentialPrior):
                print "There is an exponential prior applied to parameter {0}".format(param)
                print "This is *not* accounted for in the Fisher matrix"
                print
            #uniform prior should have no effect on the fisher matrix.
            #at least up until the assumptions of the FM are violated anyway
        return P




    def execute(self):
        #Load the starting point and covariance matrix
        #in the normalized space
        start_vector = self.pipeline.start_vector()
        for i,x in enumerate(start_vector):
            self.output.metadata("mu_{0}".format(i), x)
        start_vector = self.pipeline.normalize_vector(start_vector)

        #calculate the fisher matrix.
        #right now just a single step
        if self.use_numdifftools:
            fisher_class = fisher.NumDiffToolsFisher
        else:
            fisher_class = fisher.Fisher
        fisher_calc = fisher_class(compute_fisher_vector, start_vector, 
            self.step_size, self.tolerance, self.maxiter, pool=self.pool)

        fisher_matrix = fisher_calc.compute_fisher_matrix()
        fisher_matrix = self.pipeline.denormalize_matrix(fisher_matrix,inverse=True)

        P = self.compute_prior_matrix()
        fisher_matrix += P

        self.converged = True

        if self.converged:
            for row in fisher_matrix:
                self.output.parameters(row)

    def is_converged(self):
        return self.converged
