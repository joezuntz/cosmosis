from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
import numpy as np
from .proposal.standard import Proposal, FastSlowProposal




class MCMC(object):
    def __init__(self, start, posterior, covariance, quiet=False,
        tuning_frequency=-1, tuning_grace=np.inf, tuning_end=np.inf, 
        scaling=2.4,
        exponential_probability=0.33333,
        use_cobaya=False):
        """

        """
        #Set up basic variables
        self.posterior = posterior
        self.p = np.array(start)
        self.ndim = len(self.p)
        self.quiet=quiet
        #Run the pipeline for the first time, on the 
        #starting point
        self.Lp = self.posterior(self.p)

        #Proposal
        self.covariance = covariance
        cholesky = np.linalg.cholesky(covariance)
        self.scaling = scaling
        self.exponential_probability = exponential_probability
        self.use_cobaya = use_cobaya


        if self.use_cobaya:
            from .proposal import cobaya
            # Initial proposer, without block - replace below
            self.proposal = cobaya.CobayaProposalWrapper(
                parameter_blocks=[np.arange(self.ndim)],
                proposal_scale=self.scaling)
            self.proposal.set_covariance(covariance)
        else:
            self.proposal = Proposal(cholesky, scaling=scaling, exponential_probability=exponential_probability)

        #For adaptive sampling
        self.last_covariance_estimate = covariance.copy()       
        self.covariance_estimate = covariance.copy()
        self.S_estimate = np.zeros((self.ndim,self.ndim))
        #self.covariance_estimate.copy()
        self.mean_estimate = start.copy()
        self.tuning_frequency = tuning_frequency
        self.original_tuning_frequency = tuning_frequency
        self.tuning_grace = tuning_grace
        self.tuning_end = tuning_end

        #Set up instance variables storing samples, etc.
        self.samples = []
        self.iterations = 0
        self.accepted = 0
        self.accepted_since_tuning = 0
        self.iterations_since_tuning = 0


    def sample(self, n):
        samples = []
        blobs = []
        #Take n sample mcmc steps
        for i in range(n):
            #this function is designed to be called
            #multiple times.  keep track of overall iteration number
            self.iterations += 1
            self.iterations_since_tuning += 1
            # proposal point and its likelihood
            q = self.proposal.propose(self.p)
            #assume two proposal subsets for now
            Lq = self.posterior(q)
            if not self.quiet:
                print("  ".join(str(x) for x in q))
            #acceptance test
            delta = Lq[0] - self.Lp[0]
            if  (delta>0) or (delta >= np.log(np.random.uniform())):
                #update if accepted
                self.Lp = Lq
                self.p = q
                self.accepted += 1
                self.accepted_since_tuning += 1
                if not self.quiet:
                    print("[Accept delta={:.3g}]\n".format(delta))
            elif not self.quiet:
                print("[Reject delta={:.3g}]\n".format(delta))

            #store next point
            self.update_covariance_estimate()
            if self.should_tune_now():
                self.tune()
            samples.append((self.p, self.Lp[0], self.Lp[1]))
        return samples

    def should_tune_now(self):
        return (    
            self.tuning_frequency>0
        and self.iterations>self.tuning_grace 
        and self.iterations%self.tuning_frequency==0
        and self.iterations<self.tuning_end
        )


    def update_covariance_estimate(self):
        n = self.iterations
        delta = (self.p - self.mean_estimate)
        self.mean_estimate += delta / n
        self.S_estimate  += np.outer(delta, delta)
        self.covariance_estimate = self.S_estimate / n

    def set_fast_slow(self, fast_indices, slow_indices, oversampling):
        self.fast_indices = fast_indices
        self.slow_indices = slow_indices
        self.oversampling = oversampling

        if self.use_cobaya:
            from .proposal import cobaya
            self.proposal = cobaya.CobayaProposalWrapper(
                blocks=[self.slow_indices, self.fast_indices], 
                oversampling_factors=[1, oversampling],
                i_last_slow_block=0,
                proposal_scale=self.scaling)
        else:
            self.proposal = FastSlowProposal(self.covariance, fast_indices, slow_indices, oversampling, scaling=self.scaling, exponential_probability=self.exponential_probability)
        self.tuning_frequency = self.original_tuning_frequency * oversampling

    def tune(self):
        f = (self.covariance_estimate.diagonal()**0.5-self.last_covariance_estimate.diagonal()**0.5)/self.last_covariance_estimate.diagonal()**0.5
        i = abs(f).argmax()
        print("Largest parameter sigma fractional change = {:.1f}% for param {}".format(100*f[i], i))
        self.last_covariance_estimate = self.covariance_estimate.copy()

        print("Accepted since last tuning: {}%".format((100.*self.accepted_since_tuning)/self.iterations_since_tuning))
        self.accepted_since_tuning = 0
        self.iterations_since_tuning = 0


        if self.use_cobaya:
            print("Tuning cobaya proposal.")
            self.proposal.set_covariance(self.covariance_estimate)
        elif isinstance(self.proposal, FastSlowProposal):
            print("Tuning fast/slow sampler proposal.")
            self.proposal = FastSlowProposal(self.covariance_estimate, 
                self.fast_indices, self.slow_indices, self.oversampling, scaling=self.scaling, exponential_probability=self.exponential_probability)
        elif isinstance(self.proposal, Proposal):
            print("Tuning standard sampler proposal.")
            cholesky = np.linalg.cholesky(self.covariance_estimate)
            self.proposal = Proposal(cholesky, scaling=self.scaling, exponential_probability=self.exponential_probability)
        else:
            #unknown proposal type
            pass

