from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
import numpy as np
from .proposal.standard import Proposal, FastSlowProposal
from copy import copy


class Bad(object):
    def __init__(self):
        self.post = -np.inf


class MCMC(object):
    def __init__(self, start, posterior, covariance, quiet=False,
        tuning_frequency=-1, tuning_grace=np.inf, tuning_end=np.inf, 
        scaling=2.4,
        exponential_probability=0.33333,
        use_cobaya=False,
        n_drag=0):
        """
        posterior should return a PipelineResults object
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
        self.n_drag = n_drag
        self.fast_indices = None
        self.slow_indices = None
        self.oversampling = None
        self.fast_slow_is_ready = False

        if self.n_drag > 0 and not self.use_cobaya:
            raise ValueError('You must set use_cobaya=T to have n_drag>0')

        if self.use_cobaya:
            from .proposal import cobaya_proposal
            # Initial proposer, without block - replace below
            self.proposal = cobaya_proposal.CobayaProposalWrapper(
                parameter_blocks=[np.arange(self.ndim)],
                proposal_scale=self.scaling)
            self.proposal.set_covariance(covariance)
        else:
            self.proposal = Proposal(cholesky, scaling=scaling, exponential_probability=exponential_probability)

        #For adaptive sampling
        self.last_covariance_estimate = covariance.copy()       
        self.covariance_estimate = covariance.copy()
        self.chain = []
        self.n_cov_fail = 0

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

        sample_method = (
            self._sample_dragging 
            if (self.fast_slow_is_ready and self.n_drag) 
            else self._sample_metropolis)
        #Take n sample mcmc steps

        samples = []
        for i in range(n):
            # update counts
            self.iterations += 1
            self.iterations_since_tuning += 1

            # generate a new sample
            s = sample_method()
            samples.append(s)
            # hack - should unify these
            self.chain.append(s.vector)

            if self.should_tune_now():
                self.tune()

        return samples

    def _sample_metropolis(self):
        # proposal point and its likelihood
        q = self.proposal.propose(self.p)
        #assume two proposal subsets for now
        Lq = self.posterior(q)
        if not self.quiet:
            print("  ".join(str(x) for x in q))
        #acceptance test
        delta = Lq.post - self.Lp.post
        if  accept(Lq.post, self.Lp.post):
            #update if accepted
            self.Lp = Lq
            self.p = q
            self.accepted += 1
            self.accepted_since_tuning += 1
            if not self.quiet:
                print("[Accept delta={:.3g}]\n".format(delta))
        elif not self.quiet:
            print("[Reject delta={:.3g}]\n".format(delta))

        return self.Lp


    def _sample_dragging(self):
        # get params with same fast params but different slow ones
        start = self.p
        end = self.proposal.propose_slow(start)

        # posteriors and derived parameters etc.
        r_start = copy(self.Lp)
        r_end = self.posterior(end)

        if not np.isfinite(r_end.post):
            if not self.quiet:
                print("[Reject: nan/-inf posterior]\n")
            return self.Lp


        # coordinates of current start and end
        p1 = copy(start)
        p2 = copy(end)

        # results for current start and end
        r1 = copy(r_start)
        r2 = copy(r_end)

        drag_accepts = 0

        for i in range(self.n_drag):
            delta_fast = self.proposal.propose_fast(p1) - p1
            q1 = p1 + delta_fast
            q2 = p2 + delta_fast

            s1 = self.posterior(q1)

            if np.isfinite(s1.post):
                s2 = self.posterior(q2)
            else:
                s2 = Bad()


            f = (1+i) /(1+self.n_drag)

            P1 = (1-f)*r1.post + f*r2.post
            Q1 = (1-f)*s1.post + f*s2.post

            accept_drag = accept(Q1, P1) and np.isfinite(s1.post) and np.isfinite(s2.post)

            if accept_drag:
                p1 = q1
                p2 = q2
                r1 = s1
                r2 = s2
                drag_accepts += 1

            r_start.post += r1.post
            r_end.post += r2.post

        if not self.quiet:
            print("[Accepted {}/{} drag steps]".format(drag_accepts,self.n_drag))

        r_start.post /= self.n_drag
        r_end.post /= self.n_drag
        accept_overall = accept(r_end.post, r_start.post)

        if accept_overall:
            self.p = p2
            self.Lp = r_end
            self.accepted += 1
            self.accepted_since_tuning += 1
            if not self.quiet:
                print("[Accept delta={:.3g}]\n".format(r_end.post - r_start.post))
            return r2
        else:
            if not self.quiet:
                print("[Reject delta={:.3g}]\n".format(r_end.post - r_start.post))
            return self.Lp



    def should_tune_now(self):
        return (    
            self.tuning_frequency>0
        and self.iterations>self.tuning_grace 
        and self.iterations%self.tuning_frequency==0
        and self.iterations<self.tuning_end
        )


    def update_covariance_estimate(self):
        n = self.iterations
        self.mean_estimate = np.mean(self.chain, axis=0)
        C = np.cov(np.transpose(self.chain))
        if is_positive_definite(C):
            self.covariance_estimate = C
        else:
            print("Cov estimate not SPD.  If this keeps happening, be concerned.")
            # chain_outfile = 'joe_dump_chain_{}.txt'.format(self.n_cov_fail)
            # cov_outfile = 'joe_dump_cov_{}.txt'.format(self.n_cov_fail)
            # self.n_cov_fail += 1
            # np.savetxt(chain_outfile, self.chain)
            # np.savetxt(cov_outfile, np.transpose(C))
            # print("TEMPORARY (JOE - REMOVE LATER) - dumping to file")


    def set_fast_slow(self, fast_indices, slow_indices, oversampling):
        if self.n_drag:
            oversampling = 1
            print("Overriding oversampling parameter -> 1 since using dragging")
        self.fast_indices = fast_indices
        self.slow_indices = slow_indices
        self.oversampling = oversampling
        self.fast_slow_is_ready = True

        if self.use_cobaya:
            from .proposal import cobaya_proposal
            self.proposal = cobaya_proposal.CobayaProposalWrapper(
                parameter_blocks=[self.slow_indices, self.fast_indices], 
                oversampling_factors=[1, oversampling],
                i_last_slow_block=0,
                proposal_scale=self.scaling)
            self.proposal.set_covariance(self.covariance)
        else:
            self.proposal = FastSlowProposal(self.covariance, fast_indices, slow_indices, oversampling, scaling=self.scaling, exponential_probability=self.exponential_probability)
        self.tuning_frequency = self.original_tuning_frequency * oversampling

    def tune(self):
        self.update_covariance_estimate()

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
                self.fast_indices, self.slow_indices, self.oversampling,scaling=self.scaling, exponential_probability=self.exponential_probability)
        elif isinstance(self.proposal, Proposal):
            print("Tuning standard sampler proposal.")
            cholesky = np.linalg.cholesky(self.covariance_estimate)
            self.proposal = Proposal(cholesky, scaling=self.scaling, exponential_probability=self.exponential_probability)
        else:
            #unknown proposal type
            pass


def accept(post1, post0):
    return (post1 > post0) or (post1-post0 > np.log(np.random.uniform(0,1)))

def is_positive_definite(M):
    return np.all(np.linalg.eigvals(M) > 0)
