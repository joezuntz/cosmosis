import numpy as np
from ...runtime import logs
from .proposal.standard import Proposal, FastSlowProposal
from copy import copy


class Bad(object):
    def __init__(self):
        self.post = -np.inf


class MCMC(object):
    def __init__(self, start, posterior, covariance,
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
        self.rng = np.random.default_rng()


        if self.n_drag > 0 and not self.use_cobaya:
            raise ValueError('You must set use_cobaya=T to have n_drag>0')

        if self.use_cobaya:
            from .proposal import cobaya_proposal
            # Initial proposer, without block - replace below
            self.proposal = cobaya_proposal.CobayaProposalWrapper(
                parameter_blocks=[np.arange(self.ndim)],
                oversampling_factors=[1],
                proposal_scale=self.scaling,
                random_state=self.rng,
                )
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
        #acceptance test
        delta = Lq.post - self.Lp.post
        if  accept(Lq.post, self.Lp.post):
            #update if accepted
            self.Lp = Lq
            self.p = q
            self.accepted += 1
            self.accepted_since_tuning += 1
            logs.info(f"[Accept delta={delta:.3g}")
        else:
            logs.info(f"[Reject delta={delta:.3g}]")

        return self.Lp


    def _sample_dragging(self):
        # get params with same fast params but different slow ones
        logs.noisy("Starting drag")
        logs.noisy(f"Current post = {self.Lp.post}")
        start = self.p
        end = self.proposal.propose_slow(start)


        # posteriors and derived parameters etc.
        r_start = copy(self.Lp)
        r_end = self.posterior(end)
        logs.noisy(f"slow proposal post = {r_end.post}")

        if not np.isfinite(r_end.post):
            logs.noisy("[Reject: nan/-inf posterior]\n")
            return self.Lp


        # coordinates of current start and end
        p1 = copy(start)
        p2 = copy(end)

        # results for current start and end
        r1 = copy(r_start)
        r2 = copy(r_end)

        start_post = r1.post
        end_post = r2.post

        drag_accepts = 0

        for i in range(self.n_drag):
            delta_fast = self.proposal.propose_fast(p1) - p1
            logs.debug("delta fast", delta_fast)
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
                logs.debug("[Accept drag step delta={:.3g}]\n".format(Q1 - P1))
            else:
                logs.debug("[Reject drag step delta={:.3g}]\n".format(Q1 - P1))

            start_post += r1.post
            end_post += r2.post

        logs.noisy("[Accepted {}/{} drag steps]".format(drag_accepts,self.n_drag))

        start_post /= self.n_drag
        end_post /= self.n_drag
        accept_overall = accept(end_post, start_post)

        logs.noisy("Done drag")
        if accept_overall:
            self.p = p2
            self.Lp = r_end
            self.accepted += 1
            self.accepted_since_tuning += 1
            logs.noisy("[Accept delta={:.3g}]\n".format(end_post - start_post))
            return r2
        else:
            logs.noisy("[Reject delta={:.3g}]\n".format(end_post - start_post))
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
            logs.warning("Cov estimate not SPD.  If this keeps happening, be concerned.")


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
                proposal_scale=self.scaling,
                random_state=self.rng,
                )
            self.proposal.set_covariance(self.covariance)
        else:
            self.proposal = FastSlowProposal(self.covariance, fast_indices, slow_indices, oversampling, scaling=self.scaling, exponential_probability=self.exponential_probability)
        self.tuning_frequency = self.original_tuning_frequency * oversampling

    def tune(self):
        self.update_covariance_estimate()

        f = (self.covariance_estimate.diagonal()**0.5-self.last_covariance_estimate.diagonal()**0.5)/self.last_covariance_estimate.diagonal()**0.5
        i = abs(f).argmax()
        logs.overview("Largest parameter sigma fractional change = {:.1f}% for param {}".format(100*f[i], i))
        self.last_covariance_estimate = self.covariance_estimate.copy()

        logs.overview("Accepted since last tuning: {}%".format((100.*self.accepted_since_tuning)/self.iterations_since_tuning))
        self.accepted_since_tuning = 0
        self.iterations_since_tuning = 0


        if self.use_cobaya:
            logs.overview("Tuning cobaya proposal.")
            self.proposal.set_covariance(self.covariance_estimate)
        elif isinstance(self.proposal, FastSlowProposal):
            logs.overview("Tuning fast/slow sampler proposal.")
            self.proposal = FastSlowProposal(self.covariance_estimate, 
                self.fast_indices, self.slow_indices, self.oversampling,scaling=self.scaling, exponential_probability=self.exponential_probability)
        elif isinstance(self.proposal, Proposal):
            logs.overview("Tuning standard sampler proposal.")
            cholesky = np.linalg.cholesky(self.covariance_estimate)
            self.proposal = Proposal(cholesky, scaling=self.scaling, exponential_probability=self.exponential_probability)
        else:
            #unknown proposal type
            pass


def accept(post1, post0):
    return (post1 > post0) or (post1-post0 > np.log(np.random.uniform(0,1)))

def is_positive_definite(M):
    return np.all(np.linalg.eigvals(M) > 0)
