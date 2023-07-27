from cobaya.samplers.mcmc.proposal import BlockedProposer

# The cobaya proposal updates the parameter vector in-place,
# so we wrap it slightly to make a copy instead, as expected
# by the metropolis sampler
class CobayaProposalWrapper(BlockedProposer):
    def propose(self, p):
        q = p.copy()
        self.get_proposal(q)
        return q
    def propose_slow(self, p):
        q = p.copy()
        self.get_proposal_slow(q)
        return q
    def propose_fast(self, p):
        q = p.copy()
        self.get_proposal_fast(q)
        return q
