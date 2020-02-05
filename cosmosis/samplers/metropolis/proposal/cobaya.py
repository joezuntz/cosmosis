# cobaya imports matplotlib, which does
# some heavy duty output when logging is set to high.
# This actually reflects my lack of understanding of the
# python logging mechanism, and I should really go through
# and fix all of this, but for now we just temporarily suppress
# the debug info
import logging
logger = logging.getLogger()
level = logger.level
logger.setLevel(logging.WARNING)
from cobaya.samplers.mcmc.proposal import BlockedProposer
logger.setLevel(level)


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
