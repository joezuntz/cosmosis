
import numpy as np
import sys
import os


class Analytics(object):
    def __init__(self, params, pool=None):
        self.params = params
        self.pool = pool

        self.total_steps = 0
        self.means = np.zeros(len(params))
        self.m2 = np.zeros(len(params))
        self.best_like = -np.inf
        self.best_index = None
        self.best_params = None

    def add_traces(self, traces):
        if traces.shape[1] != len(self.params):
            raise RuntimeError("The number of traces added to Diagnostics "
                               "does not match the number of varied "
                               "parameters!")
        like_col = self.params.index("LIKE")
        num = float(self.total_steps)
        for i,x in enumerate(traces):
            num += 1.0
            delta = x - self.means
            self.means += delta/num
            self.m2 += delta*(x - self.means)
            if like_col>-1:
                like = x[like_col]
                if like>self.best_like:
                    self.best_like = like
                    self.best_index = self.total_steps + i
                    self.best_params = x

        self.total_steps += traces.shape[0]

    @classmethod
    def from_chain_files(cls, filenames, burn, thin):
        if isinstance(filenames, str):
            filenames = [filenames]

        #currently only works on text files.
        #TODO: rejig to use "load" methods on output objects
        params = open(filenames[0]).readline().strip('#').split()

        analytics = cls(params)
        for filename in filenames:
            chain = np.genfromtxt(filename).T
            if burn<1:
                nburn = len(chain[0]) * burn
            else:
                nburn = burn
            chain = chain[:,nburn:]
            if thin:
                chain = chain[:,::thin]
            analytics.add_traces(chain.T)
        return analytics

    def trace_means(self):
        if self.pool:
            return np.array(self.pool.gather(self.means)).T
        else:
            return self.means

    def trace_variances(self):
        if self.total_steps > 1:
            local_variance = self.m2 / float(self.total_steps-1)
            if self.pool:
                return np.array(self.pool.gather(local_variance)).T
            else:
                return local_variance
        return None

    def gelman_rubin(self):
        # takes current traces and returns
        if self.pool is None:
            raise RuntimeError("Gelman-Rubin statistic is only "
                               "valid for multiple chains.")

        if self.total_steps == 0:
            raise RuntimeError("Gelman-Rubin statistic not "
                               "defined for 0-length chains.")

        # gather trace statistics to master process
        means = self.trace_means()
        variances = self.trace_variances()

        if self.pool.is_master():
            B_over_n = np.var(means, ddof=1, axis=1)
            B = B_over_n * self.total_steps
            W = np.mean(variances, axis=1)
            V = ((1. - 1./self.total_steps) * W +
                 (1. + 1./self.pool.size) * B_over_n)
            # TODO: check for 0-values in W
            Rhat = np.sqrt(V/W)
        else:
            Rhat = None

        Rhat = self.pool.bcast(Rhat)
        return Rhat
