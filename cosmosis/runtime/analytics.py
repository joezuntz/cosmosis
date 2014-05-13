from cosmosis import output as output_module

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

    def add_traces(self, traces, like=None):
        if traces.shape[1] != len(self.params):
            raise RuntimeError("The number of traces added to Analytics "
                               "does not match the number of varied "
                               "parameters!")

        if like is not None:
            maxlike_index = np.argmax(like)
            if like[maxlike_index] > self.best_like:
                self.best_like = like[maxlike_index]
                self.best_index = maxlike_index + self.total_steps
                self.best_params = traces[maxlike_index,:]

        num = float(self.total_steps)
        for x in traces:
            num += 1.0
            delta = x - self.means
            self.means += delta/num
            self.m2 += delta*(x - self.means)

        self.total_steps += traces.shape[0]

    @classmethod
    def from_outputs(cls, options, burn=0, thin=1):
        column_names, data, metadata, comments, final_metadata = output_module.input_from_options(options)

        num_cols = len(column_names)
        if "LIKE" in column_names:
            like_col = column_names.index("LIKE")
            param_cols = range(num_cols)
            del param_cols[like_col]
            del column_names[like_col]
        else:
            like_col = None

        analytics = cls(column_names)
        for chain in data:
            if chain.shape[1] != num_cols:
                raise RuntimeError("Incorrect number of columns in output "
                                   "(%d, expected %d)." %
                                   (chain.shape[1], num_cols))

            if burn < 1:
                nburn = chain.shape[0] * burn
            else:
                nburn = burn
            chain = chain[nburn::thin,:]

            if like_col:
                like = chain[:,like_col]
                chain = chain[:,param_cols]
            else:
                like = None
            analytics.add_traces(chain, like)
        return analytics 

    @classmethod
    def from_chain_files(cls, filenames, burn=0, thin=1):
        if isinstance(filenames, str):
            filenames = [filenames]

        #currently only works on text files.
        #TODO: rejig to use "load" methods on output objects
        params = open(filenames[0]).readline().strip('#').split()

        num_cols = len(params)
        if "LIKE" in params:
            like_col = params.index("LIKE")
            param_cols = range(num_cols)
            del param_cols[like_col]
            del params[like_col]
        else:
            like_col = None

        analytics = cls(params)
        for filename in filenames:
            chain = np.genfromtxt(filename)
            if chain.shape[1] != num_cols:
                raise RuntimeError("Incorrect number of columns in output "
                                   "file %s (%d, expected %d)." %
                                   (filename, chain.shape[1], num_cols))

            if burn < 1:
                nburn = len(chain) * burn
            else:
                nburn = burn
            chain = chain[nburn::thin,:]

            if like_col:
                like = chain[:,like_col]
                chain = chain[:,param_cols]
            else:
                like = None

            analytics.add_traces(chain, like)
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
        if self.pool is None or not self.pool.size > 1:
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
