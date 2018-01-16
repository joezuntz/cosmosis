#coding: utf-8
from __future__ import print_function
from builtins import zip
from builtins import object
from cosmosis import output as output_module

import numpy as np
import sys
import os


class Analytics(object):
    def __init__(self, params, pool=None):
        self.params = params
        self.pool = pool

        self.total_steps = 0
        nparam = len(params)
        self.means = np.zeros(nparam)
        self.m2 = np.zeros(nparam)
        self.cov_times_n = np.zeros((nparam,nparam))

    def add_traces(self, traces):
        if traces.shape[1] != len(self.params):
            raise RuntimeError("The number of traces added to Analytics "
                               "does not match the number of varied "
                               "parameters!")
        num = float(self.total_steps)
        for x in traces:
            num += 1.0
            delta = x - self.means
            old_means = self.means.copy()
            self.means += delta/num
            self.m2 += delta*(x - self.means)
            self.cov_times_n += np.outer(x-self.means, x-old_means)

        self.total_steps += traces.shape[0]

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

    def gelman_rubin(self, quiet=True):
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

        if not quiet and self.pool.is_master():
            print()
            print("Gelman-Rubin:")
            for (p,R) in zip(self.params, Rhat):
                print("    ", p, "   ", R)
            print("Worst = ", Rhat.max())
            print()

        return Rhat
