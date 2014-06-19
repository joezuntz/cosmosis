#coding: utf-8
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
            old_means = self.means.copy()
            self.means += delta/num
            self.m2 += delta*(x - self.means)
            self.cov_times_n += np.outer(x-self.means, x-old_means)

        self.total_steps += traces.shape[0]

    @classmethod
    def from_ini(cls, ini, **kwargs):
        if isinstance(ini, basestring):
            options = {"format":"text", "filename":ini}
        else:
           options = dict(ini.items('output'))

        return cls.from_outputs(options, **kwargs)

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
        analytics.cov = analytics.cov_times_n / analytics.total_steps            
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
        analytics.cov = analytics.cov_times_n / analytics.total_steps
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
    def output_results(self):
        Mu =  self.trace_means()
        Sigma = self.trace_variances()**0.5

        #Lots of printout
        print
        print "Marginalized:"
        for p, mu, sigma in zip(self.params, Mu, Sigma):
            if p=="LIKE":continue
            print '    %s = %g ± %g' % (p, mu, sigma)
        print
        if self.best_index==None:
            print "Could not see LIKE column to get best likelihood"
        else:
            print "Best likelihood:"
            print "    Index = %d" % (self.best_index)
            for p, v in zip(self.params, self.best_params):
                print '    %s = %g' % (p, v)
        print
        import numpy as np
        print "Covariance matrix:"
        print '#' + ' '.join(self.params)
        np.savetxt(sys.stdout, self.cov)


class GridAnalytics(object):
    def __init__(self, column_names, data, grid_columns):
        self.column_names = column_names
        self.grid_columns = grid_columns
        self.data = data
        self.ndim = len(grid_columns)
        self.ncol = len(column_names)
        self.ntotal = len(data)
        self.nsample = int(self.ntotal**(1.0/self.ndim))
        self.like_col = column_names.index("like")
        self.like = data[:,self.like_col]
        self.compute_stats()

    def compute_stats(self):
        #1D stats
        self.mu = np.zeros(self.ndim)
        self.sigma = np.zeros(self.ndim)
        for i in self.grid_columns:
            name = self.column_names[i]
            vals = np.unique(self.data[:,i])
            lv = np.zeros(self.nsample)
            assert len(vals==self.nsample)
            for j,v in enumerate(vals):
                w = np.where(self.data[:,i]==v)
                lv[j] = np.exp(self.like[w]).sum()
            lv_sum = lv.sum()
            like_ratio = lv.max() / lv.min()
            if like_ratio < 20:
                print
                print "L_max/L_min = %f for %s." % (like_ratio, name)
                print "This indicates that the grid did not go far from the peak in this dimension"
                print "Marginalized values will definitely be poor estimates for this parameter, and probably"
                print "for any other parameters correlated with this one"
            mu = (vals*lv).sum() / lv_sum
            sigma2 = ((vals-mu)**2*lv).sum() / lv_sum
            self.mu[i] = mu
            self.sigma[i] = sigma2**0.5
        # Best-fit values
        best_fit_index = np.argmax(self.like)
        self.best_fit = self.data[best_fit_index]

    
    def output_results(self):
        print
        print "Marginalized mean, std-dev:"
        for i in self.grid_columns:
            p = self.column_names[i]
            mu = self.mu[i]
            sigma = self.sigma[i]
            print '    %s = %g ± %g' % (p, mu, sigma)
        print
        print "Best likelihood:"
        for name, val in zip(self.column_names, self.best_fit):
            print '    %s = %g' % (name, val)
            

    @classmethod
    def from_ini(cls, ini, **kwargs):
        options = dict(ini.items("output"))
        column_names, data, metadata, comments, final_metadata = output_module.input_from_options(options)
        column_names = [c.lower() for c in column_names]
        ns = ini.getint("grid", "nsample_dimension")
        nc = len(column_names)
        extra = ini.get("pipeline", "extra_output","").split()
        grid_columns = [i for i in xrange(nc) if not column_names[i] in extra and column_names[i]!="like"]
        return cls(column_names, data[0], grid_columns)

    def from_outputs(cls, options, burn=0, thin=1):
        column_names, data, metadata, comments, final_metadata = output_module.input_from_options(options)
        num_cols = len(column_names)
        column_names = [c.lower() for c in column_names]

        assert "like" in column_names
        analytics = cls(column_names, data)
        analytics.compute_stats()
        return analytics
