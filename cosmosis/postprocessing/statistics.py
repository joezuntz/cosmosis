#coding: utf-8
from .elements import PostProcessorElement
import numpy as np


class Statistics(PostProcessorElement):
    def run(self):
        print "I do not know how to generate statistics for this kind of data"
        return []


class MetropolisHastingsStatistics(Statistics):
    def compute_basic_stats(self):
        burn = self.options.get("burn", 0)
        thin = self.options.get("thin", 1)
        mu = []
        sigma = []
        median = []
        n = 0
        for col in self.source.colnames:
            data = self.source.get_col(col)[burn::thin]
            n = len(data)
            mu.append(data.mean())
            sigma.append(data.std())
            median.append(np.median(data))
        return n, mu, sigma, median

    def run(self):
        N, Mu, Sigma, Median = self.compute_basic_stats()
        print "Post-burn & thin length:", N
        #Lots of printout
        print
        print "Marginalized:"
        for p, mu, sigma in zip(self.source.colnames, Mu, Sigma):
            if p.lower()=="like": continue
            print '    %s = %g ± %g ' % (p, mu, sigma)
        print
        print "Medians:"
        for p, median in zip(self.source.colnames, Median):
            if p.lower()=="like": continue
            print '    %s = %g' % (p, median)
        print
        try:
            like = self.source.get_col("like")
            best_index = np.argmax(like)
            best_params = [self.source.get_col(name)[best_index] for name in self.source.colnames]
        except:
            best_index=None

        if best_index is None:
            print "Could not see LIKE column to get best likelihood"
        else:
            print "Best fit:"
            print "    Index = %d" % (best_index)
            for p, v in zip(self.source.colnames, best_params):
                print '    %s = %g' % (p, v)
        print
        return []

        # cov = np.cov([self.source.get_col(i) for i in xrange(len(self.))])
        # print "Covariance matrix:"
        # print '#' + ' '.join(self.params)
        # np.savetxt(sys.stdout, self.cov)


class GridStatistics(Statistics):
    def set_data(self):
        ns = self.source.ini.getint("grid", "nsample_dimension")
        nc = len(self.source.colnames)
        extra = self.source.ini.get("pipeline", "extra_output","").split()
        grid_columns = [i for i in xrange(nc) if not self.source.colnames[i] in extra and self.source.colnames[i]!="like"]
        data = self.source.data

        self.column_names = column_names
        self.grid_columns = grid_columns
        self.data = data
        self.ndim = len(grid_columns)
        self.ncol = len(column_names)
        self.ntotal = len(data)
        self.nsample = int(self.ntotal**(1.0/self.ndim))
        self.like_col = column_names.index("like")
        self.like = data[:,self.like_col]

    def run(self):
        self.set_data()
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

class TestStatistics(Statistics):
    def run(self):
        return []

class MultinestStatistics(Statistics):
    def run(self):
        return []

