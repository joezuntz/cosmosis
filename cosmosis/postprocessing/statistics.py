#coding: utf-8
from .elements import PostProcessorElement
import numpy as np


class Statistics(PostProcessorElement):
    def run(self):
        print "I do not know how to generate statistics for this kind of data"
        return []

    def filename(self, base, ftype='txt'):
        output_dir = self.options.get("outdir", "png")
        prefix=self.options.get("prefix","")
        return "{0}/{1}{2}.{3}".format(output_dir, prefix, base, ftype)


class ConstrainingStatistics(Statistics):

    def report_file(self):
        #Get the filenames to make
        marge_filename = self.filename("means")
        best_filename = self.filename("best_fit")
        median_filename = self.filename("medians")

        #Generate the means file
        marge_file = open(marge_filename, "w")
        marge_file.write("#parameter mean std_dev\n")
        for P in zip(self.source.colnames, self.mu, self.sigma):
            marge_file.write("%s   %e   %e\n" % P)
        marge_file.close()

        #Generate the medians file
        median_file = open(median_filename, "w")
        median_file.write("#parameter mean std_dev\n")
        for P in zip(self.source.colnames, self.median, self.sigma):
            median_file.write("%s   %e   %e\n" % P)
        median_file.close()

        #Generate the mode file
        best_file = open(best_filename, "w")
        best_file.write("#parameter value\n")
        for P in zip(self.source.colnames, self.source.get_row(self.best_fit_index)):
            best_file.write("%s        %g\n"%P)
        best_file.close()
        return [marge_filename, best_filename, median_filename]

    @staticmethod
    def find_median(x, P):
        C = [0] + P.cumsum()
        return np.interp(C[-1]/2.0,C,x)

    def report_screen(self):
        #Print the same summary stats that go into the
        #files but to the screen instead, in a pretty format        

        #Means
        print
        print "Marginalized mean, std-dev:"
        for P in zip(self.source.colnames, self.mu, self.sigma):
            print '    %s = %g ± %g' % P
        print
        #Medians
        print "Marginalized median, std-dev:"
        for P in zip(self.source.colnames, self.median, self.sigma):
            print '    %s = %g ± %g' % P
        print

        #Mode
        print "Best likelihood:"
        for name, val in zip(self.source.colnames, self.source.get_row(self.best_fit_index)):
            print '    %s = %g' % (name, val)
        print

    @staticmethod
    def likelihood_ratio_warning(marge_like, name):
        #Check for an warn about a bad likelihood ratio,
        #which would indicate that the likelihood did not fall
        #off by the edges        
        if marge_like.min()==0: return
        like_ratio = marge_like.max() / marge_like.min()
        if like_ratio < 20:
            print
            print "L_max/L_min = %f for %s." % (like_ratio, name)
            print "This indicates that the grid did not go far from the peak in this dimension"
            print "Marginalized values will definitely be poor estimates for this parameter, and probably"
            print "for any other parameters correlated with this one"


class MetropolisHastingsStatistics(ConstrainingStatistics):
    def compute_basic_stats(self):
        burn = self.options.get("burn", 0)
        thin = self.options.get("thin", 1)
        self.mu = []
        self.sigma = []
        self.median = []
        self.best_fit_index = self.source.get_col("like").argmax()
        n = 0
        for col in self.source.colnames:
            data = self.source.get_col(col)[burn::thin]
            n = len(data)
            self.mu.append(data.mean())
            self.sigma.append(data.std())
            self.median.append(np.median(data))
        return n

    def run(self):
        N = self.compute_basic_stats()
        print "Post-burn & thin length:", N

        self.report_screen()
        files = self.report_file()
        return files


class GridStatistics(ConstrainingStatistics):
    def set_data(self):
        self.nsample = self.source.ini.getint("grid", "nsample_dimension")
        self.nrow = len(self.source)
        self.ncol = len(self.source.colnames)

        extra = self.source.ini.get("pipeline", "extra_output","").split()
        self.grid_columns = [i for i in xrange(self.ncol) if not self.source.colnames[i] in extra and self.source.colnames[i]!="like"]
        self.ndim = len(self.grid_columns)

        assert self.nrow == self.nsample**self.ndim
        self.shape = np.repeat(self.nsample, self.ndim)
        self.like = np.exp(self.source.get_col("like")).reshape(self.shape)
        grid_names = [self.source.colnames[i] for i in xrange(self.ncol) if i in self.grid_columns]
        self.grid = [np.unique(self.source.get_col(name)) for name in grid_names]

    def run(self):
        self.set_data()
        self.compute_stats()
        self.report_screen()
        files = self.report_file()
        return files

    def compute_stats(self):
        #1D stats
        self.mu = np.zeros(self.ncol-1)
        self.median = np.zeros(self.ncol-1)
        self.sigma = np.zeros(self.ncol-1)
        like = self.source.get_col("like")
        self.best_fit_index = np.argmax(like)
        #Loop through colums
        for i, name in enumerate(self.source.colnames[:-1]):
            if i in self.grid_columns:
                self.mu[i], self.median[i], self.sigma[i] = self.compute_grid_stats(i)
            else:
                self.mu[i], self.median[i], self.sigma[i] = self.compute_derived_stats(i)




    def compute_grid_stats(self, i):
        name = self.source.colnames[i]
        col = self.source.get_col(name)

        #Sum the likelihood over all the axes other than this one
        #to get the marginalized likelihood
        marge_like = self.like.sum(tuple(j for j in xrange(self.ndim) if j!=i))
        marge_like = marge_like / marge_like.sum()
        
        #Find the grid points with this value
        vals = self.grid[i]

        #A quick potential error warning
        self.likelihood_ratio_warning(marge_like, name)

        #Compute the statistics
        mu = (vals*marge_like).sum()
        sigma2 = ((vals-mu)**2*marge_like).sum()
        median = self.find_median(vals, marge_like)
        return mu, median, sigma2**0.5

    def compute_derived_stats(self, i):
        #This is a bit simpler - just need to 
        #sum over everything
        name = self.source.colnames[i]
        col = self.source.get_col(name)
        like = self.source.get_col("like")
        like = like / like.sum()
        mu = (col*like).sum()
        sigma2 = ((col-mu)**2*like).sum()
        return mu, sigma2**0.5



class TestStatistics(Statistics):
    def run(self):
        return []

class MultinestStatistics(Statistics):
    def run(self):
        return []

