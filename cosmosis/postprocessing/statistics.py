#coding: utf-8
from .elements import PostProcessorElement, MCMCPostProcessorElement, MultinestPostProcessorElement
import numpy as np
from .utils import std_weight, mean_weight, median_weight


class Statistics(PostProcessorElement):
    def run(self):
        print "I do not know how to generate statistics for this kind of data"
        return []

    def filename(self, base, ftype='txt'):
        output_dir = self.options.get("outdir", "png")
        prefix=self.options.get("prefix","")
        if prefix: prefix+="_"        
        return "{0}/{1}{2}.{3}".format(output_dir, prefix, base, ftype)


class ConstrainingStatistics(Statistics):

    def report_file(self):
        #Get the filenames to make
        return [
            self.report_file_mean(),
            self.report_file_median(),
            self.report_file_mode()
        ]
    def report_file_mean(self):        
        #Generate the means file
        marge_filename = self.filename("means")
        marge_file = open(marge_filename, "w")
        marge_file.write("#parameter mean std_dev\n")
        for P in zip(self.source.colnames, self.mu, self.sigma):
            marge_file.write("%s   %e   %e\n" % P)
        marge_file.close()
        return marge_filename

    def report_file_median(self):
        #Generate the medians file
        median_filename = self.filename("medians")
        median_file = open(median_filename, "w")
        median_file.write("#parameter mean std_dev\n")
        for P in zip(self.source.colnames, self.median, self.sigma):
            median_file.write("%s   %e   %e\n" % P)
        median_file.close()
        return median_filename

    def report_file_mode(self):
        #Generate the mode file
        best_filename = self.filename("best_fit")
        best_file = open(best_filename, "w")
        best_file.write("#parameter value\n")
        for P in zip(self.source.colnames, self.source.get_row(self.best_fit_index)):
            best_file.write("%s        %g\n"%P)
        best_file.close()
        return best_filename

    @staticmethod
    def find_median(x, P):
        C = [0] + P.cumsum()
        return np.interp(C[-1]/2.0,C,x)

    def report_screen(self):
        self.report_screen_mean()
        self.report_screen_median()
        self.report_screen_mode()
        #Print the same summary stats that go into the
        #files but to the screen instead, in a pretty format        
    def report_screen_mean(self):
        #Means
        print
        print "Marginalized mean, std-dev:"
        for P in zip(self.source.colnames, self.mu, self.sigma):
            print '    %s = %g ± %g' % P
        print
    def report_screen_median(self):
        #Medians
        print "Marginalized median, std-dev:"
        for P in zip(self.source.colnames, self.median, self.sigma):
            print '    %s = %g ± %g' % P
        print
    def report_screen_mode(self):
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


class MetropolisHastingsStatistics(ConstrainingStatistics, MCMCPostProcessorElement):
    def compute_basic_stats_col(self, col):
        data = self.reduced_col(col)
        n = len(data)
        return n, data.mean(), data.std(), np.median(data)

    def compute_basic_stats(self):
        self.mu = []
        self.sigma = []
        self.median = []
        self.best_fit_index = self.source.get_col("like").argmax()
        n = 0
        for col in self.source.colnames:
            n, mu, sigma, median = self.compute_basic_stats_col(col)
            self.mu.append(mu)
            self.sigma.append(sigma)
            self.median.append(median)
        return n

    def run(self):
        N = self.compute_basic_stats()
        print "Samples after cutting:", N

        self.report_screen()
        files = self.report_file()
        return files


class GridStatistics(ConstrainingStatistics):
    def set_data(self):
        self.nsample = int(self.source.ini.get("grid", "nsample_dimension"))
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

class MultinestStatistics(MultinestPostProcessorElement, MetropolisHastingsStatistics):
    def compute_basic_stats_col(self, col):
        data = self.reduced_col(col)
        weight = self.weight_col()
        n = len(data)
        return n, mean_weight(data,weight), std_weight(data,weight), median_weight(data, weight)

    def run(self):
        # Use parent statistics, except add evidence information,
        # which is just read from the file
        files = super(MultinestStatistics,self).run()
        logz = self.source.final_metadata[0]["log_z"]
        logz_sigma = self.source.final_metadata[0]["log_z_error"]
        
        #First print to screen
        print "Bayesian evidence:"
        print "    log(Z) = %g ± %g" % (logz,logz_sigma)
        print

        #Now save to file
        filename = self.filename("evidence")
        f = open(filename,'w')
        f.write('#logz    logz_sigma\n')
        f.write('%e    %e\n'%(logz,logz_sigma))
        f.close()

        #Include evidence in list of created files
        files.append(filename)
        return files
