#coding: utf-8
from .elements import PostProcessorElement, MCMCPostProcessorElement, WeightedMCMCPostProcessorElement, MultinestPostProcessorElement
import numpy as np
from .utils import std_weight, mean_weight, median_weight


class Statistics(PostProcessorElement):
    def __init__(self, *args, **kwargs):
        super(Statistics, self).__init__(*args, **kwargs)
        self.text_files = {}

    def run(self):
        print "I do not know how to generate statistics for this kind of data"
        return []

    def filename(self, base, ftype='txt'):
        output_dir = self.options.get("outdir", "./")
        prefix=self.options.get("prefix","")
        if prefix: prefix+="_"        
        return "{0}/{1}{2}.{3}".format(output_dir, prefix, base, ftype)

    def open_output(self, base, header="", section_name="", ftype='txt'):
        filename = self.filename(base, ftype)
        if filename in self.text_files:
            f = self.text_files[filename]
            new_file = False
        else:
            f = open(filename, 'w')
            self.text_files[filename] = f
            new_file = True
            if header:
                f.write(header+'\n')
        if section_name:
            f.write("#%s\n"%section_name)
        return f, filename, new_file

    def finalize(self):
        for f in self.text_files.values():
            f.close()


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
        header = "#parameter mean std_dev"
        marge_file, marge_filename, new_file = self.open_output("means", header, self.source.name)
        for P in zip(self.source.colnames, self.mu, self.sigma):
            marge_file.write("%s   %e   %e\n" % P)
        return marge_filename

    def report_file_median(self):
        #Generate the medians file
        header = "#parameter mean std_dev\n"
        median_file, median_filename, new_file = self.open_output("medians", header, self.source.name)
        for P in zip(self.source.colnames, self.median, self.sigma):
            median_file.write("%s   %e   %e\n" % P)
        return median_filename

    def report_file_mode(self):
        #Generate the mode file
        header = "#parameter value"
        best_file, best_filename, new_file = self.open_output("best_fit", header, self.source.name)
        for P in zip(self.source.colnames, self.source.get_row(self.best_fit_index)):
            best_file.write("%s        %g\n"%P)
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

class ChainCovariance(object):
    def run(self):
        #Determine the parameters to use

        col_names = [p for p in self.source.colnames if p.lower() not in ["like", "importance", "weight"]]

        if len(col_names)<2:
            return []


        cols = [self.reduced_col(p) for p in col_names]
        covmat = np.cov(cols)

        #For the proposal we just want the first 
        #nvaried rows/cols - we don't want to include
        #extra parameters like sigma8
        n = self.source.metadata[0]['n_varied']
        proposal = covmat[:n,:n]

        #Save the covariance matrix
        filename = self.filename("covmat")
        f, filename, new_file = self.open_output("covmat")
        if new_file:
            f.write('#'+'    '.join(col_names)+'\n')
            np.savetxt(f, covmat)
        else:
            print "NOT saving more than covariance matrix - just using first ini file"

        #Save the proposal matrix
        f, proposal_filename, new_file = self.open_output("proposal")
        if new_file:
            f.write('#'+'    '.join(col_names[:n])+'\n')
            np.savetxt(f, proposal)
        else:
            print "NOT saving more than proposal matrix - just using first ini file"
        return [filename, proposal_filename]

class MetropolisHastingsCovariance(ChainCovariance, Statistics, MCMCPostProcessorElement):
    pass



class GridStatistics(ConstrainingStatistics):
    def set_data(self):
        self.nsample = int(self.source.ini.get("grid", "nsample_dimension"))
        self.nrow = len(self.source)
        self.ncol = len(self.source.colnames)

        extra = self.source.ini.get("pipeline", "extra_output","").replace('/','--').split()
        self.grid_columns = [i for i in xrange(self.ncol) if (not self.source.colnames[i] in extra) and (self.source.colnames[i]!="like")]
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
        median = self.find_median(col, like)
        return mu, median, sigma2**0.5



class TestStatistics(Statistics):
    def run(self):
        return []

class DunkleyTest(MetropolisHastingsStatistics):
    """
    Run the Dunley et al (2005) power spectrum test
    for MCMC convergence.  This is, loosely speaking,
    the Fourier domain version of an auto-correlation
    length check.

    """
    #This is the value recommended in Dunkely et al.
    #You can probably get away with less than this
    jstar_convergence_limit = 20.0

    def run(self):
        n = self.source.metadata[0]['n_varied']
        jstar = []
        #Just sample over the sample parameters,
        #not the likelihood or the derived parameters
        params = self.source.colnames[:n]
        print "Dunkely et al (2005) power spectrum test."
        print "For converged chains j* > %.1f:" %self.jstar_convergence_limit
        for param in params:
            cols = self.reduced_col(param, stacked=False)
            for c,col in enumerate(cols):
                js = self.compute_jstar(col)
                jstar.append(js)
                m = "Chain %d:  %-35s j* = %-.1f" % (c+1, param, js)
                if js>20:
                    print "    %-50s" % m
                else:
                    print "    %-50s NOT CONVERGED!" % m
        print
        if not np.min(jstar)>self.jstar_convergence_limit:
            print "The Dunkley et al (2005) power spectrum test shows that this chain has NOT CONVERGED."
            print "It is quite a conservative test, so no need to panic."
        else:
            print "The power spectra for this chain suggests good convergence."
        print
        header = '#'+'    '.join(params)
        f, filename, new_file = self.open_output("dunkley", header, self.source.name)
        f.write("\n")
        f.write('    '.join(str(js) for js in jstar))
        f.write("\n")
        return [filename]


    @staticmethod
    def compute_jstar(x):
        import scipy.optimize

        #Get the power spectrum of the chain
        n=len(x)
        p = abs(np.fft.rfft(x)[1:(n/2+1)])**2
        #And the k-axis
        j = np.arange(p.size)+1.
        k = j / (2*np.pi*n)
        #fitting is done on the log of the power 
        #spectrum
        logp = np.log(p)

        #The model for the power spectrum.
        #See Dunkley et al for info on the 
        #constant
        def template(k, L0, a, kstar):
            K = (kstar / k)**a
            euler_mascheroni = 0.5772156649015
            return L0 + np.log(K/(1.+K)) - euler_mascheroni

        #Starting guess values for parameters.
        #These are usually fine.
        L0 = logp[0:10].mean()
        a  = 1.0
        kstar = 0.1/n

        #Fit curve with LevMar least-squares
        start_params = [L0, a, kstar]
        try:
            params, cov = scipy.optimize.curve_fit(template, k, logp, start_params)
        except RuntimeError:
            #Runtime Errors signal that the fitting process has failed.
            #This usually happens because the chain is too short,
            #or has periodic or other features in like an emcee chain
            params = [np.nan, np.nan, np.nan]
        L0, a, kstar = params
        return kstar * x.size * 2 * np.pi


class WeightedStatistics(object):
    def compute_basic_stats_col(self, col):
        data = self.reduced_col(col)
        weight = self.weight_col()
        n = len(data)
        return n, mean_weight(data,weight), std_weight(data,weight), median_weight(data, weight)

class MultinestStatistics(WeightedStatistics, MultinestPostProcessorElement, MetropolisHastingsStatistics):
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
        header = '#logz    logz_sigma'
        f, filename, new_file  = self.open_output("evidence", header, self.source.name)
        f.write('%e    %e\n'%(logz,logz_sigma))

        #Include evidence in list of created files
        files.append(filename)
        return files

#The class hierarchy is getting too complex for this - revise it
class WeightedMetropolisStatistics(WeightedStatistics, ConstrainingStatistics, WeightedMCMCPostProcessorElement):
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
class MultinestCovariance(ChainCovariance, Statistics, MultinestPostProcessorElement):
    pass


class Citations(Statistics):
    #This isn't really a statistic but it uses all the same
    #mechanisms
    def run(self):
        print 
        message = "#You should cite these papers in any publication based on this pipeline."
        print message
        f, filename, new_file = self.open_output("citations", message, self.source.name)
        for comment_set in self.source.comments:
            for comment in comment_set:
                comment = comment.strip()
                if comment.startswith("CITE"):
                    citation =comment[4:].strip()
                    print "    ", citation
                    f.write("%s\n"%citation)
        print
        return [filename]