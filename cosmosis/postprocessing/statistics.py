#coding: utf-8
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from .elements import PostProcessorElement, MCMCPostProcessorElement, WeightedMCMCPostProcessorElement, MultinestPostProcessorElement
import numpy as np
from .utils import std_weight, mean_weight, median_weight, percentile_weight, find_asymmetric_errorbars
from .outputs import PostprocessText

class Statistics(PostProcessorElement):
    def __init__(self, *args, **kwargs):
        super(Statistics, self).__init__(*args, **kwargs)

    def run(self):
        print("I do not know how to generate statistics for this kind of data")
        return []
 
    def filename(self, base):
        filename = super(Statistics, self).filename("txt", base)
        return filename


    def get_text_output(self, base, header="", section_name=""):
        filename = self.filename(base)
        f = self.get_output(filename)
        if f is None:
            f = open(filename, 'w')
            self.set_output(filename, PostprocessText(base,filename,f))
            new_file = True
            if header:
                f.write(header+'\n')
        else:
            f=f.value
            new_file = False
        if section_name:
            f.write("#%s\n"%section_name)
        return f, filename, new_file

class ConstrainingStatistics(Statistics):

    def report_file(self):
        #Get the filenames to make
        return [
            self.report_file_mean(),
            self.report_file_median(),
            self.report_file_mode(),
            self.report_file_l95(),
            self.report_file_u95(),
            self.report_file_l68(),
            self.report_file_u68(),
            self.report_file_err("lerr68", self.lerr68),
            self.report_file_err("uerr68", self.uerr68),
            self.report_file_err("lerr95", self.lerr95),
            self.report_file_err("uerr95", self.uerr95),
            self.report_file_err("peak1d", self.peak1d),
        ]
    def report_file_mean(self):        
        #Generate the means file
        header = "#parameter mean std_dev"
        marge_file, marge_filename, new_file = self.get_text_output("means", header, self.source.name)
        for P in zip(self.source.colnames, self.mu, self.sigma):
            marge_file.write("%s   %e   %e\n" % P)
        return marge_filename

    def report_file_median(self):
        #Generate the medians file
        header = "#parameter mean std_dev"
        median_file, median_filename, new_file = self.get_text_output("medians", header, self.source.name)
        for P in zip(self.source.colnames, self.median, self.sigma):
            median_file.write("%s   %e   %e\n" % P)
        return median_filename

    def report_file_mode(self):
        #Generate the mode file
        header = "#parameter value"
        best_file, best_filename, new_file = self.get_text_output("best_fit", header, self.source.name)
        for P in zip(self.source.colnames, self.source.get_row(self.best_fit_index)):
            best_file.write("%s        %g\n"%P)
        return best_filename

    def report_file_l95(self):
        #Generate the medians file
        header = "#parameter low95"
        limit_file, limit_filename, new_file = self.get_text_output("low95", header, self.source.name)
        for P in zip(self.source.colnames, self.l95):
            limit_file.write("%s     %g\n" % P)
        return limit_filename

    def report_file_u95(self):
        #Generate the medians file
        header = "#parameter upper95"
        limit_file, limit_filename, new_file = self.get_text_output("upper95", header, self.source.name)
        for P in zip(self.source.colnames, self.u95):
            limit_file.write("%s     %g\n" % P)
        return limit_filename

    def report_file_l68(self):
        #Generate the medians file
        header = "#parameter low68"
        limit_file, limit_filename, new_file = self.get_text_output("low68", header, self.source.name)
        for P in zip(self.source.colnames, self.l68):
            limit_file.write("%s     %g\n" % P)
        return limit_filename

    def report_file_u68(self):
        #Generate the medians file
        header = "#parameter upper68"
        limit_file, limit_filename, new_file = self.get_text_output("upper68", header, self.source.name)
        for P in zip(self.source.colnames, self.u68):
            limit_file.write("%s     %g\n" % P)
        return limit_filename

    def report_file_err(self, name, data):
        #Generate the medians file
        header = "#parameter {}".format(name)
        limit_file, limit_filename, new_file = self.get_text_output(name, header, self.source.name)
        for P in zip(self.source.colnames, data):
            limit_file.write("%s     %g\n" % P)
        return limit_filename


    @staticmethod
    def find_median(x, P):
        C = [0] + P.cumsum()
        return np.interp(C[-1]/2.0,C,x)

    @staticmethod
    def find_percentile(x, P, p):
        C = [0] + P.cumsum()
        return np.interp(C[-1]*p/100.0,C,x)

    def report_screen(self):
        self.report_screen_mean()
        self.report_screen_asym()
        self.report_screen_asym95()
        self.report_screen_median()
        self.report_screen_mode()
        self.report_screen_limits()
        #Print the same summary stats that go into the
        #files but to the screen instead, in a pretty format        
    def report_screen_mean(self):
        #Means
        print()
        print("Marginalized mean, std-dev:")
        for P in zip(self.source.colnames, self.mu, self.sigma):
            print('    %s = %g ± %g ' % P)
        print()

    def report_screen_asym(self):
        #Means
        print()
        print("Marginalized 1D peak, 68% asymmetric error bars:")
        for P in zip(self.source.colnames, self.peak1d, self.lerr68, self.uerr68):
            name,mu,lerr,uerr = P
            print('    %s = %g + %g - %g ' % (name,mu,uerr-mu, mu-lerr))
        print()

    def report_screen_asym95(self):
        #Means
        print()
        print("Marginalized 1D peak, 95% asymmetric error bars:")
        for P in zip(self.source.colnames, self.peak1d, self.lerr95, self.uerr95):
            name,mu,lerr,uerr = P
            print('    %s = %g + %g - %g ' % (name,mu,uerr-mu, mu-lerr))
        print()

    def report_screen_median(self):
        #Medians
        print("Marginalized median, std-dev:")
        for P in zip(self.source.colnames, self.median, self.sigma):
            print('    %s = %g ± %g' % P)
        print()
    def report_screen_mode(self):
        #Mode
        print("Best likelihood:")
        for name, val in zip(self.source.colnames, self.source.get_row(self.best_fit_index)):
            print('    %s = %g' % (name, val))
        print()

    def report_screen_limits(self):
        #Mode
        print("95% lower limits:")
        for name, val in zip(self.source.colnames, self.l95):
            print('    %s > %g' % (name, val))
        print()
        print("95% upper limits:")
        for name, val in zip(self.source.colnames, self.u95):
            print('    %s < %g' % (name, val))
        print()
        #Mode
        print("68% lower limits:")
        for name, val in zip(self.source.colnames, self.l68):
            print('    %s > %g' % (name, val))
        print()
        print("68% upper limits:")
        for name, val in zip(self.source.colnames, self.u68):
            print('    %s < %g' % (name, val))
        print()

    @staticmethod
    def likelihood_ratio_warning(marge_like, name):
        #Check for an warn about a bad likelihood ratio,
        #which would indicate that the likelihood did not fall
        #off by the edges        
        if marge_like.min()==0: return

        like_ratio = marge_like.max() / marge_like.min()
        if like_ratio < 20:
            print()
            print("L_max/L_min = %f for %s." % (like_ratio, name))
            print("This indicates that the grid did not go far from the peak in this dimension")
            print("Marginalized values will definitely be poor estimates for this parameter, and probably")
            print("for any other parameters correlated with this one")


class MetropolisHastingsStatistics(ConstrainingStatistics, MCMCPostProcessorElement):
    def compute_basic_stats_col(self, col):
        data = self.reduced_col(col)
        n = len(data)
        try:
            peak1d, ((lerr68, uerr68), (lerr95, uerr95)) = find_asymmetric_errorbars([0.68, 0.95], data)
        except ValueError:
            (lerr68, uerr68), (lerr95, uerr95) = (np.nan, np.nan), (np.nan, np.nan)
            peak1d = np.nan
        return n, data.mean(), data.std(), np.median(data), np.percentile(data, 32.), np.percentile(data, 68.), np.percentile(data, 5.), np.percentile(data, 95.), lerr68, uerr68, lerr95, uerr95, peak1d

    def compute_basic_stats(self):
        self.mu = []
        self.sigma = []
        self.median = []
        self.l68 = []
        self.u68 = []
        self.l95 = []
        self.u95 = []
        self.lerr68 = []
        self.uerr68 = []
        self.lerr95 = []
        self.uerr95 = []
        self.peak1d = []
        try:self.best_fit_index = self.source.get_col("post").argmax()
        except:self.best_fit_index = self.source.get_col("like").argmax()
        
        n = 0
        for col in self.source.colnames:
            n, mu, sigma, median, l68, u68, l95, u95, lerr68, uerr68, lerr95, uerr95, peak1d = self.compute_basic_stats_col(col)
            self.mu.append(mu)
            self.sigma.append(sigma)
            self.median.append(median)
            self.l68.append(l68)
            self.u68.append(u68)
            self.l95.append(l95)
            self.u95.append(u95)
            self.lerr68.append(lerr68)
            self.uerr68.append(uerr68)
            self.lerr95.append(lerr95)
            self.uerr95.append(uerr95)
            self.peak1d.append(peak1d)
        return n

    def run(self):
        N = self.compute_basic_stats()
        print("Samples after cutting:", N)

        self.report_screen()
        files = self.report_file()
        return files

class ChainCovariance(object):
    def run(self):
        #Determine the parameters to use

        col_names = [p for p in self.source.colnames if p.lower() not in ["like","post", "importance", "weight"]]

        if len(col_names)<2:
            return []

        ps = self.posterior_sample()
        cols = [self.reduced_col(p)[ps] for p in col_names]
        covmat = np.cov(cols)

        #For the proposal we just want the first 
        #nvaried rows/cols - we don't want to include
        #extra parameters like sigma8
        n = self.source.metadata[0]['n_varied']
        proposal = covmat[:n,:n]

        #Save the covariance matrix
        f, filename, new_file = self.get_text_output("covmat")
        if new_file:
            f.write('#'+'    '.join(col_names)+'\n')
            np.savetxt(f, covmat)
        else:
            print("NOT saving more than one covariance matrix - just using first ini file")

        #Save the proposal matrix
        f, proposal_filename, new_file = self.get_text_output("proposal")
        if new_file:
            f.write('#'+'    '.join(col_names[:n])+'\n')
            np.savetxt(f, proposal)
        else:
            print("NOT saving more than one proposal matrix - just using first ini file")
        return [filename, proposal_filename]

class MetropolisHastingsCovariance(ChainCovariance, Statistics, MCMCPostProcessorElement):
    pass



class GridStatistics(ConstrainingStatistics):
    def set_data(self):
        self.nsample = int(self.source.sampler_option("nsample_dimension"))
        self.nrow = len(self.source)
        self.ncol = int(self.source.sampler_option('n_varied'))

        extra = self.source.sampler_option("extra_output","").replace('/','--').split()
        self.grid_columns = [i for i in range(self.ncol) if (not self.source.colnames[i] in extra) and (self.source.colnames[i]!="post") and (self.source.colnames[i]!="like")]
        self.ndim = len(self.grid_columns)
        assert self.nrow == self.nsample**self.ndim
        self.shape = np.repeat(self.nsample, self.ndim)

        try:
            like = self.source.get_col("post").reshape(self.shape).copy()
        except:
            like = self.source.get_col("like").reshape(self.shape).copy()

        like -= like.max()

        self.like = np.exp(like).reshape(self.shape)

        grid_names = [self.source.colnames[i] for i in range(self.ncol) if i in self.grid_columns]
        self.grid = [np.unique(self.source.get_col(name)) for name in grid_names]

    def run(self):
        self.set_data()
        self.compute_stats()
        self.report_screen()
        files = self.report_file()
        return files

    def report_screen(self):
        self.report_screen_mean()
        #self.report_screen_asym()
        # self.report_screen_asym95()
        self.report_screen_median()
        self.report_screen_mode()
        #self.report_screen_limits()

    def report_file(self):
        #Get the filenames to make
        return [
            self.report_file_mean(),
            self.report_file_median(),
            self.report_file_mode(),
            self.report_file_l95(),
            self.report_file_u95(),
#            self.report_file_l68(),
#            self.report_file_u68(),
            ]

    def compute_stats(self):
        #1D stats
        self.mu = np.zeros(self.ncol)
        self.median = np.zeros(self.ncol)
        self.sigma = np.zeros(self.ncol)
        self.l95 = np.zeros(self.ncol)
        self.u95 = np.zeros(self.ncol)        
        try:like = self.source.get_col("post")
        except:like = self.source.get_col("like")
        self.best_fit_index = np.argmax(like)
        #Loop through colums
        for i, name in enumerate(self.source.colnames[:self.ncol]):
            if i in self.grid_columns:
                self.mu[i], self.median[i], self.sigma[i], self.l95[i], self.u95[i] = self.compute_grid_stats(i)
            else:
                self.mu[i], self.median[i], self.sigma[i], self.l95[i], self.u95[i] = self.compute_derived_stats(i)




    def compute_grid_stats(self, i):
        name = self.source.colnames[i]
        col = self.source.get_col(name)
        #Sum the likelihood over all the axes other than this one
        #to get the marginalized likelihood
        marge_like = self.like.sum(tuple(j for j in range(self.ndim) if j!=i))
        marge_like = marge_like / marge_like.sum()
        
        #Find the grid points with this value
        vals = self.grid[i]

        #A quick potential error warning
        self.likelihood_ratio_warning(marge_like, name)

        #Compute the statistics
        mu = (vals*marge_like).sum()
        sigma2 = ((vals-mu)**2*marge_like).sum()
        median = self.find_median(vals, marge_like)
        l95 = self.find_percentile(vals, marge_like, 5.0)
        u95 = self.find_percentile(vals, marge_like, 95.0)
        return mu, median, sigma2**0.5, l95, u95

    def compute_derived_stats(self, i):
        #This is a bit simpler - just need to 
        #sum over everything
        name = self.source.colnames[i]
        col = self.source.get_col(name)
        try:like = self.source.get_col("post")
        except:like = self.source.get_col("like")
        like = like / like.sum()
        mu = (col*like).sum()
        sigma2 = ((col-mu)**2*like).sum()
        median = self.find_median(col, like)
        l95 = self.find_percentile(col, like, 5.0)
        u95 = self.find_percentile(col, like, 95.0)        
        return mu, median, sigma2**0.5, l95, u95



class TestStatistics(Statistics):
    def run(self):
        return []


class GelmanRubinStatistic(MetropolisHastingsStatistics):
    def gelman_rubin(self, name):
        # This simplified form compared to the online analytics code:
        # - assumes the chains are fairly long
        # - does one parameter at a time

        # Get the chains for each input file
        chains = self.source.reduced_col(name,stacked=False)

        steps = min([len(chain) for chain in chains])
        chains = [chain[:steps] for chain in chains]
        means = [chain.mean() for chain in chains]
        variances = [chain.var() for chain in chains]

        number_chains = len(chains)

        B_over_n = np.var(means, ddof=1)
        B = B_over_n * steps
        W = np.mean(variances)
        V = W + (1. + 1./number_chains) * B_over_n
        # TODO: check for 0-values in W
        Rhat = np.sqrt(V/W)
        return Rhat - 1.0

    def run(self):
        if len(self.source.data)<2:
            print()
            print("(One chain found. Run multiple chains if you want the Gelman-Rubin test)")
            print()
            return []
        names = [c for c in self.source.colnames if  c not in ['weight', 'like', 'post']]
        header = "#parameter   R-1\n"
        f, filename, is_new = self.get_text_output("gelman", header)
        print()
        print("Gelman-Rubin tests")
        print("------------------")
        print("(Variance of means / Mean of variances.  Smaller is better, a few percent is usually good convergence)")
        print()
        for name in names:
            R1 = self.gelman_rubin(name)
            f.write("{}   {}\n".format(name,R1))
            if R1>0.1:
                print("{}    {}  -- POORLY CONVERGED PARAMETER AT 10% LEVEL".format(name,R1))
            else:
                print("{}    {}".format(name,R1))
        print()
        return [filename]




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
        print("Dunkely et al (2005) power spectrum test.")
        print("For converged chains j* > %.1f:" %self.jstar_convergence_limit)
        for param in params:
            cols = self.reduced_col(param, stacked=False)
            for c,col in enumerate(cols):
                js = self.compute_jstar(col)
                jstar.append(js)
                m = "Chain %d:  %-35s j* = %-.1f" % (c+1, param, js)
                if js>20:
                    print("    %-50s" % m)
                else:
                    print("    %-50s NOT CONVERGED!" % m)
        print()
        if not np.min(jstar)>self.jstar_convergence_limit:
            print("The Dunkley et al (2005) power spectrum test shows that this chain has NOT CONVERGED.")
            print("It is quite a conservative test, so no need to panic.")
        else:
            print("The power spectra for this chain suggests good convergence.")
        print()
        header = '#'+'    '.join(params)
        f, filename, new_file = self.get_text_output("dunkley", header, self.source.name)
        f.write("\n")
        f.write('    '.join(str(js) for js in jstar))
        f.write("\n")
        return [filename]


    @staticmethod
    def compute_jstar(x):
        import scipy.optimize

        #Get the power spectrum of the chain
        n=len(x)
        p = abs(np.fft.rfft(x)[1:n/2])**2
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
        try:
            print(col)
            peak1d, ((lerr68, uerr68), (lerr95, uerr95)) = find_asymmetric_errorbars([0.68, 0.95], data, weight)
        except RuntimeError:
            (lerr68, uerr68), (lerr95, uerr95) = (np.nan, np.nan), (np.nan, np.nan)
            peak1d = np.nan

        return (n, mean_weight(data,weight), std_weight(data,weight), 
            median_weight(data, weight), percentile_weight(data, weight, 32.), percentile_weight(data, weight, 68.),
            percentile_weight(data, weight, 5.), percentile_weight(data, weight, 95.), lerr68, uerr68, lerr95, uerr95, peak1d)

class MultinestStatistics(WeightedStatistics, MultinestPostProcessorElement, MetropolisHastingsStatistics):
    def run(self):
        # Use parent statistics, except add evidence information,
        # which is just read from the file
        files = super(MultinestStatistics,self).run()
        logz = self.source.final_metadata[0]["log_z"]
        logz_sigma = self.source.final_metadata[0]["log_z_error"]
        
        #First print to screen
        print("Bayesian evidence:")
        print("    log(Z) = %g ± %g" % (logz,logz_sigma))
        print()


        weight = self.weight_col()
        w = weight/weight.max()
        n_eff = w.sum()

        print("Effective number samples = ", n_eff)
        print()
        #Now save to file
        header = '#logz    logz_sigma'
        f, filename, new_file  = self.get_text_output("evidence", header, self.source.name)
        f.write('%e    %e\n'%(logz,logz_sigma))

        #Include evidence in list of created files
        files.append(filename)
        return files

class PolychordStatistics(MultinestStatistics):
    pass

#The class hierarchy is getting too complex for this - revise it
class WeightedMetropolisStatistics(WeightedStatistics, ConstrainingStatistics, WeightedMCMCPostProcessorElement):
    def compute_basic_stats(self):
        #TODO make this code less ridiculous
        self.mu = []
        self.sigma = []
        self.median = []
        self.l95 = []
        self.u95 = []
        self.l68 = []
        self.u68 = []
        self.lerr68 = []
        self.uerr68 = []
        self.lerr95 = []
        self.uerr95 = []
        self.peak1d = []      
        try:self.best_fit_index = self.source.get_col("post").argmax()
        except:self.best_fit_index = self.source.get_col("like").argmax()
        n = 0
        for col in self.source.colnames:
            n, mu, sigma, median, l68, u68, l95, u95,  lerr68, uerr68, lerr95, uerr95, peak1d = self.compute_basic_stats_col(col)
            self.mu.append(mu)
            self.sigma.append(sigma)
            self.median.append(median)
            self.l68.append(l68)
            self.u68.append(u68)
            self.l95.append(l95)
            self.u95.append(u95)
            self.lerr68.append(lerr68)
            self.uerr68.append(uerr68)
            self.lerr95.append(lerr95)
            self.uerr95.append(uerr95)
            self.peak1d.append(peak1d)
        return n

    def run(self):
        N = self.compute_basic_stats()
        print("Samples after cutting:", N)

        self.report_screen()
        files = self.report_file()
        return files

class MultinestCovariance(ChainCovariance, Statistics, MultinestPostProcessorElement):
    pass

class PolychordCovariance(MultinestCovariance):
    pass


class CovarianceMatrix1D(Statistics):
    def run(self):
        params = self.source.colnames
        Sigma = np.linalg.inv(self.source.data[0]).diagonal()**0.5
        Mu = [float(self.source.metadata[0]['mu_{0}'.format(i)]) for i in range(Sigma.size)]        
        header = '#'+'    '.join(params)
        f, filename, new_file = self.get_text_output("means", header, self.source.name)

        for P in zip(self.source.colnames, Mu, Sigma):
            f.write("%s   %e   %e\n" % P)

        print()
        print("Marginalized mean, std-dev:")
        for P in zip(self.source.colnames, Mu, Sigma):
            print('    %s = %g ± %g' % P)
        print()

        return [filename]

class CovarianceMatrixEllipseAreas(Statistics):
    def run(self):
        params = self.source.colnames
        header = '#param1  param2  area figure_of_merit'
        f, filename, new_file = self.get_text_output("ellipse_areas", header, self.source.name)

        covmat_estimate = np.linalg.inv(self.source.data[0])
        for i,p1 in enumerate(params[:]):
            for j,p2 in enumerate(params[:]):
                if j>=i: continue
                #Get the 2x2 sub-matrix
                C = covmat_estimate[:,[i,j]][[i,j],:]
                area = np.pi * np.linalg.det(C)
                fom = 1.0/area
                f.write("{0}  {1}  {2}  {3}\n".format(p1, p2, area, fom))

        return [filename]



class Citations(Statistics):
    #This isn't really a statistic but it uses all the same
    #mechanisms
    def run(self):
        print() 
        message = "#You should cite these papers in any publication based on this pipeline."
        print(message)
        citations = set()
        f, filename, new_file = self.get_text_output("citations", message, self.source.name)
        for comment_set in self.source.comments:
            for comment in comment_set:
                comment = comment.strip()
                if comment.startswith("CITE"):
                    citation =comment[4:].strip()
                    citations.add(citation)
        for citation in citations:
            print("    ", citation)
            f.write("%s\n"%citation)
        print()
        return [filename]
