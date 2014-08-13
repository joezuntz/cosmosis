from .elements import PostProcessorElement, MCMCPostProcessorElement, MultinestPostProcessorElement 
from ..plotting.kde import KDE
from .utils import std_weight, mean_weight
from . import cosmology_theory_plots
import ConfigParser
import numpy as np
import scipy.optimize
from . import lazy_pylab as pylab
import itertools
import os

default_latex_file = os.path.join(os.path.split(__file__)[0], "latex.ini")

class Plots(PostProcessorElement):
    def __init__(self, *args, **kwargs):
        super(Plots, self).__init__(*args, **kwargs)
        self.figures = {}
        self.no_latex = self.options.get("no_latex")
        latex_file = self.options.get("more_latex") 
        self._latex = {}
        if self.source.cosmosis_standard_output and not self.no_latex:
            self.load_latex(latex_file)

    def load_latex(self, latex_file):
        latex_names = {}
        latex_names = ConfigParser.ConfigParser()
        latex_names.read(default_latex_file)
        if latex_file:
            latex_names.read(latex_file)
        for i,col_name in enumerate(self.source.colnames):
            display_name=col_name
            if '--' in col_name:
                section,name = col_name.lower().split('--')
                try:
                    display_name = latex_names.get(section,name)
                except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
                    pass

            else:
                if col_name in ["LIKE","like", "likelihood"]:
                    display_name=r"{\cal L}"
                else:
                    try:
                        display_name = latex_names.get("misc",col_name)
                    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
                        pass

            self._latex[col_name]=display_name

    def latex(self, name, dollar=True):
        l = self._latex.get(name)
        if l is None: return name
        if dollar:
            l = "$"+l+"$"
        return l

    def filename(self, base, *bases):
        if bases:
            base = base + "_" + ("_".join(bases))
        output_dir = self.options.get("outdir", "png")
        prefix=self.options.get("prefix","")
        if prefix: prefix+="_"
        ftype=self.options.get("file_type", "png")
        return "{0}/{1}{2}.{3}".format(output_dir, prefix, base, ftype)

    def figure(self, name):
        #we want to be able to plot multiple chains on the same
        #figure at some point.  So when we make figures we should
        #call this function
        fig = self.figures.get(name)
        if fig is None:
            fig = pylab.figure()
            self.figures[name] = fig
        return fig

    def save_figures(self):
        for filename, figure in self.figures.items():
            pylab.figure(figure.number)
            pylab.savefig(filename)
            pylab.close()
        self.figures = {}

    def finalize(self):
        self.save_figures()

    def run(self):
        print "I do not know how to generate plots for this kind of data"
        return []

class GridPlots(Plots):
    @staticmethod
    def find_grid_contours(like, contour1, contour2):
        like_total = like.sum()
        def objective(limit, target):
            w = np.where(like>limit)
            return like[w].sum() - target
        target1 = like_total*contour1
        target2 = like_total*contour2
        level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
        level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
        return level1, level2


class GridPlots1D(GridPlots):

    def run(self):
        return [self.plot_1d(name) for name in self.source.colnames]

    def plot_1d(self, name1):
        filename = self.filename(name1)
        cols1 = self.source.get_col(name1)
        like = self.source.get_col("like")
        vals1 = np.unique(cols1)
        n1 = len(vals1)
        like_sum = np.zeros(n1)

        #marginalize
        for k,v1 in enumerate(vals1):
            w = np.where(cols1==v1)
            like_sum[k] = np.log(np.exp(like[w]).sum())
        like = like_sum.flatten()

        #linearly interpolate
        n1 *= 10
        vals1_interp = np.linspace(vals1[0], vals1[-1], n1)
        like_interp = np.interp(vals1_interp, vals1, like)
        vals1 = vals1_interp
        like = like_interp

        #normalize
        like -= like.max()

        #Determine the spacing in the different parameters
        dx = vals1[1]-vals1[0]

        #Set up the figure
        fig = self.figure(filename)
        pylab.figure(fig.number)

        #Plot the likelihood
        pylab.plot(vals1, np.exp(like), linewidth=3)

        #Find the levels of the 68% and 95% contours
        X, L = self.find_edges(np.exp(like), 0.68, 0.95, vals1)
        #Plot black dotted lines from the y-axis at these contour levels
        for (x, l) in zip(X,L):
            pylab.plot([x[0],x[0]], [0, l[0]], ':', color='black')
            pylab.plot([x[1],x[1]], [0, l[1]], ':', color='black')

        #Set the x and y limits
        pylab.xlim(cols1.min()-dx/2., cols1.max()+dx/2.)
        pylab.ylim(0,1.05)
        #Add label
        pylab.xlabel(self.latex(name1))
        return filename

    @classmethod
    def find_edges(cls, like, contour1, contour2, vals1):
        X = []
        L = []
        level1,level2=cls.find_grid_contours(like, contour1, contour2)
        for level in [level1, level2]:
            above = np.where(like>level)[0]
            left = above[0]
            right = above[-1]
            x1 = vals1[left]
            x2 = vals1[right]
            L1 = like[left]
            L2 = like[right]
            X.append((x1,x2))
            L.append((L1,L2))
        return X,L



class GridPlots2D(GridPlots):
    def run(self):
        filenames=[]
        for i, name1 in enumerate(self.source.colnames[:]):
            for name2 in self.source.colnames[i:]:
                if name1<=name2: continue
                filename=self.plot_2d(name1, name2)
                if filename: filenames.append(filename)
        return filenames

    def plot_2d(self, name1, name2):    
        # Load the columns
        cols1 = self.source.get_col(name1)
        cols2 = self.source.get_col(name2)
        like = self.source.get_col("like")
        vals1 = np.unique(cols1)
        vals2 = np.unique(cols2)
        n1 = len(vals1)
        n2 = len(vals2)
        if n1!=n2: return        
        filename = self.filename("2D", name1, name2)

        #Marginalize over all the other parameters by summing
        #them up
        like_sum = np.zeros((n1,n2))
        for k,(v1, v2) in enumerate(itertools.product(vals1, vals2)):
            w = np.where((cols1==v1)&(cols2==v2))
            i,j = np.unravel_index(k, like_sum.shape)
            like_sum[i,j] = np.log(np.exp(like[w]).sum())
        like = like_sum.flatten()

        #Normalize the log-likelihood to peak=0
        like -= like.max()

        #Choose a color mapping
        norm = pylab.matplotlib.colors.Normalize(np.exp(like.min()), np.exp(like.max()))
        colormap = pylab.cm.Reds

        #Create the figure
        fig = self.figure(filename)
        pylab.figure(fig.number)

        #Decide whether to do a smooth or block plot
        smooth = self.options.get("smooth", True)
        if smooth:
            interpolation='bilinear'
        else:
            interpolation='nearest'

        like = np.exp(like).reshape((n1,n2))
        extent=(vals2[0], vals2[-1], vals1[0], vals1[-1])
        #Make the plot
        pylab.imshow(like, extent=extent, 
            aspect='auto', cmap=colormap, norm=norm, interpolation=interpolation, origin='lower')
        
        sm = pylab.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm._A = [] #hack from StackOverflow to make this work
        pylab.colorbar(sm, label='Likelihood')

        #Add contours
        level1, level2 = self.find_grid_contours(like, 0.68, 0.95)
        pylab.contour(like, levels = [level1, level2], extent=extent)
        pylab.xlabel(self.latex(name2))
        pylab.ylabel(self.latex(name1))

        return filename

class MetropolisHastingsPlots(Plots, MCMCPostProcessorElement):
    pass


class MetropolisHastingsPlots1D(MetropolisHastingsPlots):
    excluded_colums = ["like"]
    def keywords_1d(self):
        return {}

    def smooth_likelihood(self, x):
        #Interpolate using KDE
        n = self.options.get("n_kde", 100)
        factor = self.options.get("factor_kde", 2.0)
        kde = KDE(x, factor=factor)
        x_axis, like = kde.grid_evaluate(n, (x.min(), x.max()) )
        return n, x_axis, like

    def make_1d_plot(self, name):
        x = self.reduced_col(name)
        filename = self.filename(name)
        figure = self.figure(filename)

        if x.max()-x.min()==0: return

        n, x_axis, like = self.smooth_likelihood(x)

        #Make the plot
        pylab.figure(figure.number)
        keywords = self.keywords_1d()
        pylab.plot(x_axis, like, '-', **keywords)
        pylab.xlabel(self.latex(name, dollar=True))

        return filename

    def run(self):
        filenames = []
        for name in self.source.colnames:
            if name.lower() in self.excluded_colums: continue
            filename = self.make_1d_plot(name)
            if filename: filenames.append(filename)
        return filenames


class MetropolisHastingsPlots2D(MetropolisHastingsPlots):
    excluded_colums = ["like"]
    def keywords_2d(self):
        return {}

    def _find_contours(self, like, x, y, n, xmin, xmax, ymin, ymax, contour1, contour2):
        N = len(x)
        x_axis = np.linspace(xmin, xmax, n+1)
        y_axis = np.linspace(ymin, ymax, n+1)
        histogram, _, _ = np.histogram2d(x, y, bins=[x_axis, y_axis])

        def objective(limit, target):
            w = np.where(like>limit)
            count = histogram[w]
            return count.sum() - target
        target1 = N*(1-contour1)
        target2 = N*(1-contour2)
        level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
        level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
        return level1, level2, like.sum()

    def smooth_likelihood(self, x, y):
        n = self.options.get("n_kde", 100)
        factor = self.options.get("factor_kde", 2.0)
        kde = KDE([x,y], factor=factor)
        x_range = (x.min(), x.max())
        y_range = (y.min(), y.max())
        (x_axis, y_axis), like = kde.grid_evaluate(n, [x_range, y_range])
        return n, x_axis, y_axis, like        

    def make_2d_plot(self, name1, name2):
        #Get the data
        x = self.reduced_col(name1)
        y = self.reduced_col(name2)

        if x.max()-x.min()==0 or y.max()-y.min()==0:
            return
        print "  (making %s vs %s)" % (name1, name2)

        filename = self.filename("2D", name1, name2)
        figure = self.figure(filename)

        #Interpolate using KDE
        n, x_axis, y_axis, like = self.smooth_likelihood(x, y)


        #Choose levels at which to plot contours
        contour1=1-0.68
        contour2=1-0.95
        level1, level2, total_mass = self._find_contours(like, x, y, n, x_axis[0], x_axis[-1], y_axis[0], y_axis[-1], contour1, contour2)
        level0 = 1.1
        levels = [level2, level1, level0]


        #Make the plot
        pylab.figure(figure.number)
        keywords = self.keywords_2d()
        fill = self.options.get("fill", True)
        imshow = self.options.get("imshow", False)
        plot_points = self.options.get("plot_points", False)

        if imshow:
            pylab.imshow(like.T, extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]), aspect='auto', origin='lower')
            pylab.colorbar()
        elif fill:
            pylab.contourf(x_axis, y_axis, like.T, [level2,level0], colors=['b'], alpha=0.25)
            pylab.contourf(x_axis, y_axis, like.T, [level1,level0], colors=['b'], alpha=0.25)
        else:
            pylab.contour(x_axis, y_axis, like.T, [level2,level1], colors='b')
        if plot_points:
            pylab.plot(x, y, ',')


        #Do the labels
        pylab.xlabel(self.latex(name1))
        pylab.ylabel(self.latex(name2))

        return filename        


    def run(self):
        filenames = []
        print "(Making 2D plots using KDE; this takes a while but is really cool)"
        for name1 in self.source.colnames[:]:
            for name2 in self.source.colnames[:]:
                if name1<=name2: continue
                if name1.lower() in self.excluded_colums: continue
                if name2.lower() in self.excluded_colums: continue
                filename = self.make_2d_plot(name1, name2)
                if filename:
                    filenames.append(filename)
        return filenames


class TestPlots(Plots):
    def run(self):
        ini=self.source.ini
        dirname = ini.get("test", "save_dir")
        output_dir = self.options.get("outdir", "png")
        prefix=self.options.get("prefix","")
        ftype=self.options.get("file_type", "png")
        filenames = []
        for cls in cosmology_theory_plots.plot_list:
            try:
                p=cls(dirname, output_dir, prefix, ftype, figure=1)
                filename=p.filename
                fig = self.figure(filename)
                p.figure=fig
                p.plot()
                filenames.append(filename)
            except IOError as err:
                print err
        return filenames



class MultinestPlots1D(MultinestPostProcessorElement, MetropolisHastingsPlots1D):
    excluded_colums = ["like", "weight"]
    def smooth_likelihood(self, x):
        #Interpolate using KDE
        n = self.options.get("n_kde", 100)
        weights = self.reduced_col("weight")
        #speed things up by removing zero-weighted samples

        dx = std_weight(x, weights)*4
        mu_x = mean_weight(x, weights)
        x_range = (max(x.min(), mu_x-dx), min(x.max(), mu_x+dx))

        factor = self.options.get("factor_kde", 2.0)
        kde = KDE(x, factor=factor, weights=weights)
        x_axis, like = kde.grid_evaluate(n, x_range )
        return n, x_axis, like


class MultinestPlots2D(MultinestPostProcessorElement, MetropolisHastingsPlots2D):
    excluded_colums = ["like", "weight"]
    def smooth_likelihood(self, x, y):
        n = self.options.get("n_kde", 100)
        fill = self.options.get("fill", True)
        factor = self.options.get("factor_kde", 2.0)
        weights = self.weight_col()

        kde = KDE([x,y], factor=factor, weights=weights)
        dx = std_weight(x, weights)*4
        dy = std_weight(y, weights)*4
        mu_x = mean_weight(x, weights)
        mu_y = mean_weight(y, weights)
        x_range = (max(x.min(), mu_x-dx), min(x.max(), mu_x+dx))
        y_range = (max(y.min(), mu_y-dy), min(y.max(), mu_y+dy))
        (x_axis, y_axis), like = kde.grid_evaluate(n, [x_range, y_range])
        return n, x_axis, y_axis, like        

    def _find_contours(self, like, x, y, n, xmin, xmax, ymin, ymax, contour1, contour2):
        N = len(x)
        x_axis = np.linspace(xmin, xmax, n+1)
        y_axis = np.linspace(ymin, ymax, n+1)
        weights = self.weight_col()
        histogram, _, _ = np.histogram2d(x, y, bins=[x_axis, y_axis], weights=weights)
        def objective(limit, target):
            w = np.where(like>=limit)
            count = histogram[w]
            return count.sum() - target
        target1 = histogram.sum()*(1-contour1)
        target2 = histogram.sum()*(1-contour2)

        level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
        level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
        return level1, level2, like.sum()
