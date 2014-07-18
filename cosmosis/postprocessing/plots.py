from .elements import PostProcessorElement
from ..plotting import cosmology_theory_plots
from ..plotting.kde import KDE
import pylab
import ConfigParser

class Plots(PostProcessorElement):
    def __init__(self, *args, **kwargs):
        super(Plots, self).__init__(*args, **kwargs)
        self.figures = {}
        latex_file = self.options.get("latex") 
        self._latex = {}
        if latex_file:
            self.load_latex(latex_file)

    def load_latex(self, latex_file):
        latex_names = {}
        latex_names = ConfigParser.ConfigParser()
        latex_names.read(latex_file)
        for i,col_name in enumerate(self.source.colnames):
            display_name=col_name
            if '--' in col_name:
                section,name = col_name.lower().split('--')
                try:
                    display_name = latex_names.get(section,name)
                except ConfigParser.NoOptionError:
                    pass                    
            else:
                if col_name in ["LIKE","like", "likelihood"]:
                    display_name=r"{\cal L}"
            self._latex[col_name]=display_name

    def latex(self, name, dollar):
        l = self._latex.get(name, name)
        if dollar:
            l = "$"+l+"$"
        return l

    def filename(self, base):
        output_dir = self.options.get("outdir", "png")
        prefix=self.options.get("prefix","")
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
    pass



class MetropolisHastingsPlots(Plots):

    def keywords_1d(self):
        return {}

    def make_1d_plot(self, name):
        x = self.source.get_col(name)
        filename = self.filename(name)
        figure = self.figure(filename)

        #Interpolate using KDE
        n = self.options.get("n_kde", 100)
        factor = self.options.get("factor_kde", 2.0)
        kde = KDE(x, factor=factor)
        x_axis, like = kde.grid_evaluate(n, (x.min(), x.max()) )

        #Make the plot
        pylab.figure(figure.number)
        keywords = self.keywords_1d()
        pylab.plot(x_axis, like, '-', **keywords)
        pylab.xlabel(self.latex(name, dollar=True))

        return filename

    def make_1d_plots(self):
        filenames = []
        for name in self.source.colnames:
            filename = self.make_1d_plot(name)
            filenames.append(filename)
        return filenames

    def run(self):
        filenames = self.make_1d_plots()
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
                p=cls(dirname, output_dir, prefix, ftype, figure=None)
                filename=p.filename
                fig = self.figure(filename)
                p.figure=fig
                p.plot()
                filenames.append(filename)
            except IOError as err:
                print err
        return filenames

class MultinestPlots(Plots):
    pass
