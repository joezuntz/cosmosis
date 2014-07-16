import abc
from . import plots
from . import statistics
from cosmosis import output as output_module

postprocessor_registry = {}

class PostProcessMetaclass(abc.ABCMeta):
    def __init__(cls, name, b, d):
        abc.ABCMeta.__init__(cls, name, b, d)
        sampler = d.get("sampler")
        if d is None: return
        postprocessor_registry[sampler] = cls



class PostProcessor(object):
    __metaclass__=PostProcessMetaclass
    plotClass=plots.Plots
    statsClass=statistics.Statistics
    additionalElements = []
    sampler=None
    cosmosis_standard_output=True
    def __init__(self, ini, **options):
        self.plotter = self.plotClass(self, **options)
        self.stats = self.statsClass(self, **options)
        self.elements = [e(self, **options) for e in self.additionalElements]
        self.ini = ini
        if self.cosmosis_standard_output:
            output_options = dict(ini.items('output'))
            self.colnames, self.data, self.metadata, self.comments, self.final_metadata = \
                output_module.input_from_options(output_options)
            self.data = self.data[0].T
            self.colnames = [c.lower() for c in self.colnames]

    def get_col(self, index_or_name):
        if isinstance(index_or_name, int):
            index = index_or_name
        else:
            name = index_or_name
            index = self.colnames.index(name)
        return self.data[index]

    def run(self, stats=True, plots=True):
        files = []
        if plots:
            files += self.plotter.run()
        if stats:
            files += self.stats.run()
        for e in self.elements:
            files += e.run()
        for f in files:
            print "File: ", f

