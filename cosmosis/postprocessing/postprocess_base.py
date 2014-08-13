import abc
from . import plots
from . import statistics
from cosmosis import output as output_module
from ..runtime.config import Inifile
postprocessor_registry = {}

class PostProcessMetaclass(abc.ABCMeta):
    def __init__(cls, name, b, d):
        abc.ABCMeta.__init__(cls, name, b, d)
        sampler = d.get("sampler")
        if d is None: return
        postprocessor_registry[sampler] = cls



class PostProcessor(object):
    __metaclass__=PostProcessMetaclass
    sampler=None
    cosmosis_standard_output=True
    def __init__(self, ini, **options):
        super(PostProcessor,self).__init__()
        self.load(ini)
        elements = [el for el in self.elements if (not issubclass(el, plots.Plots) or (not options.get("no_plots")))]
        self.steps = [e(self, **options) for e in elements]
        self.options=options

    def load(self, ini):
        if self.cosmosis_standard_output:
            if isinstance(ini, tuple):
                self.colnames, self.data, self.metadata, self.comments, self.final_metadata = ini
            else:
                if isinstance(ini, dict):
                    output_options=ini["output"]
                    sampler = ini['sampler']
                    sampler_options = {}
                    for key,val in ini[sampler].items():
                        sampler_options[(sampler,key)]=str(val)
                    self.colnames, self.data, self.metadata, self.comments, self.final_metadata = ini['data']
                    ini = Inifile(None, override=sampler_options)
                else:
                    output_options = dict(ini.items('output'))
                    self.colnames, self.data, self.metadata, self.comments, self.final_metadata = \
                    output_module.input_from_options(output_options)
            self.data = self.data[0].T
            self.colnames = [c.lower() for c in self.colnames]
        self.ini = ini

    def __len__(self):
        return self.data.shape[1]

    def get_row(self, index):
        return self.data[:,index]

    def get_col(self, index_or_name):
        if isinstance(index_or_name, int):
            index = index_or_name
        else:
            name = index_or_name
            index = self.colnames.index(name)
        return self.data[index]

    def run(self):
        files = []
        for e in self.steps:
            files += e.run()
        for f in files:
            print "File: ", f
    def finalize(self):
        for e in self.steps:
            e.finalize()

