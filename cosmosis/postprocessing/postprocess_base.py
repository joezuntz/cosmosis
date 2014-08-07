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
    sampler=None
    cosmosis_standard_output=True
    def __init__(self, ini, **options):
        self.steps = [e(self, **options) for e in self.elements]
        self.ini = ini
        if self.cosmosis_standard_output:
            output_options = dict(ini.items('output'))
            self.colnames, self.data, self.metadata, self.comments, self.final_metadata = \
                output_module.input_from_options(output_options)
            self.data = self.data[0].T
            self.colnames = [c.lower() for c in self.colnames]

            burn = options.get("burn")
            if burn:
                self.data=self.data[:,burn:]

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

