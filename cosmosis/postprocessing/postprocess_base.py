import abc
from . import elements
from . import plots
from . import statistics
import numpy as np
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
        self.steps = []
        self.load(ini)
        elements = [el for el in self.elements if (not issubclass(el, plots.Plots) or (not options.get("no_plots")))]
        self.steps = [e(self, **options) for e in elements]
        self.options=options

    def load_extra_steps(self, filename):
        extra = elements.PostProcessorElement.instances_from_file(filename, self, **self.options)
        for e in extra:
            print "Adding post-processor step: %s" % (e.__class__.__name__)
        self.steps.extend(extra)

    def load(self, ini):
        for step in self.steps:
            step.reset()
        filename = "Unknown"
        if self.cosmosis_standard_output:
            if isinstance(ini, tuple):
                self.colnames, self.data, self.metadata, self.comments, self.final_metadata = ini
            else:
                if isinstance(ini, dict):
                    output_options=ini["output"]
                    filename = output_options['filename']
                    sampler = ini['sampler']
                    sampler_options = {}
                    for key,val in ini[sampler].items():
                        sampler_options[(sampler,key)]=str(val)
                    self.colnames, self.data, self.metadata, self.comments, self.final_metadata = ini['data']
                    ini = Inifile(None, override=sampler_options)
                else:
                    output_options = dict(ini.items('output'))
                    filename = output_options['filename']
                    self.colnames, self.data, self.metadata, self.comments, self.final_metadata = \
                        output_module.input_from_options(output_options)
            #self.data = self.data[0].T
            self.data_stacked = np.concatenate(self.data).T
            self.colnames = [c.lower() for c in self.colnames]
        self.ini = ini
        self.name = filename

    def __len__(self):
        return self.data_stacked.shape[1]

    def get_row(self, index):
        return self.data_stacked[:,index]

    def get_col(self, index_or_name, stacked=True):
        """Get the named or numbered column."""
        if isinstance(index_or_name, int):
            index = index_or_name
        else:
            name = index_or_name
            index = self.colnames.index(name)
        cols = [d[:,index] for d in self.data]
        if stacked:
            return np.concatenate(cols)
        else:
            return cols

    def run(self):
        files = []
        for e in self.steps:
            files += e.run()
        for f in files:
            print "File: ", f
    def finalize(self):
        for e in self.steps:
            e.finalize()

    def apply_tweaks(self, tweaks):
        if tweaks.filename==plots.Tweaks.filename:
            print tweaks.filename
            print "Please fill in the 'filename' attribute of your tweaks"
            print "Put the base name (without the directory, prefix, or suffix)"
            print "of the filename you want to tweak."
            print "You can use also use a list for more than one plot,"
            print "or put '%s' to apply to all plots."%plots.Tweaks._all_filenames
            return
        for step in self.steps:
            if isinstance(step, plots.Plots):
                step.tweak(tweaks)



