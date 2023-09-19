import abc
from . import elements
from . import plots
from . import statistics
import numpy as np
from cosmosis import output as output_module
from ..runtime import Inifile
from ..runtime.utils import import_by_path
import os
postprocessor_registry = {}

from io import StringIO


class PostProcessMetaclass(abc.ABCMeta):
    def __init__(cls, name, b, d):
        abc.ABCMeta.__init__(cls, name, b, d)
        sampler = d.get("sampler")
        if d is None: return
        postprocessor_registry[sampler] = cls

class PostProcessor(metaclass=PostProcessMetaclass):
    sampler=None
    cosmosis_standard_output=True
    def __init__(self, ini, label, index, **options):
        super(PostProcessor,self).__init__()
        self.options=options
        self.sampler_options={}
        self.steps = []
        self.index = index
        self.label = label
        self.derive_file = options.get("derive", "")
        self.load(ini)
        elements = [el for el in self.elements if (not issubclass(el, plots.Plots) or (not options.get("no_plots")))]
        self.steps = [e(self, **options) for e in elements]
        self.outputs = {} #outputs can be anything, but for now matplotlib figures and open text files

    def load_extra_steps(self, filename):
        extra = elements.PostProcessorElement.instances_from_file(filename, self, **self.options)
        for e in extra:
            print("Adding post-processor step: %s" % (e.__class__.__name__))
        self.steps.extend(extra)

    def add_rerun_bestfit_step(self, dirname):
        from .reruns import BestFitRerunner
        rerunner = BestFitRerunner(dirname, self, **self.options)
        self.steps.append(rerunner)



    def approximate_scale_ceiling(self, c):
        r = np.mean([d[:,c].std() for d in self.data])
        scale = (10**np.ceil(np.log10(abs(r))))
        return scale

    def additive_blind_column(self, c, value):
        for d in self.data:
            d[:,c] += value

    def multiplicative_blind_column(self, c, value):
        for d in self.data:
            d[:,c] *= (1+value)

    def derive_extra_column(self, function):
        new_data = []
        for d in self.data:
            chain = SingleChainData(d,self.colnames)
            col, code = function(chain)
            if col is None:
                break
            #insert a new column into the chain, second from the end
            d = np.insert(d, -2, col, axis=1)
            #save the new chain
            new_data.append(d)
        if code is None:
            return

        self.colnames.insert(-2, code)
        self.data = new_data



    def derive_extra_columns(self):
        if not self.derive_file: return
        name = os.path.splitext(os.path.split(self.derive_file)[1])[0]
        module = import_by_path(name, self.derive_file)
        functions = [getattr(module,f) for f in dir(module) if f.startswith('derive_')]
        print("Deriving new columns from these functions in {}:".format(self.derive_file))
        for f in functions:
            self.derive_extra_column(f)

    def load_tuple(self, inputs):
        self.colnames, self.data, self.metadata, self.comments, self.final_metadata = inputs
        self.name = "Data"
        for chain in self.metadata:
            for key,val in list(chain.items()):
                self.sampler_options[key] = val

    def load_dict(self, inputs):
        output_options=inputs["output"]
        filename = output_options['filename']
        self.name = filename
        sampler = inputs['sampler']
        self.colnames, self.data, self.metadata, self.comments, self.final_metadata = inputs['data']
        for chain in self.metadata:
            for key,val in list(chain.items()):
                self.sampler_options[key] = val

    def load_ini(self, inputs):
        output_options = dict(inputs.items('output'))
        filename = output_options['filename']
        self.name = filename
        sampler = inputs.get("runtime", "sampler")
        self.colnames, self.data, self.metadata, self.comments, self.final_metadata = \
            output_module.input_from_options(output_options)
        for chain in self.metadata:
            for key,val in list(chain.items()):
                self.sampler_options[key] = val

    def load_in_memory_storage(self, inputs):
        self.name = "chain"
        self.colnames = [c[0] for c in inputs.columns]
        self.data = [np.array(inputs.rows)]
        self.metadata = [{k:v[0] for k, v in inputs.meta.items()}]
        self.final_metadata = [{k:v[0] for k, v in inputs.final_meta.items()}]
        self.comments = [inputs.comments[:]]
        for chain in self.metadata:
            for key,val in list(chain.items()):
                self.sampler_options[key] = val
    
    def load_astropy(self, inputs):
        self.name = inputs.meta.get("chain_name", "chain")
        self.colnames = inputs.colnames
        # convert astropy table to numpy array
        self.data = [np.array([inputs[c] for c in self.colnames]).T]
        self.metadata = [inputs.meta]
        self.final_metadata = [{
            k[6:]: v
            for k, v in meta.items()
            if k.startswith("final:")
        } for meta in self.metadata]
        self.comments = [meta["comments"] for meta in self.metadata]
        for meta in self.metadata:
            for key, val in meta.items():
                self.sampler_options[key] = val

    def sampler_option(self, key, default=None):
        return self.sampler_options.get(key, default)

    def load_chain(self, ini):
        if "astropy" in str(type(ini)):
            self.load_astropy(ini)
        elif isinstance(ini, tuple):
            self.load_tuple(ini)
        elif isinstance(ini, dict):
            self.load_dict(ini)
        elif isinstance(ini, output_module.InMemoryOutput):
            self.load_in_memory_storage(ini)
        else:
            self.load_ini(ini)

        #derive any additional parameters
        self.derive_extra_columns()

        #set the column names
        self.colnames = [c.lower() for c in self.colnames]
        self.data_stacked = np.concatenate(self.data).T

    def load(self, ini):
        if self.cosmosis_standard_output:
            self.load_chain(ini)

    def extract_ini(self, tag):
        in_ = False
        lines = []
        for line in self.comments[0]:
            line = line.strip()
            if line == 'START_OF_{}_INI'.format(tag):
                in_ = True
            elif line == 'END_OF_{}_INI'.format(tag):
                break
            elif in_:
                lines.append(line)

        s = StringIO(u"\n".join(lines))
        ini = Inifile(None)
        ini.read_file(s)
        return ini



        

    def __len__(self):
        return self.data_stacked.shape[1]

    def get_row(self, index):
        return self.data_stacked[:,index]

    def has_col(self, name):
        return name in self.colnames

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
            try:
                files += e.run()
            except KeyboardInterrupt:
                raise
            except:
                import traceback
                if self.options.get("pdb", False):
                    print()
                    import pdb
                    pdb.post_mortem()
                elif self.options.get("fatal_errors", False):
                    raise
                else:                    
                    print("Failed in one of the postprocessing steps: ", e)
                    print("Here is the error stack:")
                    print(traceback.format_exc())
        return files

    def finalize(self):
        print("Finalizing:")
        for e in self.steps:
            e.finalize()
    
    def save(self):
        for f in list(self.outputs.values()):
            print("Output: ", f.filename)
            f.save()

    def apply_tweaks(self, tweaks):
        if tweaks.filename==plots.Tweaks.filename:
            print(tweaks.filename)
            print("Please fill in the 'filename' attribute of your tweaks")
            print("Put the base name (without the directory, prefix, or suffix)")
            print("of the filename you want to tweak.")
            print("You can use also use a list for more than one plot,")
            print("or put '%s' to apply to all plots."%plots.Tweaks._all_filenames)
            return
        elif tweaks.filename==tweaks._all_filenames:
            filenames = [o.name for o in list(self.outputs.values()) if isinstance(o, plots.PostprocessPlot)]
        elif isinstance(tweaks.filename, list):
                filenames = tweaks.filename
        else:
            filenames = [tweaks.filename]
        for output in list(self.outputs.values()):
            if output.name in filenames:
                output.tweak(tweaks)



class SingleChainData(object):
    """
    This helper object is to make it easier for users to write functions
    that derive new parameters.
    """
    def __init__(self, data, colnames):
        self.data = data
        self.colnames = colnames

    def __getitem__(self, index_or_name):
        if isinstance(index_or_name, int):
            index = index_or_name
        else:
            name = index_or_name
            index = self.colnames.index(name)
        return self.data[:,index]
