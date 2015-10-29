import abc
from . import elements
from . import plots
from . import statistics
import numpy as np
import hashlib
from cosmosis import output as output_module
from ..runtime.config import Inifile
import imp
import os
postprocessor_registry = {}


def blinding_value(name):
    #hex number derived from code phrase
    m = hashlib.md5(name).hexdigest()
    #convert to decimal
    s = int(m, 16)
    # last 8 digits
    f = s%100000000
    # turn 8 digit number into value between 0 and 1
    g = f*1e-8
    #get value between -1 and 1
    return g*2-1



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
    def __init__(self, ini, index, **options):
        super(PostProcessor,self).__init__()
        self.options=options
        self.sampler_options={}
        self.steps = []
        self.index = index
        self.derive_file = options.get("derive", "")
        self.load(ini)
        elements = [el for el in self.elements if (not issubclass(el, plots.Plots) or (not options.get("no_plots")))]
        self.steps = [e(self, **options) for e in elements]
        self.outputs = {} #outputs can be anything, but for now matplotlib figures and open text files

    def load_extra_steps(self, filename):
        extra = elements.PostProcessorElement.instances_from_file(filename, self, **self.options)
        for e in extra:
            print "Adding post-processor step: %s" % (e.__class__.__name__)
        self.steps.extend(extra)

    def blind_data(self,multiplicative):
        #blind self.data
        for c,col in enumerate(self.colnames):
            if col.lower() in ['like', 'weight', 'log_weight', 'old_weight', 'old_log_weight']: continue
            #get col mean to get us a rough scale to work with
            if multiplicative:
                #use upper here so it is different from non-multiplicative
                #scale by value between 0.75 and 1.25
                for d in self.data:
                    scale = 0.2
                    d[:,c] *= (1+scale*blinding_value(col.lower()))
                print "Blinding scale value for %s in %f - %f" % (col, 1-scale, 1+scale)

            else:
                r = np.mean([d[:,c].std() for d in self.data])
                if r==0.0:
                    print "Not blinding constant %s" % col
                    continue
                scale = (10**np.ceil(np.log10(abs(r))))
                #make a random number between -1 and 1 based on the column name
                for d in self.data:
                    d[:,c] += scale * blinding_value(col.upper())
                print "Blinding additive value for %s ~ %f" % (col, scale)

    def derive_extra_columns(self):
        if not self.derive_file: return
        name = os.path.splitext(os.path.split(self.derive_file)[1])[0]
        module = imp.load_source(name, self.derive_file)
        functions = [getattr(module,f) for f in dir(module) if f.startswith('derive_')]
        print "Deriving new columns from these functions in {}:".format(self.derive_file)
        for f in functions:
            print "    - ", f.__name__
            new_data = []
            for d in self.data:
                chain = SingleChainData(d,self.colnames)
                col, code = f(chain)
                #insert a new column into the chain, second from the end
                d = np.insert(d, -2, col, axis=1)
                #save the new chain
                new_data.append(d)
            self.colnames.insert(-2, code)
            print "Added a new column called ", code
            self.data = new_data

    def load_tuple(self, inputs):
        self.colnames, self.data, self.metadata, self.comments, self.final_metadata = inputs
        self.name = "Data"
        for chain in self.metadata:
            for key,val in chain.items():
                self.sampler_options[key] = val

    def load_dict(self, inputs):
        output_options=inputs["output"]
        filename = output_options['filename']
        self.name = filename
        sampler = inputs['sampler']
        for key,val in inputs[sampler].items():
            self.sampler_options[key]=str(val)
        self.colnames, self.data, self.metadata, self.comments, self.final_metadata = inputs['data']

    def load_ini(self, inputs):
        output_options = dict(inputs.items('output'))
        filename = output_options['filename']
        self.name = filename
        sampler = inputs.get("runtime", "sampler")
        for key,val in inputs.items(sampler):
            self.sampler_options[key]=str(val)        
        self.colnames, self.data, self.metadata, self.comments, self.final_metadata = \
            output_module.input_from_options(output_options)

    def sampler_option(self, key, default=None):
        return self.sampler_options.get(key, default)

    def load_chain(self, ini):
        if isinstance(ini, tuple):
            self.load_tuple(ini)
        elif isinstance(ini, dict):
            self.load_dict(ini)
        else:
            self.load_ini(ini)

        #derive any additional parameters
        self.derive_extra_columns()

        #set the column names
        self.colnames = [c.lower() for c in self.colnames]
        if self.options.get('blind_add',False):
            self.blind_data(multiplicative=False)
        if self.options.get('blind_mul',False):
            self.blind_data(multiplicative=True)
        self.data_stacked = np.concatenate(self.data).T

    def load(self, ini):
        if self.cosmosis_standard_output:
            self.load_chain(ini)
        

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
                print "Failed in one of the postprocessing steps: ", e
                print "Here is the error stack:"
                print(traceback.format_exc())

    def finalize(self):
        print "Finalizing:"
        for f in self.outputs.values():
            print "Output: ", f.filename
            f.finalize()

    def apply_tweaks(self, tweaks):
        if tweaks.filename==plots.Tweaks.filename:
            print tweaks.filename
            print "Please fill in the 'filename' attribute of your tweaks"
            print "Put the base name (without the directory, prefix, or suffix)"
            print "of the filename you want to tweak."
            print "You can use also use a list for more than one plot,"
            print "or put '%s' to apply to all plots."%plots.Tweaks._all_filenames
            return
        elif tweaks.filename==tweaks._all_filenames:
            filenames = [o.name for o in self.outputs if isinstance(o, plots.Plots)]
        elif isinstance(tweaks.filename, list):
                filenames = tweaks.filename
        else:
            filenames = [tweaks.filename]
        for output in self.outputs.values():
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
