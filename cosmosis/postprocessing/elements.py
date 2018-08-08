from __future__ import print_function
from builtins import object
import os
import sys
import numpy as np

class Loadable(object):
    """ A series of methods to allow creating instances from python files"""
    @classmethod
    def subclasses_in_file(cls, filepath):
        dirname, filename = os.path.split(filepath)
        # allows .pyc and .py modules to be used
        impname, ext = os.path.splitext(filename)
        sys.path.insert(0, dirname)
        try:
            library = __import__(impname)
        except Exception as error:
            print("Could not find/load extension file %s:" % filepath)
            print("ERROR:", error)
            return []

        subclasses = []
        for x in dir(library):
            x = getattr(library, x)
            if type(x) == type and issubclass(x, cls) and not x is cls:
                subclasses.append(x)
        return subclasses


    @classmethod
    def instances_from_file(cls, filename, *args, **kwargs):
        subclasses = cls.subclasses_in_file(filename)
        return [s(*args, **kwargs) for s in subclasses]




class PostProcessorElement(Loadable):
    def __init__(self, data_source, **options):
        super(PostProcessorElement,self).__init__()
        self.source = data_source
        self.options = {}
        self.options.update(options)

    def get_output(self, name):
        return self.source.outputs.get(name)

    def filename(self, ftype, base, *bases):
        if bases:
            base = base + "_" + ("_".join(bases))
        output_dir = self.options.get("outdir", ".")
        prefix=self.options.get("prefix","")
        if prefix: prefix+="_"
        return "{0}/{1}{2}.{3}".format(output_dir, prefix, base, ftype)

    def set_output(self, name, value):
        if name in self.source.outputs:
            raise ValueError("Two different postprocessors tried to create the same named file - use get_output to re-open an existing file")
        self.source.outputs[name] = value

    def run(self):
        print("I do not know how to produce some results for this kind of data")
        return []

    def finalize(self):
        pass

class MCMCPostProcessorElement(PostProcessorElement):
    def reduced_col(self, name, stacked=True):
        return self.source.reduced_col(name, stacked=stacked)

    def posterior_sample(self):
        return self.source.posterior_sample()

class WeightedMCMCPostProcessorElement(MCMCPostProcessorElement):
    def weight_col(self):
        return self.source.weight_col()

class MultinestPostProcessorElement(MCMCPostProcessorElement):
    def weight_col(self):
        return self.source.weight_col()

class PolychordPostProcessorElement(MultinestPostProcessorElement):
    pass
