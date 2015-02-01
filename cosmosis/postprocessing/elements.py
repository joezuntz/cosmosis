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
            print "Could not find/load extension file %s:" % filepath
            print "ERROR:", error
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

    def reset(self):
        pass

    def run(self):
        print "I do not know how to produce some results for this kind of data"
        return []

    def finalize(self):
        pass

class MCMCPostProcessorElement(PostProcessorElement):
    def reduced_col(self, name, stacked=True):
        cols = self.source.get_col(name, stacked=False)
        burn = self.options.get("burn", 0)
        thin = self.options.get("thin", 1)

        if 0.0<burn<1.0:
            burn = len(col)*burn
        else:
            burn = int(burn)
        cols = [col[burn::thin] for col in cols]
        if stacked:
            return np.concatenate(cols)
        else:
            return cols

    def posterior_sample(self):
        """
        A posterior sample of MCMC is just all the samples.

        Return an array of Trues with the same length as the chain

        """
        n = self.reduced_col(self.source.colnames[0]).size
        return np.ones(n, dtype=bool)

class WeightedMCMCPostProcessorElement(MCMCPostProcessorElement):
    def reduced_col(self, name, stacked=True):
        col = MCMCPostProcessorElement.reduced_col(self, name, stacked=stacked)
        return col

    def reset(self):
        super(WeightedMCMCPostProcessorElement, self).reset()
        if hasattr(self, "_weight_col"):
            del self._weight_col
        
    def weight_col(self):
        if hasattr(self, "_weight_col"):
            return self._weight_col
        w = MCMCPostProcessorElement.reduced_col(self, "weight").copy()
        w/=w.max()
        self._weight_col = w
        return self._weight_col    

class MultinestPostProcessorElement(PostProcessorElement):
    def reduced_col(self, name):
        #we only use the last n samples from a multinest output
        #file.  And omit zero-weighted samples.
        n = int(self.source.final_metadata[0]["nsample"])
        col = self.source.get_col(name)
        w = self.source.get_col("weight")[-n:]
        return col[-n:][w>0]

    def reset(self):
        super(MultinestPostProcessorElement, self).reset()
        if hasattr(self, "_weight_col"):
            del self._weight_col
        
    def weight_col(self):
        if hasattr(self, "_weight_col"):
            return self._weight_col
        n = int(self.source.final_metadata[0]["nsample"])
        w = self.source.get_col("weight")[-n:]
        w = w[w>0].copy()
        self._weight_col = w
        return self._weight_col

    def posterior_sample(self):
        """
        Multinest chains are *not* drawn from the posterior distribution
        but we do have the information we need to construct such a sample.

        This function returns a boolean array with True where we should
        use the sample at that index, and False where we should not.

        """
        w = self.weight_col()
        w = w / w.max()
        u = np.random.uniform(size=w.size)
        return u<w

