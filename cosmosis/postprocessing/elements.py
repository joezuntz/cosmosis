class PostProcessorElement(object):
    def __init__(self, data_source, **options):
        super(PostProcessorElement,self).__init__()
        self.source = data_source
        self.options = {}
        self.options.update(options)

    def run(self):
        print "I do not know how to produce some results for this kind of data"
        return []

    def finalize(self):
        pass

class MCMCPostProcessorElement(PostProcessorElement):
    def reduced_col(self, name):
        col = self.source.get_col(name)
        burn = self.options.get("burn", 0)
        thin = self.options.get("thin", 1)
        if 0.0<burn<1.0:
            burn = len(col)*burn
        else:
            burn = int(burn)
        return col[burn::thin]

class MultinestPostProcessorElement(PostProcessorElement):
    def reduced_col(self, name):
        #we only use the last n samples from a multinest output
        #file.  And omit zero-weighted samples.
        n = int(self.source.final_metadata[0]["nsample"])
        col = self.source.get_col(name)
        w = self.source.get_col("weight")[-n:]
        return col[-n:][w>0]
        
    def weight_col(self):
        if hasattr(self, "_weight_col"):
            return self._weight_col
        n = int(self.source.final_metadata[0]["nsample"])
        w = self.source.get_col("weight")[-n:]
        w = w[w>0].copy()
        self._weight_col = w
        return self._weight_col

