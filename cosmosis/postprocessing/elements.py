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
