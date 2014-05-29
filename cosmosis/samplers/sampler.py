sampler_registry = {}
from cosmosis.runtime.attribution import PipelineAttribution

class Sampler(object):
    needs_output = True
    class __metaclass__(type):
        def __init__(cls, name, b, d):
            type.__init__(cls, name, b, d)
            if name in ["Sampler", "ParallelSampler"]:
                return
            if not name.endswith("Sampler"):
                raise ValueError("Sampler classes must be named [Name]Sampler")
            config_name = name[:-len("Sampler")].lower()
            sampler_registry[config_name] = cls
    
    def __init__(self, ini, pipeline, output):
        self.ini = ini
        self.pipeline = pipeline
        self.output = output
        self.attribution = PipelineAttribution(self.pipeline.modules)
        if self.output:
            for p in pipeline.output_names():
                self.output.add_column(p, float)
            self.attribution.write_output(self.output)
        blinding_header = self.ini.getboolean("output","blinding-header", False)
        if blinding_header:
            output.comment("")
            output.comment("Blank lines prevent accidental unblinding")
            for i in xrange(250):
                output.comment("")


    def config(self):
        ''' Set up sampler (could instead use __init__) '''
        pass

    def execute(self):
        ''' Run one (self-determined) iteration of sampler.
            Should be enough to test convergence '''
        raise NotImplementedError

    def is_converged(self):
        return False


class ParallelSampler(Sampler):
    parallel_output = True

    def __init__(self, ini, pipeline, output, pool=None):
        Sampler.__init__(self, ini, pipeline, output)
        self.pool = pool

    def worker(self):
        ''' Default to a map-style worker '''
        if self.pool:
            self.pool.wait()
        else:
            raise RuntimeError("Worker function called when no parallel pool exists!")

    def is_master(self):
        return self.pool is None or self.pool.is_master()
