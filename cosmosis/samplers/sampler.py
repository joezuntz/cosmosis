sampler_registry = {}
from cosmosis.runtime.attribution import PipelineAttribution
from .hints import Hints

class Sampler(object):
    needs_output = True
    sampler_outputs = []
    parallel_output = False
    is_parallel_sampler = False
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
        self.distribution_hints = Hints()
        self.name = self.__class__.__name__[:-len("Sampler")].lower()
        if self.output:
            for p in pipeline.output_names():
                self.output.add_column(p, float)
            for p,ptype in self.sampler_outputs:
                self.output.add_column(p, ptype)
            output.metadata("n_varied", len(self.pipeline.varied_params))

            self.attribution.write_output(self.output)
        blinding_header = self.ini.getboolean("output","blinding-header", False)
        if blinding_header and self.output:
            output.comment("")
            output.comment("Blank lines prevent accidental unblinding")
            for i in xrange(250):
                output.comment("")

    def read_ini(self, option, option_type, default=None):
        """
        Read option from the ini file for this sampler
        and also save to the output file if it exists
        """
        if option_type is float:
            val = self.ini.getfloat(self.name, option, default)
        elif option_type is int:
            val = self.ini.getint(self.name, option, default)
        elif option_type is bool:
            val = self.ini.getboolean(self.name, option, default)
        elif option_type is str:
            val = self.ini.get(self.name, option, default)
        else:
            raise ValueError("Internal cosmosis sampler error: "
                "tried to read ini file option with unknown type %s"%str(option_type))
        if self.output:
            self.output.metadata(option, str(val))
        return val


    def config(self):
        ''' Set up sampler (could instead use __init__) '''
        pass

    def execute(self):
        ''' Run one (self-determined) iteration of sampler.
            Should be enough to test convergence '''
        raise NotImplementedError

    def is_converged(self):
        return False
    
    def start_estimate(self):
        if self.distribution_hints.has_peak():
            start = self.distribution_hints.get_peak()
        else:
            start = self.pipeline.start_vector()
        return start



class ParallelSampler(Sampler):
    parallel_output = True
    is_parallel_sampler = True
    supports_smp = True
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

