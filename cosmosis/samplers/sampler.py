from future.utils import with_metaclass
from builtins import str
from builtins import range
from builtins import object
from cosmosis.runtime.attribution import PipelineAttribution
from .hints import Hints

# Sampler metaclass that registers each of its subclasses

class RegisteredSampler(type):
    registry = {}
    def __new__(meta, name, bases, class_dict):
        if name.endswith("Sampler"):
            meta.registry = {name : cls for name, cls in meta.registry.items() if cls not in bases}
            config_name = name[:-len("Sampler")].lower()
            cls = type.__new__(meta, name, bases, class_dict)
            meta.registry[config_name] = cls
            return cls
        else:
            raise ValueError("Sampler classes must be named [Name]Sampler")

class Sampler(with_metaclass(RegisteredSampler, object)):
    needs_output = True
    sampler_outputs = []
    understands_fast_subspaces = False
    parallel_output = False
    is_parallel_sampler = False
    supports_resume = False

    
    def __init__(self, ini, pipeline, output):
        self.ini = ini
        self.pipeline = pipeline
        self.output = output
        self.attribution = PipelineAttribution(self.pipeline.modules)
        self.distribution_hints = Hints()
        self.name = self.__class__.__name__[:-len("Sampler")].lower()
        self.write_header()

    def write_header(self):
        if self.output:
            for p in self.pipeline.output_names():
                self.output.add_column(p, float)
            for p,ptype in self.sampler_outputs:
                self.output.add_column(p, ptype)
            self.output.metadata("n_varied", len(self.pipeline.varied_params))

            self.attribution.write_output(self.output)
        blinding_header = self.ini.getboolean("output","blinding-header", fallback=False)
        if blinding_header and self.output:
            self.output.blinding_header()

    def read_ini(self, option, option_type, default=None):
        """
        Read option from the ini file for this sampler
        and also save to the output file if it exists
        """
        if option_type is float:
            val = self.ini.getfloat(self.name, option, fallback=default)
        elif option_type is int:
            val = self.ini.getint(self.name, option, fallback=default)
        elif option_type is bool:
            val = self.ini.getboolean(self.name, option, fallback=default)
        elif option_type is str:
            val = self.ini.get(self.name, option, fallback=default)
        else:
            raise ValueError("Internal cosmosis sampler error: "
                "tried to read ini file option with unknown type {}".format(option_type))
        if self.output:
            self.output.metadata(option, str(val))
        return val

    def read_ini_choices(self, option, option_type, choices, default=None):
        value = self.read_ini(option, option_type, default=default)
        if value not in choices:
            name = self.__class__.__name__
            raise ValueError("Parameter {} for sampler {} must be one of: {}\n Parameter file said: {}".format(option, name, choices, value))
        return value


    def config(self):
        ''' Set up sampler (could instead use __init__) '''
        pass

    def execute(self):
        ''' Run one (self-determined) iteration of sampler.
            Should be enough to test convergence '''
        raise NotImplementedError

    def resume(self):
        raise NotImplementedError("The sampler {} does not support resuming".format(self.name))

    def is_converged(self):
        return False
    
    def start_estimate(self):
        if self.distribution_hints.has_peak():
            start = self.distribution_hints.get_peak()
        else:
            start = self.pipeline.start_vector()
        return start

    def cov_estimate(self):
        covmat_file = self.read_ini("covmat", str, "")
        n = len(self.pipeline.varied_params)

        if self.distribution_hints.has_cov():
            # hints from a previous sampler take
            # precendence
            covmat = self.distribution_hints.get_cov()

        elif covmat_file:
            covmat = np.loadtxt(covmat_file)
            # Fix the size.
            # If there is only one sample parameter then 
            # we assume it is a 1x1 matrix
            # If it's a 1D vector then assume these are
            # standard deviations
            if covmat.ndim == 0:
                covmat = covmat.reshape((1, 1))
            elif covmat.ndim == 1:
                covmat = np.diag(covmat ** 2)

            # Error on incorrect shapes or sizes
            if covmat.shape[0] != covmat.shape[1]:
                raise ValueError("Covariance matrix from {}"
                                 "not square".format(covmat_file))
            if covmat.shape[0] != n:
                raise ValueError("Covariance matrix from {} "
                                 "is the wrong shape ({}x{}) "
                                 "for the {} varied params".format(
                                    covmat_file, covmat.shape[0], n))
        else:
            # Just try a minimal estimate - 5% of prior width as standard deviation
            covmat = np.diag([w*p.width() for p in self.pipeine.varied_params])**2

        return covmat



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
