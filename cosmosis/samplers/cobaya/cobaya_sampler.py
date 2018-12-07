from builtins import zip
from builtins import range
from builtins import str
from .. import ParallelSampler
import numpy as np
import sys
from ...runtime import prior as priors
from collections import OrderedDict

def callback(mcmc):
    global sampler
    start = sampler.n_saved
    end = mcmc.collection.n()
    if end>start:
        print("Cosmosis saving samples {}-{}.".format(start,end))
        samples = mcmc.collection.data[start:end]
        for i,row in samples.iterrows():
            sampler.output_row(row)
        sampler.output.flush()
    sampler.n_saved = end

class CobayaSampler(ParallelSampler):
    parallel_output = True
    supports_resume = False
    sampler_outputs = [("weight", float), ("prior", float), ("post", float)]

    def config(self):
        try:
            import cobaya.run
        except ImportError:
            sys.stderr.write("")
            sys.stderr.write(r"*********************************************")
            sys.stderr.write("COBAYA NOT INSTALLED")
            sys.stderr.write("It can usually be installed using pip")
            sys.stderr.write(r"*********************************************")
            sys.stderr.write("")
            raise

        # For MPI - this needs to be a global variable
        global sampler
        sampler = self

        self.cobaya_run = cobaya.run.run
        self.done = False
        

        default_drag = self.pipeline.do_fast_slow

        self.covmat = self.cov_estimate()
        self.proposal_scale = self.read_ini("proposal_scale", float, 2.4)

        self.check_every = self.read_ini("check_every", str, "40d")
        self.output_every = self.read_ini("output_every", int, 20)

        self.burn_in = self.read_ini("burn_in", str, "20d")
        self.max_tries = self.read_ini("max_tries", str, "40d")
        self.max_samples = self.read_ini("max_samples", int, 0)
        if self.max_samples == 0:
            self.max_samples = np.inf


        # Parameters controlling the adaptive proposal
        self.learn_proposal = self.read_ini("learn_proposal", bool, True)
        self.learn_proposal_Rminus1_max = self.read_ini("learn_proposal_Rminus1_max", float, 2.0)
        self.learn_proposal_Rminus1_max_early = self.read_ini("learn_proposal_Rminus1_max_early", float, 30.0)
        self.learn_proposal_Rminus1_min = self.read_ini("learn_proposal_Rminus1_min", float, 0.0)

        # Parameters controlling Gelman-Rubin convergence testing
        self.Rminus1_stop = self.read_ini("Rminus1_stop", float, 0.01)
        self.Rminus1_cl_stop = self.read_ini("Rminus1_cl_stop", float, 0.2)
        self.Rminus1_cl_level = self.read_ini("Rminus1_cl_level", float, 0.95)
        self.Rminus1_single_split = self.read_ini("Rminus1_single_split", int, 4)


        self.oversample = self.read_ini("oversample", bool, False)
        self.drag = self.read_ini("drag", bool, default_drag)
        self.drag_limits = [int(x) for x in self.read_ini("drag_limits", str, "1,10").split(',')]


        if self.oversample and self.drag:
            raise ValueError("Can only set one oversample=T and drag=T for cobaya")
        if (self.oversample or self.drag) and not self.pipeline.do_fast_slow:
            raise ValueError("Need to set fast_slow=T in [pipeline] to use oversampling or dragging")
        


    def make_cobaya_info(self):

        varied_params = self.pipeline.varied_params
        if self.pipeline.do_fast_slow:
            fast_params = self.pipeline.fast_params
            slow_params = self.pipeline.slow_params
            # Slightly different definitions of the fast and slow times here
            fast_speed = 1.0/(self.pipeline.fast_time)
            slow_speed = 1.0/(self.pipeline.slow_time+self.pipeline.fast_time)
        else:
            fast_params = []
            slow_params = varied_params
            fast_speed = 0.1
            slow_speed = 1.0


        self.cobaya_derived_names = [
            "{}__{}".format(*d).lower() 
            for d in self.pipeline.extra_saves
        ]

        # The mock likelihood is a workaround for
        # cobaya's fast/slow parameter splitting.  We create
        # an initial mock likelihood function that depends
        # only on the slow parameters, but always returns zero
        likelihood = make_main_likelihood(varied_params, self.cobaya_derived_names)
        mock_slow_likelihood = make_slow_mock_likelihood(slow_params)

        likelihood_info = {
            "total": {"external": likelihood, "speed": fast_speed},
            "mock": {"external": mock_slow_likelihood, "speed": slow_speed},
        }

        # The dictionary that we pass cobaya describing the parameters.
        # Cobaya wants to handle the priors itself, so we let it
        params_info = OrderedDict()

        # cobaya doesn't like generic functions as priors specified against
        # individual parameters as part of the main definition
        # we have to specify them as additional priors that could nominally
        # apply to multiple parameters at once.
        # This in turn means we have to do some weird stuff to make the prior
        # parameter have the right name
        external_priors = OrderedDict()
        self.cobaya_param_names = []
        for p in varied_params:
            # For the initial prior we just supply the min/max range of the parameter
            # We add any actual priors below
            pname = str(p).replace('-','_')
            self.cobaya_param_names.append(pname)
            params_info[pname] = {
                "ref": p.start,
                "prior": {
                    "min": p.limits[0],
                    "max": p.limits[1],
                },
                "proposal": 0.05*(p.prior.b-p.prior.a),
            }
            # we have to do the p=p thing to capture the Parameter object in a closure
            external_priors['logpr_'+pname] = eval("lambda {0},p=p: p.prior({0}) ".format(pname))

        # This is how we tell cobaya that a parameter
        # is generated inside the likelihood itself
        # as a derived parameter
        for d in self.cobaya_derived_names:
            params_info[d] = None

        # Our various sampling options
        sampler_info = {
            "mcmc": {
                "callback_function": callback,
                "covmat": self.covmat,
                "covmat_params": self.cobaya_param_names,
                "proposal_scale": self.proposal_scale,
                "check_every": self.check_every,
                "output_every": self.output_every,
                "burn_in": self.burn_in,
                "max_tries": self.max_tries,
                "max_samples": self.max_samples,
                "learn_proposal": self.learn_proposal,
                "learn_proposal_Rminus1_max": self.learn_proposal_Rminus1_max,
                "learn_proposal_Rminus1_max_early": self.learn_proposal_Rminus1_max_early,
                "learn_proposal_Rminus1_min": self.learn_proposal_Rminus1_min,
                "Rminus1_stop": self.Rminus1_stop,
                "Rminus1_cl_stop": self.Rminus1_cl_stop,
                "Rminus1_cl_level": self.Rminus1_cl_level,
                "Rminus1_single_split": self.Rminus1_single_split,
                "oversample": self.oversample,
                "drag": self.drag,
                "drag_limits": self.drag_limits,
            },
        }

        # And the overall info - this 
        info = {
            "likelihood": likelihood_info,
            "params": params_info,
            "sampler": sampler_info,
            "prior": external_priors,
        }


        return info

    def output_row(self, row):
        out_row = [row[pname] for pname in self.cobaya_param_names+self.cobaya_derived_names]
        out_row.append(row['weight'])
        out_row.append(row['minuslogprior'])
        out_row.append(row['minuslogpost'])
        self.output.parameters(out_row)


    def resume(self):
        raise NotImplemented("resuming cobaya")

    def execute(self):
        self.n_saved = 0
        info = self.make_cobaya_info()
        self.cobaya_run(info)
        self.done = True

    def worker(self):
        self.execute()

    def is_converged(self):
        return self.done




def make_slow_mock_likelihood(params):
    function_template = """
def slow_like({}):
    return 0.0
    """
    pnames = [str(p).replace("-",'_') for p in params]
    params_text = ", ".join(str(p) for p in pnames)
    function_text = function_template.format(params_text)
    d = locals()
    exec(function_text, globals(), d)
    print(function_text)
    return d['slow_like']

def make_main_likelihood(params, derived):
    function_template = """
def likelihood({}, _derived=[{}]):
    global sampler
    _full_params_input = np.zeros({})
    {}
    like, derived = sampler.pipeline.likelihood(_full_params_input)
    {}
    if not sampler.pipeline.quiet:
        print(_full_params_input, like)
    return like
    """
    pnames = [str(p).replace("-",'_') for p in params]
    dnames = [str(p).replace("-",'_') for p in derived]
    derived_text = ','.join(['"{}"'.format(dname) for dname in dnames])
    params_text = ", ".join(str(p) for p in pnames)
    length_text = len(params)
    set_text = "\n    ".join(["_full_params_input[{}] = {}".format(i,p) for i,p in enumerate(pnames)])
    set_derived = "\n    ".join(["_derived['{}'] = derived[{}]".format(dname, i) for i,dname in enumerate(dnames)])
    function_text = function_template.format(params_text, derived_text, length_text, set_text, set_derived)
    d=locals()
    exec(function_text, globals(), d)
    print(function_text)
    return d['likelihood']



