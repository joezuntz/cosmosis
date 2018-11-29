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
    print("Called back!  R-1 = ", mcmc.Rminus1_last)
    print("output things now using.", sampler, mcmc)
    start = sampler.n_saved
    end = mcmc.collection.n()
    if end>start:
        print("\n\nCosmosis saving {}-{}\n\n".format(start,end))
        samples = mcmc.collection.data[start:end]
        print("Len = ", len(samples))
        for i,row in samples.iterrows():
            sampler.output_row(row)
        sampler.output.flush()
    sampler.n_saved = end

class CobayaSampler(ParallelSampler):
    parallel_output = True
    supports_resume = False
    sampler_outputs = [("weight", float), ("prior", float), ("post", float)]

    def config(self):
        import cobaya.run

        # For MPI - this needs to be a global variable
        global sampler
        sampler = self

        self.cobaya_run = cobaya.run.run
        self.done = False
        

        default_drag = self.pipeline.do_fast_slow
        self.oversample = self.read_ini("oversample", bool, False)
        self.drag = self.read_ini("drag", bool, default_drag)

        if self.oversample and self.drag:
            raise ValueError("Can only set one oversample=T and drag=T for cobaya")
        if self.oversample or self.drag and not self.pipeline.do_fast_slow:
            raise ValueError("Need to set fast_slow=T in [pipeline] to use oversampling or dragging")
        


    def make_cobaya_info(self):

        varied_params = self.pipeline.varied_params
        if self.pipeline.do_fast_slow:
            fast_params = self.pipeline.fast_params
            slow_params = self.pipeline.slow_params
        else:
            fast_params = []
            slow_params = varied_params

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
            "total": {"external": likelihood, "speed":1},
            "mock": {"external": mock_slow_likelihood, "speed":0},
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
                "oversample": self.oversample,
                "drag": self.drag,
                "callback_function": callback
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
    exec(function_text)
    print(function_text)
    return locals()['slow_like']

def make_main_likelihood(params, derived):
    function_template = """
def likelihood({}, _derived=[{}]):
    global sampler
    _full_params_input = np.zeros({})
    {}
    like, derived = sampler.pipeline.likelihood(_full_params_input)
    {}
    #print(_full_params_input, like)
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
    exec(function_text)
    print(function_text)
    return locals()['likelihood']



