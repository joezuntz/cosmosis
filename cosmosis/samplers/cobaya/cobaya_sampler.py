from builtins import zip
from builtins import range
from builtins import str
from .. import ParallelSampler
import numpy as np
import sys
from ...runtime import prior as priors
from collections import OrderedDict


class CobayaSampler(ParallelSampler):
    parallel_output = True
    supports_resume = False
    sampler_outputs = [("weight", float), ("prior", float), ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline
        import cobaya.run
        self.cobaya_run = cobaya.run.run
        self.done = False



    def resume(self):
        raise NotImplemented("resuming cobaya")

    def execute(self):
        if self.pipeline.do_fast_slow:
            fast_params = pipeline.fast_params
            slow_params = pipeline.slow_params
        else:
            fast_params = []
            slow_params = pipeline.varied_params

        derived_params = ["{}__{}".format(*d).lower() for d in pipeline.extra_saves]
        info = make_cobaya_info(fast_params, slow_params, derived_params)
        print("Info:", info)
        results_info, results = self.cobaya_run(info)        
        sample = results['sample']
        pnames = [str(p).replace("--",'__') for p in pipeline.varied_params]
        for i in range(sample.n()):
            row = sample[i]
            out_row = [row[pname] for pname in pnames+derived_params]
            out_row.append(row['weight'])
            out_row.append(row['minuslogprior'])
            out_row.append(row['minuslogpost'])
            self.output.parameters(out_row)
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
    global pipeline
    _full_params_input = np.zeros({})
    {}
    like, derived = pipeline.likelihood(_full_params_input)
    {}
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


def make_cobaya_info(fast_params, slow_params, derived_params):
    # fast_params: list of param objects
    # slow_params: list of param objects
    likelihood = make_main_likelihood(slow_params + fast_params, derived_params)
    mock_slow_likelihood = make_slow_mock_likelihood(slow_params)
    # Copied from your example
    info = {"likelihood": {"total": likelihood, "mock":mock_slow_likelihood}}
    pdict = OrderedDict()
    for p in slow_params + fast_params:
        pname = str(p).replace('-','_')
        if isinstance(p.prior, priors.UniformPrior):
            pdict[pname] = {
                "ref": p.start,
                "prior": {
                    "min": p.prior.a,
                    "max": p.prior.b,
                },
                "proposal": 0.05*(p.prior.b-p.prior.a),
            }
        elif isinstance(p.prior, priors.GaussianPrior):
            pdict[pname] = {
                "ref": p.start,
                "prior": {
                    "dist": "norm",
                    "loc": p.prior.mu,
                    "scale": p.prior.sigma,
                },
                "proposal": p.prior.sigma,
            }
    for d in derived_params:
        pdict[d] = None


    info["params"] = pdict
    info["sampler"]= {"mcmc": None}

    return info