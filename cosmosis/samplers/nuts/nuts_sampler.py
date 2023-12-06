from ...runtime import logs
from .. import ParallelSampler
import numpy as np

def total_derivative(block, top_sec, top_name, bottom_sec, bottom_name, cache):
    if (top_sec, top_name, bottom_sec, bottom_name) in cache:
        return cache[(top_sec, top_name, bottom_sec, bottom_name)]
    if block.has_derivative(top_sec, top_name, bottom_sec, bottom_name):
        return block.get_derivative(top_sec, top_name, bottom_sec, bottom_name)
    if top_sec == bottom_sec and top_name == bottom_name:
        return np.atleast_2d(1)
    t = 0
    found = False
    for next_sec, next_name, d in block.get_derivatives(top_sec, top_name):
        x = total_derivative(block, next_sec, next_name, bottom_sec, bottom_name, cache)
        t += d @ x
        found = True
    if not found:
        t = np.atleast_2d(0.0)
    cache[(top_sec, top_name, bottom_sec, bottom_name)] = t
    return t
            
            
def transform_from_unconstrained(q, bounds):
    """
    q is the parameters in the transformed space, all defined from -inf .. inf

    bounds 
    """
    u = np.exp(q) / ( 1 + np.exp(q))
    x = np.array([b[0] + (b[1] - b[0])* ui for ui, b in zip(u, bounds)])
    return x

def transform_to_unconstrained(x, bounds):
    """
    x is the parameters in the original space

    bounds 
    """
    u = np.array([(xi - b[0]) / (b[1] - b[0]) for xi, b in zip(x, bounds)])
    q = np.log(u / (1 - u))
    return q

def log_jacobians_from_unconstrained(q, bounds):
    return np.array([
        np.log(b[1] - b[0]) + qi - 2 * np.log(1 + np.exp(qi))
    for qi, b in zip(q, bounds)] )


def total_derivatives(block, pipeline):
    nparam = len(nuts_pipeline.varied_params)
    cache = {}
    grad = np.zeros(nparam)
    for i, p in enumerate(pipeline.varied_params):
        for like_name in pipeline.likelihood_names:
            grad[i] += total_derivative(block, "likelihoods", like_name + "_like", p.section, p.name, cache)
    return grad

def log_prior_derivatives(p, pipe):
    """
    Derivatives of the log_priors of the parameters
    """
    return np.array([par.log_prior_derivative(p_i) for (par, p_i) in zip(pipe.varied_params, p)])


def posterior_and_gradient(q):
    nparam = nuts_pipeline.nvaried
    bounds = [p.limits for p in nuts_pipeline.varied_params]
    widths = np.array([b[1] - b[0] for b in bounds])
    x = transform_from_unconstrained(q, bounds)

    r = nuts_pipeline.run_results(x)
    if r.block is None:
        grad = np.repeat(np.nan, nparam)
        return -np.inf, grad

    # Get the posterior and gradient in the original space
    logpx = r.post
    dlogpx_dx = total_derivatives(r.block, nuts_pipeline) + log_prior_derivatives(x, nuts_pipeline)

    # transform the posterior and gradient to the unconstrained space
    logpq = logpx + log_jacobians_from_unconstrained(q, bounds).sum()
    eq = np.exp(q)
    logpq_grad = widths * eq / (1 + eq)**2 * dlogpx_dx + 1 - 2 * eq / (1 + eq)

    return logpq, logpq_grad

    


class NUTSSampler(ParallelSampler):
    parallel_output = False
    supports_resume = False
    sampler_outputs = [("post", float)]

    def config(self):
        from . nuts import PinNUTS
        global nuts_pipeline
        nuts_pipeline = self.pipeline

        self.samples = self.read_ini("nsample", int, 1000)
        self.nstep = self.read_ini("nstep", int, 500)
        self.nadapt = self.read_ini("nadapt", int, 1000)
        self.target_accept = self.read_ini("target_accept", float, 0.6)
        # self.samples_generated = 0
        # self.epsilon = None
        self.iterations = 0
        self.p0 = transform_to_unconstrained(self.pipeline.start_vector(), [p.limits for p in self.pipeline.varied_params])
        self.sampler = PinNUTS(posterior_and_gradient, self.pipeline.nvaried)

    def resume(self):
        if self.output.resumed:
            # Just load the final position from the output file
            # and the epsilon from the 
            pass


    def execute(self):
        from . nuts import PinNUTS
        start = self.iterations
        end = start + self.nstep
        if start > self.nadapt:
            n_adapt = 0
            n_draw = self.nstep
        elif end > self.nadapt:
            n_adapt = self.nadapt - start
            n_draw = self.nstep - n_adapt
        else:
            n_adapt = self.nstep
            n_draw = 0
        logs.overview(f"Running NUTS with {n_adapt} adaptive samples and {n_draw} regular samples")

        samples, probs = self.sampler.sample(self.p0, n_draw, n_adapt)
        self.p0 = samples[-1]
        if n_draw:
            for params, post in zip(samples,probs):
                x = transform_from_unconstrained(params, [p.limits for p in self.pipeline.varied_params])
                self.output.parameters(x, post)


        self.iterations += self.nstep
        



    def is_converged(self):
        return self.iterations >= self.samples + self.nadapt
