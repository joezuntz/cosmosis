from cosmosis.runtime.config import Inifile
from cosmosis.samplers.sampler import Sampler
import cosmosis.samplers.minuit.minuit_sampler
from cosmosis.runtime.pipeline import LikelihoodPipeline
from cosmosis.output.in_memory_output import InMemoryOutput
import tempfile
import os
import sys
import pytest
import numpy as np

minuit_compiled = os.path.exists(cosmosis.samplers.minuit.minuit_sampler.libname)

def run(sampler, check_prior, check_extra=True, **options):

    sampler_class = Sampler.registry[sampler]

    values = tempfile.NamedTemporaryFile('w')
    values.write(
        "[parameters]\n"
        "p1=-3.0  0.0  3.0\n"
        "p2=-3.0  0.0  3.0\n")
    values.flush()

    override = {
        ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
        ("pipeline", "debug"): "F",
        ("pipeline", "quiet"): "T",
        ("pipeline", "modules"): "test1",
        ("pipeline", "extra_output"): "parameters/p3",
        ("pipeline", "values"): values.name,
        ("test1", "file"): "test_module.py",
    }

    for k,v in options.items():
        override[(sampler,k)] = str(v)


    ini = Inifile(None, override=override)

    # Make the pipeline itself
    pipeline = LikelihoodPipeline(ini)

    output = InMemoryOutput()
    sampler = sampler_class(ini, pipeline, output)
    sampler.config()


    while not sampler.is_converged():
        sampler.execute()

    if check_prior:
        pr = np.array(output['prior'])
        # a few samples might be outside the bounds and have zero prior
        assert np.all((pr==-3.58351893845611)|(pr==-np.inf))
        # but not all of them
        assert not np.all(pr==-np.inf)

    if check_extra:
        p1 = output['parameters--p1']
        p2 = output['parameters--p2']
        p3 = output['PARAMETERS--P3']
        assert np.all((p1+p2==p3)|(np.isnan(p3)))

    return output


def test_apriori():
    run('apriori', True, nsample=100)

def test_dynesty():
    # dynesty does not support extra params
    run('dynesty', False, check_extra=False, nlive=50, sample='unif')

def test_emcee():
    run('emcee', True, walkers=8, samples=100)

def test_fisher():
    run('fisher', False, check_extra=False)

def test_grid():
    run('grid', True, nsample_dimension=10)

def test_gridmax():
    run('gridmax', True, max_iterations=1000)

# def test_kombine():
#     run('kombine')

def test_maxlike():
    run('maxlike', True)

def test_metropolis():
    run('metropolis', True, samples=20)

@pytest.mark.skipif(not minuit_compiled,reason="requires Minuit2")
def test_minuit():
    run('minuit', True)

def test_multinest():
    run('multinest', True, max_iterations=10000, live_points=50, feedback=False)

def test_pmaxlike():
    run('pmaxlike', True)

def test_pmc():
    old_settings = np.seterr(invalid='ignore', divide='ignore')
    run('pmc', True, iterations=10)
    np.seterr(**old_settings)  

def test_zeus():
    run('zeus', True, maxiter=100_000, walkers=10, samples=100, nsteps=50)
    run('zeus', True, maxiter=100_000, walkers=10, samples=100, nsteps=50, verbose=True)
    run('zeus', True, maxiter=100_000, walkers=10, samples=100, nsteps=50, tune=False)
    run('zeus', True, maxiter=100_000, walkers=10, samples=100, nsteps=50, tolerance=0.1)
    run('zeus', True, maxiter=100_000, walkers=10, samples=100, nsteps=50, patience=5000)
    run('zeus', True, maxiter=100_000, walkers=10, samples=100, nsteps=50, moves="differential:2.0  global")
    run('zeus', True, maxiter=50_000, walkers=10, samples=100, nsteps=50)

def test_polychord():
    with tempfile.TemporaryDirectory() as base_dir:
        run('polychord', True, live_points=20, feedback=0, base_dir=base_dir, polychord_outfile_root='pc')

def test_pymc():
    if sys.version_info.major==2:
        run('pymc', False)
    else:
        pytest.skip("Skipping pymc test on python 3")

def test_snake():
        run('snake', True)

def test_star():
        run('star', False)

def test_test():
    run('test', False)

if __name__ == '__main__':
    import sys
    locals()[sys.argv[1]]()