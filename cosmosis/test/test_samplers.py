from cosmosis import Inifile, Sampler, LikelihoodPipeline, InMemoryOutput, TextColumnOutput
from cosmosis.postprocessing import postprocessor_for_sampler
import tempfile
import os
import sys
import pytest
import numpy as np
from astropy.table import Table


minuit_compiled = os.path.exists(Sampler.get_sampler("minuit").libminuit_name)

def run(name, check_prior, check_extra=True, can_postprocess=True, do_truth=False, no_extra=False, pp_extra=True, pp_2d=True, **options):

    sampler_class = Sampler.registry[name]

    values = tempfile.NamedTemporaryFile('w')
    values.write(
        "[parameters]\n"
        "p1=-3.0  0.0  3.0\n"
        "p2=-3.0  0.0  3.0\n")
    values.flush()

    override = {
        ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
        ("pipeline", "debug"): "F",
        ("pipeline", "modules"): "test1",
        ("pipeline", "extra_output"): "parameters/p3",
        ("pipeline", "values"): values.name,
        ("test1", "file"): "example_module.py",
    }

    for k,v in options.items():
        override[(name,k)] = str(v)

    if no_extra:
        del override[("pipeline", "extra_output")]


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

    if check_extra and not no_extra:
        p1 = output['parameters--p1']
        p2 = output['parameters--p2']
        p3 = output['PARAMETERS--P3']
        assert np.all((p1+p2==p3)|(np.isnan(p3)))

    if can_postprocess:
        pp_class = postprocessor_for_sampler(name)
        print(pp_class)
        with tempfile.TemporaryDirectory() as dirname:
            truth_file = values.name if do_truth else None
            pp = pp_class(output, "Chain", 0, outdir=dirname, prefix=name, truth=truth_file, fatal_errors=True)
            pp_files = pp.run()
            pp.finalize()
            pp.save()
            for p in pp_files:
                print(p)
            postprocess_files = ['parameters--p1', 'parameters--p2']
            if pp_2d:
                postprocess_files.append('2D_parameters--p2_parameters--p1')
            if check_extra and pp_extra and not no_extra:
                postprocess_files.append('parameters--p3')
                if pp_2d:
                    postprocess_files += ['2D_parameters--p3_parameters--p2', '2D_parameters--p3_parameters--p1']
            for p in postprocess_files:
                filename = f"{dirname}{os.path.sep}{name}_{p}.png"
                print("WANT ", filename)
                assert filename in pp_files
                assert os.path.exists(filename)
            for p in os.listdir(dirname):
                p = os.path.join(dirname, p)
                if p.endswith(".txt"):
                    Table.read(p, format='ascii.commented_header')
                    print(f"Read file {p} as a table")


    return output


def test_apriori():
    run('apriori', True, can_postprocess=False, nsample=100)

def test_dynesty():
    # dynesty does not support extra params
    run('dynesty', False, check_extra=False, nlive=50, sample='unif')

def test_emcee():
    run('emcee', True, walkers=8, samples=100)
    run('emcee', True, walkers=8, samples=100, a=3.0)

def test_truth():
    run('emcee', True, walkers=8, samples=100, do_truth=True)

def test_fisher():
    run('fisher', False, check_extra=False)

def test_grid():
    run('grid', True, pp_extra=False, nsample_dimension=10)

def test_gridmax():
    run('gridmax', True, can_postprocess=False, max_iterations=1000)

# def test_kombine():
#     run('kombine')

def test_maxlike():
    run('maxlike', True, can_postprocess=False)

def test_metropolis():
    run('metropolis', True, samples=20)
    run('metropolis', True, samples=20, covmat_sample_start=True)

@pytest.mark.skipif(not minuit_compiled,reason="requires Minuit2")
def test_minuit():
    run('minuit', True, can_postprocess=False)

def test_multinest():
    run('multinest', True, max_iterations=10000, live_points=50, feedback=False)

def test_pmaxlike():
    run('pmaxlike', True, can_postprocess=False)

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

def test_snake():
        run('snake', True, pp_extra=False)

def test_nautilus():
    run('nautilus', True)
    run('nautilus', True, n_live=500, enlarge_per_dim=1.05,
        split_threshold=95., n_networks=3, n_batch=50, verbose=True, f_live=0.02, n_shell=100)


def test_star():
        run('star', False, pp_extra=False, pp_2d=False)

def test_test():
    run('test', False, can_postprocess=False)

def test_list_sampler():
    # test that the burn and thin parameters work
    # make a mock output file with some samples
    with tempfile.TemporaryDirectory() as dirname:

        input_chain_filename = os.path.join(dirname, "input_chain.txt")
        input_chain = TextColumnOutput(input_chain_filename)
        input_chain.add_column('parameters--p1', float)
        input_chain.add_column('parameters--p2', float)
        nrow = 100
        data = np.random.normal(size=(nrow, 2)).clip(-2.99, 2.99)
        for i in range(nrow):
            input_chain.parameters(data[i, 0], data[i, 1])
        input_chain.close()

        burn = 0.2
        thin = 2

        data = data[20::thin]

        output = run("list", True, can_postprocess=False, filename=input_chain_filename, burn=burn, thin=thin)
        p1 = output['parameters--p1']
        p2 = output['parameters--p2']
        p3 = p1 + p2
        expected_like = -(p1**2 + p2**2)/2.
        assert p1.size == 40
        assert np.allclose(p1, data[:, 0])
        assert np.allclose(p2, data[:, 1])
        assert np.allclose(output['PARAMETERS--P3'], p3)
        assert np.allclose(output["post"] - output["prior"], expected_like)

@pytest.mark.skipif(sys.version_info < (3,7), reason="pocomc requires python3.6+")
def test_poco():
    try:
        import pocomc
    except ImportError:
        pytest.skip("pocomc not installed")
    run('pocomc', True, check_extra=False, n_effective=32, n_active=16,  n_total=32, n_evidence=32, )

if __name__ == '__main__':
    import sys
    locals()[sys.argv[1]]()