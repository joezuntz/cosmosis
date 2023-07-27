from cosmosis.runtime.config import Inifile
from cosmosis.samplers.sampler import Sampler
import cosmosis.samplers.minuit.minuit_sampler
from cosmosis.runtime.pipeline import LikelihoodPipeline
from cosmosis.output.in_memory_output import InMemoryOutput
from cosmosis.runtime.handler import activate_segfault_handling
import tempfile
import os
import sys
import pytest
import numpy as np


activate_segfault_handling()

minuit_compiled = os.path.exists(cosmosis.samplers.minuit.minuit_sampler.libname)

def run(sampler, check_prior, check_extra=True, fast_slow=False, **options):

    sampler_class = Sampler.registry[sampler]

    values = tempfile.NamedTemporaryFile('w')
    values.write(
        "[parameters]\n"
        "p1=-3.0  0.0  3.0\n"
        "p2=-3.0  0.0  3.0\n"
        "p4=-3.0  0.0  3.0\n")
    values.flush()

    override = {
        ('runtime', 'root'): os.path.split(os.path.abspath(__file__))[0],
        ("pipeline", "debug"): "F",
        ("pipeline", "modules"): "test1  test3 test4",
        ("pipeline", "extra_output"): "parameters/p3",
        ("pipeline", "values"): values.name,
        ("pipeline", "fast_slow"): str(fast_slow)[0],
        ("pipeline", "first_fast_module"): "test4",
        ("test1", "file"): "example_module.py",
        ("test3", "file"): "example_module3.py",
        ("test4", "file"): "example_module4.py",
    }

    for k,v in options.items():
        override[(sampler,k)] = str(v)


    ini = Inifile(None, override=override)

    # Make the pipeline itself
    pipeline = LikelihoodPipeline(ini)

    if pipeline.do_fast_slow:
        pipeline.setup_fast_subspaces()

    output = InMemoryOutput()
    sampler = sampler_class(ini, pipeline, output)
    sampler.config()


    while not sampler.is_converged():
        sampler.execute()

    if check_prior:
        pr = np.array(output['prior'])
        # but not all of them
        assert not np.all(pr==-np.inf)

    if check_extra:
        p1 = output['parameters--p1']
        p2 = output['parameters--p2']
        p3 = output['PARAMETERS--P3']
        assert np.all((p1+p2==p3)|(np.isnan(p3)))

    return output


def test_polychord_hang():
    # with tempfile.TemporaryDirectory() as base_dir:
    #     run('polychord', True, live_points=20, feedback=0, base_dir=base_dir, polychord_outfile_root='pc')

    # Test for issue # https://bitbucket.org/joezuntz/cosmosis/issues/382/samplers-will-hang-if-fast_fraction-is-non
    with tempfile.TemporaryDirectory() as base_dir:
        run('polychord', True, live_points=20, feedback=0, base_dir=base_dir, polychord_outfile_root='pc', fast_slow=True, fast_fraction=0.5)


if __name__ == '__main__':
    import sys
    locals()[sys.argv[1]]()