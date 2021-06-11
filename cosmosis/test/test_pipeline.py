from cosmosis.runtime import Inifile, register_new_parameter, LikelihoodPipeline, Parameter, Module
from cosmosis.datablock import DataBlock
from cosmosis.samplers.sampler import Sampler
from cosmosis.runtime.prior import TruncatedGaussianPrior, DeltaFunctionPrior
from cosmosis.output.in_memory_output import InMemoryOutput
import numpy as np
import os
import tempfile

root = os.path.split(os.path.abspath(__file__))[0]

def test_add_param():

    sampler_class = Sampler.registry["emcee"]

    values = tempfile.NamedTemporaryFile('w')
    values.write(
        "[parameters]\n"
        "p1=-3.0  0.0  3.0\n"
        "p2=-3.0  0.0  3.0\n")
    values.flush()

    override = {
        ('runtime', 'root'): root,
        ("pipeline", "debug"): "F",
        ("pipeline", "quiet"): "T",
        ("pipeline", "modules"): "test2",
        ("pipeline", "values"): values.name,
        ("test2", "file"): "test_module2.py",
        ("emcee", "walkers"): "8",
        ("emcee", "samples"): "10"
    }

    ini = Inifile(None, override=override)

    # Make the pipeline itself
    pipeline = LikelihoodPipeline(ini)

    # test the the new added parameter has worked
    assert len(pipeline.varied_params) == 3
    p = pipeline.varied_params[2]
    assert str(p) == "new_parameters--p3"
    assert isinstance(p, Parameter)
    assert isinstance(p.prior, TruncatedGaussianPrior)
    assert np.isclose(p.prior.mu, 0.1)
    assert np.isclose(p.prior.sigma, 0.2)

    assert len(pipeline.fixed_params) == 1
    p = pipeline.fixed_params[0]
    assert isinstance(p, Parameter)
    assert isinstance(p.prior, DeltaFunctionPrior)


    output = InMemoryOutput()
    sampler = sampler_class(ini, pipeline, output)
    sampler.config()


    # check that the output is working
    assert output.column_index_for_name("new_parameters--p3") == 2

    while not sampler.is_converged():
        sampler.execute()


    p1 = output['parameters--p1']
    p2 = output['parameters--p2']
    p3 = output['new_parameters--p3']
    assert p3.max() < 1.0
    assert p3.min() > -1.0

    return output

def test_missing_setup():
    # check the register_new_parameter feature when no
    # setup is currently happening
    module = Module("test2", root + "/test_module2.py")
    config = DataBlock()
    module.setup(config)

def test_unused_param_warning(capsys):
    # check that an appropriate warning is generated
    # when a parameter is unused
    module = Module("test", root + "/test_module.py")
    config = DataBlock()
    config['test', 'unused'] = "unused_parameter"
    module.setup(config)
    out, _ = capsys.readouterr()
    assert "**** WARNING: Parameter 'unused'" in out



if __name__ == '__main__':
    test_add_param()
    test_missing_setup()
