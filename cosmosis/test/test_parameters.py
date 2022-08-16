from ..runtime import parameter
import numpy as np
import tempfile


def test_override_new():
    with tempfile.NamedTemporaryFile('w') as values:
        values.write(
            "[parameters]\n"
            "p1=-3.0  0.0  3.0\n"
            "p2=-3.0  0.0  3.0\n"
            "p3=20.0\n")
        values.flush()
        print(values.name)
        params = parameter.Parameter.load_parameters(values.name, override=None)

    assert params[0] == ("parameters", "p1")
    assert params[0].start == 0.0
    assert np.allclose(params[0].limits, [-3, 3])

    assert params[1] == ("parameters", "p2")
    assert params[1].start == 0.0

    assert params[2] == ("parameters", "p3")
    assert np.isclose(params[2].start, 20.0)


    with tempfile.NamedTemporaryFile('w') as values:
        values.write(
            "[parameters]\n"
            "p1=-3.0  0.0  3.0\n"
            "p2=-3.0  0.0  3.0\n"
            "p3=20.0\n")
        values.flush()
        override = {
            ("parameters", "p1"): "1.0",
            ("xxx", "aaa"): "100.0",
        }
        params = parameter.Parameter.load_parameters(values.name, override=override)
    

    assert params[0] == ("parameters", "p1")
    assert params[0].start == 1.0
    assert np.allclose(params[0].limits, 1.0)

    assert params[1] == ("parameters", "p2")
    assert params[1].start == 0.0
    assert np.allclose(params[1].limits, [-3, 3])

    assert params[2] == ("parameters", "p3")
    assert np.isclose(params[2].start, 20.0)

    print(params)
    assert params[3] == ("xxx", "aaa")
    assert np.isclose(params[3].start, 100.0)
    assert np.allclose(params[3].limits, 100.0)

