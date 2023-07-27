from ..campaign import *
from ..runtime import Inifile
import tempfile
import yaml
import contextlib

@contextlib.contextmanager
def run_from_source_dir():
    this_dir = os.path.split(os.path.abspath(__file__))[0]
    source_dir = os.path.join(this_dir, "..", "..")
    cur_dir = os.getcwd()
    try:
        os.chdir(source_dir)
        yield
    finally:
        os.chdir(cur_dir)


def test_pipeline_after():
    params = Inifile(None)
    params.add_section("pipeline")
    params.set("pipeline", "modules", "a b c")
    pipeline_after(params, "a", "d")
    assert params.get("pipeline", "modules") == "a d b c"
    pipeline_after(params, "c", ["e", "f"])
    assert params.get("pipeline", "modules") == "a d b c e f"

def test_pipeline_before():
    params = Inifile(None)
    params.add_section("pipeline")
    params.set("pipeline", "modules", "a b c")
    pipeline_before(params, "a", "d")
    assert params.get("pipeline", "modules") == "d a b c"
    pipeline_before(params, "c", ["e", "f"])
    assert params.get("pipeline", "modules") == "d a b e f c"

def test_pipeline_replace():
    params = Inifile(None)
    params.add_section("pipeline")
    params.set("pipeline", "modules", "a b c")
    pipeline_replace(params, "b", "d")
    assert params.get("pipeline", "modules") == "a d c"
    pipeline_replace(params, "c", ["e", "f"])
    assert params.get("pipeline", "modules") == "a d e f"

def test_pipeline_delete():
    params = Inifile(None)
    params.add_section("pipeline")
    params.set("pipeline", "modules", "a b c")
    pipeline_delete(params, "b")
    assert params.get("pipeline", "modules") == "a c"
    pipeline_delete(params, ["a", "c"])
    assert params.get("pipeline", "modules") == ""

def test_pipeline_append():
    params = Inifile(None)
    params.add_section("pipeline")
    params.set("pipeline", "modules", "a b c")
    pipeline_append(params, ["d"])
    assert params["pipeline", "modules"] == "a b c d"

def test_pipeline_prepend():
    params = Inifile(None)
    params.add_section("pipeline")
    params.set("pipeline", "modules", "a b c")
    pipeline_prepend(params, ["d", "e"])
    assert params["pipeline", "modules"] == "d e a b c"


def test_campaign_functions():
    with run_from_source_dir():
        runs = parse_yaml_run_file("cosmosis/test/campaign.yml")

        assert len(runs) == 4
        assert "v1" in runs
        assert runs["v2"]["values"].get("parameters", "p1") == "-2.0 0.0 2.0"
        assert runs["v2"]["priors"].get("parameters", "p2") == "gaussian 0.0 1.0"
        assert runs["v3"]["params"].get("runtime", "sampler") == "emcee"
        assert runs["v3"]["params"].get("emcee", "walkers") == "8"

        assert not runs["v4"]["priors"].has_option("parameters", "p2")

        for name in runs:
            print(name)

        show_run(runs["v1"])
        perform_test_run(runs["v1"])
        show_run_status(runs)
        show_run_status(runs, ["v1"])
        show_run_status(runs, ["v2"])
        perform_test_run(runs["v2"])
        show_run_status(runs, ["v1"])
        show_run_status(runs, ["v2"])
        launch_run(runs["v2"])
        show_run_status(runs, ["v1"])
        show_run_status(runs, ["v2"])

def test_campaign_functions2():
    with run_from_source_dir():
        with open("cosmosis/test/campaign.yml") as f:
            runs_config = yaml.safe_load(f)

        with tempfile.TemporaryDirectory() as dirname:
            runs_config['output_dir']  = dirname
            runs = parse_yaml_run_file(runs_config)

            for name in runs:
                print(name)


            assert len(runs) == 4
            assert "v1" in runs
            assert runs["v2"]["values"].get("parameters", "p1") == "-2.0 0.0 2.0"
            assert runs["v2"]["priors"].get("parameters", "p2") == "gaussian 0.0 1.0"
            assert runs["v3"]["params"].get("runtime", "sampler") == "emcee"
            assert runs["v3"]["params"].get("emcee", "walkers") == "8"

            assert not runs["v4"]["priors"].has_option("parameters", "p2")

            show_run(runs["v1"])
            perform_test_run(runs["v1"])
            show_run_status(runs)
            show_run_status(runs, ["v1"])
            show_run_status(runs, ["v2"])
            perform_test_run(runs["v2"])
            show_run_status(runs, ["v1"])
            show_run_status(runs, ["v2"])
            launch_run(runs["v2"])
            show_run_status(runs, ["v1"])
            show_run_status(runs, ["v2"])
