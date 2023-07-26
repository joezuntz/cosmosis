from ..campaign import *
from ..runtime import Inifile

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