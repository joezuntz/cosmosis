import cosmosis
import pytest
import tempfile

def test_missing_module():
    with pytest.raises(cosmosis.runtime.SetupError) as error_info:
        m = cosmosis.Module("pretend", "./this_file_does_not_exist.py")

    assert "file does not exist" in str(error_info.value)


def test_wrong_path_type():
    with tempfile.NamedTemporaryFile(suffix=".x") as f:
        with pytest.raises(cosmosis.runtime.SetupError) as error_info:
            m = cosmosis.Module("pretend", f.name)

    assert "do not know what kind of module this is" in str(error_info.value)
