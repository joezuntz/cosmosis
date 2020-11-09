from cosmosis.datablock import DataBlock
import numpy as np
import tempfile
import os
import yaml

def test_string_array():
    block = DataBlock()
    # 1 input as list
    strings = "I am a mole and I live in a hole".split()
    print(strings)
    block.put_string_array_1d("my_section", "my_key", strings)
    strings2 = block.get_string_array_1d("my_section", "my_key")
    print(strings2)

    assert (strings == strings2).all()
    # 2 input as array
    block.put_string_array_1d("my_section2", "my_key2", np.array(strings))
    strings3 = block.get_string_array_1d("my_section2", "my_key2")

    assert (strings == strings3).all()
    print(strings3)


def test_string_array_getset():
    block = DataBlock()
    # 1 input as list
    strings = "I am a mole and I live in a hole".split()
    block["my_section", "my_key"] = strings

    strings2 = block.get_string_array_1d("my_section", "my_key")
    assert (strings == strings2).all()

    strings3 = block["my_section", "my_key"]


def test_string_array_save():
    block = DataBlock()
    # 1 input as list
    strings = "I am a mole and I live in a hole".split()
    block["my_section", "my_key"] = strings

    with tempfile.TemporaryDirectory() as dirname:
        dn = os.path.join(dirname, "dir")
        fn = os.path.join(dirname, "tar")
        yn = os.path.join(dirname, "block.yml")
        block.save_to_directory(dn)
        block.save_to_directory(fn)
        block.to_yaml(yn)

        strings2 = np.loadtxt(f"{dn}/my_section/my_key.txt", dtype=str)
        assert (strings == strings2).all()

        strings3 = yaml.safe_load(open(yn))["my_section"]["my_key"]
        assert (strings == strings3)





if __name__ == '__main__':
    # test_string_array()
    test_string_array_save()    