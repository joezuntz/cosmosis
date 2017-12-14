import os
import sys

if '--debug' in sys.argv:
    common_flags = "-O3 -g -fPIC"


dirname = os.path.split(__file__)[0]
cosmosis_src_dir =  os.path.abspath(os.path.join(dirname, os.path.pardir))



commands = """

export COSMOSIS_SRC_DIR={cosmosis_src_dir}
export C_INCLUDE_PATH=$C_INCLUDE_PATH:{cosmosis_src_dir}
export LIBRARY_PATH=$LIBRARY_PATH:{cosmosis_src_dir}
""".format(**locals())


if __name__ == '__main__':
    print(commands)
