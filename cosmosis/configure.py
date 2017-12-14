import os
import sys
from . import compilers

if '--debug' in sys.argv:
    common_flags = "-O3 -g -fPIC"


dirname = os.path.split(__file__)[0]
cosmosis_src_dir =  os.path.abspath(os.path.join(dirname, os.path.pardir))



commands = """
export COSMOSIS_SRC_DIR={cosmosis_src_dir}
export C_INCLUDE_PATH=$C_INCLUDE_PATH:{cosmosis_src_dir}/cosmosis/datablock
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH:{cosmosis_src_dir}/cosmosis/datablock
export LIBRARY_PATH=$LIBRARY_PATH:{cosmosis_src_dir}/cosmosis/datablock
export COSMOSIS_ALT_COMPILERS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{cosmosis_src_dir}/cosmosis/datablock
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:{cosmosis_src_dir}/cosmosis/datablock
""".format(**locals())

commands += compilers.compilers

if __name__ == '__main__':
    print(commands)


# export GSL_LIB=/usr/local/lib
# export GSL_INC=/usr/local/include
# export FFTW_LIBRARY=/usr/local/lib
# export FFTW_INCLUDE_DIR=/usr/local/include
# export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH 
# export LAPACK_LINK=-framework accelerate
# export LAPACK_LINK="-framework accelerate"
# export CFITSIO_INC=/usr/local/incldue
# export CFITSIO_INC=/usr/local/include
# export CFITSIO_LIB=/usr/local/lib
# export COSMOSIS_SRC_DIR=../../
# export COSMOSIS_SRC_DIR=../../cosmosis
# export COSMOSIS_SRC_DIR=../../../cosmosis
# export COSMOSIS_SRC_DIR=/Users/jaz/src/cosmosis-standalone/env-test/lib/python2.7/site-packages/cosmosis-0.0.0-py2.7.egg
# export C_INCLUDE_PATH=$C_INCLUDE_PATH:/Users/jaz/src/cosmosis-standalone/env-test/lib/python2.7/site-packages/cosmosis-0.0.0-py2.7.egg/cosmosis/datablock
# export LIBRARY_PATH=$LIBRARY_PATH:/Users/jaz/src/cosmosis-standalone/env-test/lib/python2.7/site-packages/cosmosis-0.0.0-py2.7.egg/cosmosis/datablock
# export COSMOSIS_ALT_COMPILERS=1
# export LIBRARY_PATH=""
# export C_INCLUDE_PATH=""
