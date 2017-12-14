#from numpy.distutils.core import Extension
#from distutils.core import Extension
from setuptools import setup, find_packages, Extension
import os
import sys

cc_files = [
    "cosmosis/datablock/c_datablock.cc",
    "cosmosis/datablock/datablock_logging.cc",
    "cosmosis/datablock/entry.cc",
    "cosmosis/datablock/section.cc",
    "cosmosis/datablock/datablock.cc",
]


f90_files = [
    "cosmosis/datablock/cosmosis_section_names.F90",
    "cosmosis/datablock/cosmosis_types.F90",
    "cosmosis/datablock/cosmosis_wrappers.F90",
    "cosmosis/datablock/cosmosis_modules.F90",
]

scripts = [
    'bin/cosmosis',
    'bin/cosmosis-py2',
    'bin/cosmosis-py3',
    'bin/cosmosis-ini-from-output',
    'bin/cosmosis-sample-fisher',
    'bin/postprocess',
    'bin/postprocess-py3',
    'bin/postprocess-py3',
]

c_headers = [
    "cosmosis/datablock/c_datablock.h",
    "cosmosis/datablock/datablock_logging.h",
    "cosmosis/datablock/datablock_types.h",
    "cosmosis/datablock/cosmosis_constants.h",
    "cosmosis/datablock/datablock_status.h",
    "cosmosis/datablock/section_names.h",
]

cc_headers = [
    "cosmosis/datablock/clamp.hh",
    "cosmosis/datablock/entry.hh",
    "cosmosis/datablock/fakearray.hh",
    "cosmosis/datablock/ndarray.hh",
    "cosmosis/datablock/datablock.hh",
    "cosmosis/datablock/exceptions.hh",
    "cosmosis/datablock/mdarraygen.hh",
    "cosmosis/datablock/section.hh"
]

if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')


f90_objects = [f[:-4]+".o" for f in f90_files]
f90_headers = [f[:-4]+".mod" for f in f90_files]

def compile_library():
    os.chdir('cosmosis/datablock/')
    try:
        os.system("make")
    finally:
        os.chdir('../../')


compile_library()



if __name__ == "__main__":
    setup(name = 'cosmosis',
          description       = "Joe Test",
          author            = "Joe Zuntz",
          author_email      = "joezuntz@googlemail.com",
          packages = find_packages(),
          include_package_data = True,
          scripts = scripts,
          )

