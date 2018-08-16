from setuptools import setup, find_packages, Extension
from distutils.command.install import install
from distutils.command.build import build
import os
import sys

version = '0.0.8'

f90_mods = [
    "cosmosis/datablock/cosmosis_section_names.mod",
    "cosmosis/datablock/cosmosis_types.mod",
    "cosmosis/datablock/cosmosis_wrappers.mod",
    "cosmosis/datablock/cosmosis_modules.mod",
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

def compile_library():
    cosmosis_src_dir = os.getcwd()
    os.chdir('cosmosis/datablock/')
    try:
        status = os.system("COSMOSIS_ALT_COMPILERS=1 COSMOSIS_SRC_DIR={} make".format(cosmosis_src_dir))
    finally:
        os.chdir('../../')
    if status:
        raise RuntimeError("Failed to compile cosmosis core")

def setup_compilers():
    try:
        cc = os.environ['CC']
        fc = os.environ['FC']
        cxx = os.environ['CXX']
    except KeyError:
        sys.stderr.write("\n")
        sys.stderr.write("    For the avoidance of later problems you need to set\n")
        sys.stderr.write("    these environment variables before installing cosmosis:\n")
        sys.stderr.write("    CC, FC, CXX for the C compiler, fortran compiler, and C++ compiler.\n\n")
        sys.stderr.write("    Your compilers need to be recent enough to compile cosmosis.\n\n")
        sys.stderr.write("\n")
        sys.exit(1)
    f = open("./cosmosis/compilers.py", "w")
    f.write("compilers = '''\n".format(cc))
    f.write("export CC={}\n".format(cc))
    f.write("export FC={}\n".format(fc))
    f.write("export CXX={}\n".format(cxx))
    f.write("export COSMOSIS_ALT_COMPILERS=1\n")
    f.write("'''\n")
    f.close()

def check_compilers():
    pass

class my_build(build):
    def run(self):
        check_compilers()
        setup_compilers()
        compile_library()
        build.run(self)


class my_install(install):
    def run(self):
        check_compilers()
        setup_compilers()
        compile_library()
        install.run(self)



if __name__ == "__main__":
    setup(name = 'cosmosis-standalone',
          description       = "A testbed stand-alone installation of the CosmoSIS project. Not ready for primetime!",
          author            = "Joe Zuntz",
          author_email      = "joezuntz@googlemail.com",
          packages = find_packages(),
          package_data = {"" : ["datablock/libcosmosis.so",] + c_headers + cc_headers + f90_mods},
          scripts = scripts,
          install_requires = ['pyyaml', 'future', 'configparser', 'emcee', 'numpy', 'scipy'],
          cmdclass={"install":my_install, "build":my_build},
          version=version,
          )

