from setuptools import setup, find_packages, Extension
from distutils.command.install import install
from distutils.command.build import build
from distutils.command.clean import clean
import os
import sys

version = '0.0.8'

f90_mods = [
    "datablock/cosmosis_section_names.mod",
    "datablock/cosmosis_types.mod",
    "datablock/cosmosis_wrappers.mod",
    "datablock/cosmosis_modules.mod",
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
    "datablock/c_datablock.h",
    "datablock/datablock_logging.h",
    "datablock/datablock_types.h",
    "datablock/cosmosis_constants.h",
    "datablock/datablock_status.h",
    "datablock/section_names.h",
]

cc_headers = [
    "datablock/clamp.hh",
    "datablock/entry.hh",
    "datablock/fakearray.hh",
    "datablock/ndarray.hh",
    "datablock/datablock.hh",
    "datablock/exceptions.hh",
    "datablock/mdarraygen.hh",
    "datablock/section.hh"
]

datablock_libs = ["datablock/libcosmosis.so"]

sampler_libs = ["samplers/multinest/multinest_src/libnest3.so",
                "samplers/multinest/multinest_src/libnest3_mpi.so",
                "samplers/polychord/polychord_src/libchord.so",
                "samplers/polychord/polychord_src/libchord_mpi.so",
                "samplers/minuit/minuit_wrapper.so"]

runtime_libs = ["runtime/experimental_fault_handler.so"]

compilers_config = ["compilers.mk"]

if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')

def compile_library():
    cosmosis_src_dir = os.getcwd()
    os.chdir('cosmosis/')
    try:
        status = os.system("COSMOSIS_ALT_COMPILERS=1 COSMOSIS_SRC_DIR={} make".format(cosmosis_src_dir))
    finally:
        os.chdir('../')
    if status:
        raise RuntimeError("Failed to compile cosmosis core")

def clean_library():
    cosmosis_src_dir = os.getcwd()
    os.chdir('cosmosis/')
    try:
        status = os.system("COSMOSIS_ALT_COMPILERS=1 COSMOSIS_SRC_DIR={} make clean".format(cosmosis_src_dir))
    finally:
        os.chdir('../')
    if status:
        raise RuntimeError("Failed to make clean")

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
    def __init__(self, dist):
        install.__init__(self, dist)
        self.build_args = {}
        if self.record is None:
            self.record = "install-record.txt"

    def run(self):
        check_compilers()
        setup_compilers()
        compile_library()
        install.run(self)

class my_clean(clean):
    def run(self):
        clean_library()
        clean.run(self)

if __name__ == "__main__":
    setup(name = 'cosmosis-standalone',
          description       = "A testbed stand-alone installation of the CosmoSIS project. Not ready for primetime!",
          author            = "Joe Zuntz",
          author_email      = "joezuntz@googlemail.com",
          packages = find_packages(),
          package_data = {"" : datablock_libs + sampler_libs + runtime_libs 
                             + c_headers + cc_headers + f90_mods 
                             + compilers_config,},
          scripts = scripts,
          install_requires = ['pyyaml', 'future', 'configparser', 'emcee', 'numpy', 'scipy'],
          cmdclass={"install"   : my_install,
                    "build"     : my_build,
                    "clean"     : my_clean},
          version=version,
          )

