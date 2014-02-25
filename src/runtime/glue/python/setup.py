from distutils.core import setup, Extension
import os
try:
    cfitsio_include = os.environ['CFITSIO_INC']
    cfitsio_lib = os.environ['CFITSIO_LIB']
except KeyError:
    import sys
    sys.stderr.write("Must specify environment variables CFITSIO_INC and CFITSIO_LIB to use desglue setup.py\n")
    sys.exit(1)

if 'LDFLAGS' in os.environ:
    extra_link_args=[os.environ['LDFLAGS']]
else:
    extra_link_args=[]

desglue = Extension('desglue',
					include_dirs = [cfitsio_include, '../C','python'],
					library_dirs=["../lib",cfitsio_lib],
					libraries = ['cfitsio','cfitswrap','ini'],
                    extra_link_args=extra_link_args,
                    sources = ['pydesglue.c'])

setup (name = 'pydesglue',
       version = '1.0',
       description = 'DES Glue Code',
       packages = ['pydesglue'],
       #py_modules = ['pydesglue'],#, 'termcolor','IncludingConfigParser'],
       ext_modules = [desglue])
