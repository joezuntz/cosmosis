This version of CosmoSIS is not the right one to use.  Yet.

Please use the bitbucket version for now:
https://bitbucket.org/joezuntz/cosmosis

Documentation
=============


You can find full documentation at https://cosmosis.readthedocs.io


Installation
=============

Full CosmoSIS + Standard Library
--------------------------------

We strongly recommend installing CosmoSIS and the standard library using Conda, which will take care of everything for you:

```
# Installs CosmoSIS core and dependencies
conda create -p ./conda-env -c conda-forge cosmosis cosmosis-build-standard-library
conda activate ./conda-env

# Downloads the standard library
git clone https://github.com/joezuntz/cosmosis-standard-library
cd cosmosis-standard-library

# You should run this whenever you use CosmoSIS
source cosmosis-configure

# Build the library
make
```



Sampling tools only
-------------------

Most cosmology users will want the full CosmoSIS + Standard Library package.  See the previous section.

If you only want the CosmoSIS sampling tools and framework, then you can simply run:

```
pip install cosmosis
```

You can customize this with the environment variables:
- `CC` (C compiler)
- `CXX` (C++ compiler)
- `FC` (Fortran compiler)
- `MPIFC` (Fortran compiler)
- `USER_CFLAGS` (C compilation flags)
- `USER_CXXFLAGS` (C++ compilation flags)
- `USER_FFLAGS` (Fortran compilation flags)
- `LAPACK_LINK` (Lapack linking flags flags)
- `COSMOSIS_OMP` (set to 1 for OpenMP)
- `MINUIT2_LIB` (link directory if you need the Minuit2 library; if you're not sure then you don't)
- `MINUIT2_INC` (header directory if you need the Minuit2 library)

If you have problems then we recommend installing using Conda:

```
conda create -p ./conda-env -c conda-forge cosmosis
conda activate ./conda-env
```

