This version of CosmoSIS is not the right one to use.  Yet.

Please use the bitbucket version for now:
https://bitbucket.org/joezuntz/cosmosis

Documentation
-------------

You can find full documentation at https://cosmosis.readthedocs.io


Installation
------------

Full CosmoSIS + Standard Library
================================


```
conda create -p ./conda-env -c conda-forge cosmosis cosmosis-build-standard-library
conda activate ./conda-env

source cosmosis-configure
git clone https://github.com/joezuntz/cosmosis-standard-library
cd cosmosis-standard-library
make
```



Sampling tools only
===================

Most cosmology users will want the full CosmoSIS + Standard Library package.  See the previous section.

If you only want the CosmoSIS sampling tools and framework, then you can simply run:

```
pip install cosmosis
```

If you have problems then we recommend installing using Conda:

```
conda create -p ./conda-env -c conda-forge cosmosis
conda activate ./conda-env
```
