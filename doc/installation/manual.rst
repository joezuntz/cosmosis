Manual Installation
-------------------

We recommend using one of the automatic installation methods described above.  They install everything neatly in a single directory and will not mess up your current installation of anything.  Even if you're an expert it really is much easier.

If you can't or don't want to use the bootstrap method then you can manually install CosmoSIS.

Library dependencies
======================

You need to install several libraries and programs before installing CosmoSIS:

* `CAMB <http://camb.info/>`_
* `gcc/g++/gfortran 4.8 <https://gcc.gnu.org/>`_ or above
* `gsl 1.16 <http://ftpmirror.gnu.org/gsl/>`_ or above
* `cfitsio 3.30 <http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html>`_ or above
* `FFTW 3 <http://www.fftw.org/download.html>`_
* `lapack <http://www.netlib.org/lapack/#_lapack_version_3_5_0>`_ (except on MacOS)
* `git <https://git-scm.com/downloads>`_
* `python 2.7 <https://www.python.org/downloads/release/python-2710/>`_

Python dependencies
======================

You also need to install several python packages before installing CosmoSIS.  These are also listed in `config/requirements.txt`:

* astropy
* Cython
* matplotlib
* numpy
* PyYAML
* scikit-learn
* scipy
* CosmoloPy
* emcee
* fitsio
* kombine


Download
======================

Download CosmoSIS and the standard library using git::

    git clone http://bitbucket.org/joezuntz/cosmosis
    cd cosmosis
    git clone http://bitbucket.org/joezuntz/cosmosis-standard-library
    cd cosmosis-standard-library
    cd ..


Setup script
======================

From the cosmosis directory make a copy of the manual setup script::

    cp config/manual-install-setup setup-cosmosis

**Edit the new file setup-my-cosmosis** and replace all the places where it says :code:`/path/to/XXX` in this file with correct paths based on how you installed things.

Build
======================

Source the setup script and make::

    source setup-cosmosis
    make

If you get any errors then check your setup-cosmosis script for errors, and whether there are any environment variables like :code:`LD_LIBRARY_PATH` set before you start. 

Usage
======================

Each time you start a new terminal shell then you need to repeat this step::

    source setup-cosmosis

Then test your install is working by :doc:`following our first tutorial </tutorials/tutorials/tutorial1>`.
