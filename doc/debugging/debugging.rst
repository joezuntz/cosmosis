Debugging
----------

CosmoSIS has a number of tools to help you debug problems with your analyses.


The Test Sampler
================

The first thing to try when debugging an MCMC failure is the test sampler - try to see if you can recreate the problem when running on a single parameter set.  Then you can switch on the options below for more information about the error.

You can set the parameter::

    [test]
    fatal_errors = T

to get more output on any error in the test sampler.

Parameter file options
======================

You can set several options in the CosmoSIS parameter file to find problems more easily::

    [pipeline]
    quiet=F
    debug=T

Setting :code:`quiet=F` will print when each stage is reached so you can tell where errors happen.

Setting :code:`debug=T` will, after an error, print out a log of all the values saved to the CosmoSIS data block - if you are getting an error saying that some value has not been found in the block then this can be useful to figure out why.

Note that this setting prints out a lot of text *after* any other errors are reported, so you may need to scroll up past it to look for any other messages.

Python errors
=============

If you are finding a crash in a Python function then you can use the :code:`--pdb` flag on the command line to enter the `Python Debugger <https://docs.python.org/2/library/pdb.html#debugger-commands//>`_. When the program crashes you will be dropped into an interactive debugger shell and you can print the values of parameters or run functions to try to determine the source of the error::

    cosmosis my_params.ini --pdb

To do this in the test sampler you may need to set the parameter :code:`fatal_errors=T`  in the :code:`[test]` section of the parameter file.

This option doesn't work when running in parallel.


Compiling in debug mode
=======================

You can compile all CosmoSIS code in debug mode to make finding errors in compiled code easier.  To do so, run these commands in the main directory::

    export COSMOSIS_DEBUG=1
    make clean
    make

Crashes in C/C++/Fortran code
=============================

If you are seeing a crash in C/C++/Fortran code you can use the command line flag :code:`--experimental-fault-handling` to try to track it down.  If you are using python 2 you first need to install a package to support this::

    pip install faulthandler # python 2 only; it comes with python3

Then::

    cosmosis my_params.ini --experimental-fault-handling

On a crash this should print out a traceback revealing in which function the error ocurred.


Special Samplers
================

CosmoSIS has several samplers that explore the parameter space in ways that help debug or compare to other code.

The *apriori* sampler samples points from throughout the prior.   It can be useful if your error is only triggered by extreme parameters.

The *list* sampler takes a list of parameter sets to evaluate one by one.  It can be useful if you have a set of possible parameters that might cause errors, or which you want to compare to another code.

The *star* sampler samples a set of 1D parameters slices in each parameter direction through the central point.  It can be useful to compare to the output of another code.
