Tutorial 1: Computing a single cosmology likelihood
---------------------------------------------------

This tutorial shows you how to build a pipeline to evaluate a single cosmology likelihood, in this case the Planck satellite's 2015 measurements of the cosmic microwave background. 

Installation
============

Before you start, :doc:`follow the instructions to install cosmosis </installation/installation>`.  Once you are complete, you should be able to run::

    cosmois --help

and see a usage message.

Parameter files
============================

Have a look at the file :code:`demos/demo2.ini`.  CosmoSIS is always run on a single parameter file like this.  It specifies how to construct a pipeline that generates a likelihood, and what to do with that likelihood once it is calculated.  

CosmoSIS parameter files use the :code:`ini` format, which has section names with square brackets around them and parameters specified with an equals sign.  This example says there is a section called :code:`runtime` with a parameter named :code:`sampler` with the values "test"::


    [runtime]
    sampler = test

Each CosmoSIS run uses at least two ini files, this one, called a parameter file, and a second *values* file, specifying the cosmological and other varied parameters used in the pipeline.  In this case the values file is :code:`demos/values2.ini`.

Running CosmoSIS on a parameter file
=====================================


Run CosmoSIS on this parameter file with this command::

    cosmosis demos/demo2.ini

You will see a lot of output showing:

* What parameters are used in this pipeline, e.g. ::

    Parameter Priors
    ----------------
    planck--a_planck                      ~ delta(1.0)
    cosmological_parameters--h0           ~ delta(0.6726)
    cosmological_parameters--omega_m      ~ delta(0.3141)


* The set-up phase for each step (module) in the calculation, e.g.::

    -- Setting up module camb --
     camb mode  =            1
     camb cmb_lmax =         2650
     camb FeedbackLevel =            2
     accuracy boost =    1.1000000000000001     
     HighAccuracyDefault =  T


* The sampler that is being run::

    ****************************
    * Running sampler 1/1: test
    ****************************

* The output of each module, e.g.::

    Running camb ...
    Reion redshift       =  11.751
    Integrated opt depth =  0.0800
    Om_b h^2             =  0.018096
    Om_c h^2             =  0.123356
    Om_nu h^2            =  0.000644

Defining a sampler
===================

The first lines in the parameter file :code:`demos/demo2.ini` are::

    [runtime]
    sampler = test

    [test]
    save_dir=demo_output_2
    fatal_errors=T

The first option, :code:`sampler`, tells CosmoSIS what it should do with the likelihood that we will construct - how the parameter space should be *sampled*.

CosmoSIS has lots of different samplers in it, designed to move around parameter spaces in different ways.  The :code:`test` sampler is the simplest possible one: it doesn't move around the parameter space at all - it just computes a likelihood (runs the pipeline) for a single set of values.  These tutorials will discuss several samplers; the full list is described in :doc:`the samplers page </reference/samplers/samplers>`.

Once you have chosen a sampler you configure that sampler with the second section shown in above, which has the name of the sampler, in this case :code:`test`.

Defining a pipeline
===================

Defining input values
======================
