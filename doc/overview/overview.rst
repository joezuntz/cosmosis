CosmoSIS Overview
-----------------



What CosmoSIS does
==================

CosmoSIS connects together **samplers**, which decide how to explore a cosmological parameter space, with **pipelines** made from a sequence of **modules**, which calculate the steps needed to get a likelihood functions. You have to decide on what calculations your likelihood function should consist of and choose or write modules that perform them.

Here is an example schematic of a CosmoSIS run of a weak lensing analysis:

.. image:: /images/cosmosis-architecture.png


* The green sampler generates parameters and sends them to the pipeline. At the end it gets a total likelihood back.
* The blue modules are independent codes run in the numbered sequence.  They each perform one step in the calculation of the likelihood.
* The purples connections show each module reading inputs from the datablock, and then saving their results back to it.
* The yellow DataBlock acts like a big lookup table for data.  It stores the initial parameters and then the results from each module.  
* The blue Likelihood module (number 11) has a special output - it computes a final likelihood value.


Modules
=======

For many cosmological models, calculating the theoretical prediction of a model is a long and complicated process with many steps.  CosmoSIS organizes each of those steps as a separate piece of code, called a module.  The CosmoSIS Standard Library comes with a large collection of popular cosmology codes packaged as modules, and it is easy to make your own modules.

For example, we package the Boltzmann code CAMB as a module - it takes in cosmological parameters as inputs and calculates cosmological power spectra as outputs.

To become a CosmoSIS module, a piece of code in python, C, C++, or Fortran just needs to have two specially named functions in it: :code:`setup` and :code:`execute`.  The setup function is run once, a the start of the CosmoSIS process.  The execute function is run again each time there are new input parameters for which the sampler wants the likelihood.

.. image:: /images/cosmosis-connections.png

DataBlocks
===========

CosmoSIS modules do not send their calculations on to each other directly.  Instead they communicate only with CosmoSIS, via a DataBlock.  A DataBlock is a look-up table that takes a pair of strings as keys (a section and a name) and maps them to a value, which can be either a single scalar value (integer, real, or complex number, or a string) or a (multi-dimensional) array of them.


Samplers
========

In CosmoSIS a `sampler` is a code that generates a series of sample sets of parameters on which the likelihood must be run.  Some of these are trivial, like the `test` sampler, which just generates a single sample.  Others are much more complex and explore the parameter space dynamically.  Most of them are what statisticians would genuinely consider to be a *sampler* - they generate samples from the posterior distribution.

There are many different samplers packaged with the code that are useful in different circumstances. They are all interact with the rest of CosmoSIS in very similar ways - they are configured from options in a parameter file, and they build a likelihood pipeline which they run whenever they have a new sample whose posterior they want to evaluate.


User Interface
================

CosmoSIS decides what analysis to perform using three configuration files that you write (one of them optional).  Theyt are all in the "ini" format, and are:

 * The parameter file, which describes the pipeline of the model you want to run and the sampler you want to run on it.  The :code:`setup` functions for modules look in this file.
 * The values file, which describes the numerical input parameters, some of which will probably be varied throughout the run. The samplers decide how to vary the parameter within the ranges it gives and modules look for these parameters in their :code:`execute` functions.
 * The priors file, which optionally describes additional priors on the input parameters.  There is always an implicit Uniform prior on the value; adding a prior here creates an additional prior.  If you do not set a priors file then there will be no additional priors.

When you execute CosmoSIS you tell it the parameter file on the command line.  The parameter file tells CosmoSIS where to find the other two files.

The "ini" format splits a file into sections, which are named using square brackets, and keys and values within those sections.  For example::

    [section_name]
    parameter_name = value

Values can be integers, doubles, complex, or strings.
