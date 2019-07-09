Tutorial 4: Building new pipelines
----------------------------------

In CosmoSIS a *pipeline* is a sequence of *modules* that computes one or more *likelihoods*.

You can build a new pipeline or modify an existing one by choosing which modules to include in the pipeline, and by selecting their configuration.

Extending a calculation by adding modules
=========================================

To know which module to add you have to understand the calculation you want to perform, and make sure that all the necessary calculations for it are done at some point in the pipeline.
You can add modules into the middle of an existing pipeline, or at the end.

Adding the new module
=====================

In the :code:`[pipeline]` section of :code:`demos/demo2.ini` you will find::

    [pipeline]
    modules = consistency camb planck bicep2


Let's modify this pipeline, by removing the old BICEP2 likelihood, and adding a more recent and accurate one, the BOSS DR12 Baryon Acoustic Oscillation measurement.  We can consult the `list of standard library modules <https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules/>`_. on the CosmoSIS wiki to find out what we will need.

The `BOSS DR12 page <https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules/boss_dr12_1607.03155v1>`_ gives some basic details about the likelihood

It tells us the file we need to use for the likelihood::

    File: cosmosis-standard-library/likelihood/boss_dr12/boss_dr12.py

This tells us to make a new section in the parameter file that we are using, with this information in.  We can add new text to the bottom of :code:`demos/demo2.ini`, like this::

    [boss]
    file = cosmosis-standard-library/likelihood/boss_dr12/boss_dr12.py

Configuring the module
======================

The wiki page also tells us what parameters we can use to configure the pipeline, and what inputs the likelihood will need.  The only mandatory parameter is described like this::

    mode    0 for BAO only, 1 for BAO+FS measurements

Let's use both the BAO and full-shape information, so we can set it to 1.  So our new parameter file section becomes this::

    [boss]
    file = cosmosis-standard-library/likelihood/boss_dr12/boss_dr12.py
    mode = 1

Running the module
==================


Right now, nothing will change if we run this pipeline, because we have not told CosmoSIS to use this new module.  We can do so by changing the modules option from above to this::

    [pipeline]
    modules = consistency camb planck boss

This tells CosmoSIS to look for a section called "boss" in the parameter file and configure a module based on it.

It's fine to include unused modules in the parameter file - it can be useful later when you run different variations of a similar pipeline.

The missing growth function
===========================

If we run this pipeline with :code:`cosmosis demos/demo2.ini` then we will get this error::

    cosmosis.datablock.cosmosis_py.errors.BlockSectionNotFound: 3: Could not find section called growth_parameters (name was z)

This is because the logic of our pipeline didn't add up - we need the growth rate of cosmic structure to calculate this likelihood, but we never calculated it.  In fact the wiki page above showed us a table of inputs we needed for the pipeline, but never supplied.

A quick search of the `list of standard modules <https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules>`_ shows us several modules that calculate the growth factor.  They make different assumptions about physics, so which we should use will depend on our science case.  In this case, let's use the one called `growth_factor <https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules/extract_factor_1>`_ which solves the growth factor differential equation to calculate it.  You'll notice that the missing :code:`growth_parameters` section is listed as one of the outputs it generates.

So let's add this module to the bottom of the parameter file::

    [growth]
    file = cosmosis-standard-library/structure/extract_growth/extract_growth.py

and change our pipeline like this::

    [pipeline]
    modules = consistency camb planck growth boss

The total likelihood
========================


There's one final change we need to make - we need to tell CosmoSIS to add the BOSS likelihood to the total::

    likelihoods = planck2015 boss_dr12

(in fact if we had just left out the likelihoods line altogether CosmoSIS would have done this by default.  It's only because we explicitly listed planck that the issue arose).

If we do this and run then the pipeline will calculate both our likelihoods, and their total::

        Likelihood planck2015 = -1726.9467027368746
        Likelihood boss_dr12 = -4.313138812597072
    Likelihood total = -1731.2598415494717

So we have successfully extended our pipeline!
