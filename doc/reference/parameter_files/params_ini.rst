Parameter Files
===============


Options that are set in the parameter file but never used will just be silently ignored, so check your spelling carefully.

This can be very useful if you have some options that you only switch on sometimes but not others.


Runtime Options
-----------------

A section called :code:`[runtime]` must be present in the parameter file.  It has one mandatory parameter, :code:`sampler`, and one optional one, :code:`root`.

The :code:`sampler` parameter must be one of the CosmoSIS samplers.  It determines how the parameter space is explored.

The :code:`root` parameter changes how paths to modules in the parameter file are looked for.  All paths are defined relative to the root.  By default this is just the CosmoSIS directory::

    [runtime]
    sampler = metropolis
    ; This is the default value of root:
    root = ${COSMOSIS_SRC_DIR}


Sampler Options
-----------------

The specific sampler that you chose in the :code:`[runtime]` section must also be configured in a section 

Different samplers have different options - some mandatory and some which take a default value if you don't set them. The pages for the individual samplers describe what parameter they can take.  For example, the emcee sampler might take these parameters::

    [emcee]
    walkers = 64
    samples = 400
    nsteps = 100


Output Options
-----------------

All the samplers except the test sampler generate output chain files. You can choose where this should go and what format it should have in the :code:`output` section.   The only mandatory parameter is :code:`filename`::

    [output]
    filename = my_output_chain.txt
    ; the default format is "text".  It can also be set to "fits".
    format = text
    verbosity = noisy


The verbosity parameter can be set to any of these values::

    highest
    debug
    noisy
    standard
    gentle
    quiet
    silent

Or a value from 0 (silent) to 50 (highest).
The verbosity settings in the code are a little mixed up right now, and the biggest effect can actually be had by modifying the "quiet" parameter described below in the :code:`[pipeline]` section.  This will be addressed in future releases.


Pipeline Options
-----------------

The mandatory :code:`pipeline` section defines the pipeline that you want to run to generate a likelihood.

It must have these parameters::

    [pipeline]
    modules = consistency camb jla riess11
    values = demos/values5.ini
    likelihoods = jla riess
    quiet=T
    debug=F
    timing=F
    extra_output = cosmological_parameters/Yhe

The **modules** parameter must contain a sequence of the modules that make up the pipeline.  Each module is a single collection of code that performs a piece of the calculation.  See also the overview section for more information on the concept modules, the list of CosmoSIS standard modules, and details on how to make your own modules.  The names used in this option can be absolutely anything, as long as there is a correspondingly named section somewhere else in the file (see Module Options below).

The **values** parameter must point to the values file, which defines what parameters are put in to the start of the pipeline.  The path is relative to the current working directory, not to the parameter file.

The **likelihoods** parameter defines which likelihoods are extracted from the pipeline.  After running the pipeline CosmoSIS looks in the data block for parameters called :code:`X_LIKE` for each X in the list given here. If this parameter is not set then all available likelihoods are used.

The **quiet** parameter tells the code to minimize the output it prints out if it is set to "T".

The **debug** parameter can be set to "T" to print out more detailed debugging output if the pipeline fails.  It will print out the complete list of all things written to and read from the pipeline.

The **timing** parameter can be set to "T" to print out how long each stage in the pipeline takes to configure and run.

The **extra_output** parameter can be used to save derived parameters in the output chain.  Set it to a sequence of strings each of the form :code:`section/name`, and after the pipeline is run each these will be extracted from the output as a new column.


Module Options
-----------------

Every entry found in the :code:`modules` option in the :code:`pipeline` section (see above) must have a section with the same name in the parameter file.

That section must have at least one mandatory parameter in it, :code:`file`:

.. code-block:: ini

    [pipeline]
    ; We must include two sections below with these names:
    modules = my_theory_module   my_likelihood_module

    [my_theory_module]
    file = modules/path/to/theory/filename.so

    [my_likelihood_module]
    file = modules/path/to/likelihood/filename.py


The file option must be the path to either a shared library (.so) or a python (.py) file.  The paths that you need for CosmoSIS standard library modules are described in the reference section for them.  See the documentation on making modules for more information on creating your own new modules.

In addition to this mandatory parameter, you can also specify other options in the file.  These options can be read in the setup phase of the module::

    [my_likelihood_module]
    file = modules/path/to/likelihood/filename.py
    data_file = some_path_to_a_data_file.dat
    xxx = 1


