Tutorial 5: Writing new modules
---------------------------------

So far we've used modules from the CosmoSIS standard library to build our pipeline.  This works fine for many projects, where using or modifying the standard library will be enough.  But we can go further by creating entirely new modules to calculate new observables or include new physics effects in them.


Pipelines & Modules
-------------------

CosmoSIS modules are isolated from each other - all the calculations done by a module are stored in a DataBlock, which is passed through the list of modules in the pipeline.  Each module reads what previous modules have done, performs its own calculations, and then adds these to the pipeline.

Typically, a pipeline is run many times on different sets of input parameters, for an example in an MCMC.

Module requirements
-------------------

CosmoSIS modules can be written in C, C++, Fortran, Python, or (experimentally) Julia.

In each of these languages the structure of the file you write is the same: you write functions with specific names.  Two of these are required, and one is optional::

    setup(options)
    # Takes the information from the parameter file as a DataBlock and configures the module.
    # This is called once, when the pipeline is created.

    execute(block, config)
    # Takes a DataBlock (see below) from the values file and any preivous pipeline, and runs the module.
    # This is called for each set of parameters the pipeline is run on.

    cleanup(config)
    # (Optional) frees memory allocated in the setup function
    # This is run once, when the pipeline is deallocated.
    # We will skip it in this tutorial, as it is rarely needed in python.


Writing our new module
----------------------

Let's write our new module in Python, since that's usually much easier.

Make a new file.  You can put it anywhere, but to keep things organized it's usally better to keep your work separate from the existing standard library.

In that file, put these lines, which represent an empty but runnable CosmoSIS module::

    from cosmosis.datablock import names, option_section

    def setup(options):
        return {}

    def execute(block, config):
        return 0

