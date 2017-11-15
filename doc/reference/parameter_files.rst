CosmoSIS Parameter Files
========================

Parameter files
---------------

CosmoSIS uses three configuration files to describe an analysis:

* The parameter file defines the sampler, output, and pipeline.
* The values file defines the input parameters and their allowed ranges.
* The priors file defines additional priors on the parameters. 

Details about these files are given on these sub-pages.  This page describes common features of the ini format that they all use.


.. toctree::
   :maxdepth: 1

   parameter_files/params_ini
   parameter_files/values_ini
   parameter_files/priors_ini


The ini format
--------------

All the CosmoSIS parameter files use the standard :code:`ini` format, with a few additional features.  In this format the file is divided into named sections surrounded by square brackets, and has named parameters in each section::

    [section_name]
    param1 = 123
    param2 = 1.4e14
    param3 = potato potato

    [another_section_name]
    param1 = abc

The parameter name and section name can be any ascii string.  CosmoSIS will try to interpret the parameter value first as an integer, then it will fall back to a double, then as a boolean, and finally if all else fails it will assume it is a string.  Different sections can store parameters with the same name; they are completely separate.

True / False values can be specified using "T"/"F" or "Y"/"N".

You can "re-open" a section later to specify more parameters if you want, for example::

    [section1]
    a = 1
    b = 2

    [section2]
    x = 4.5

    [section1]
    b = 3
    c = 10.

If you specify a parameter twice then the second one will overwrite the first one, so in the example above the "section1" parameter "b" will have the value 3.

Case
-----

CosmoSIS ini file parameter names are CASE-INSENSITIVE.  The parameter "x" is the same as "X".
The entries can be case-sensitive depending how you write your modules.

Comments
---------

Comments in ini files can be marked with a semi-colon :code:`; anything after this is a comment`.

You can also use hashes :code:`#`, but we recommend sticking with semi-colons for consistency.


Include statements
---------------------

The first feature that CosmoSIS adds to the :code:`ini` format is that it allows files to include other files, so that you can have nested parameter files.  For example, you might have one basic parameter file and a number of small variants.

You can tell a parameter file to include another file like this::

    %include /path/to/other_file.ini

This has the effect of "pasting" in the other file into the current one, so if you :code:`%include` file "A" at the start of file "B" then file B's parameters will take precedence and any repeated options will overwrite "A", whereas if you include it at the end file A will take precedence.

The path is looked up relative to the current working directory, not to the first parameter file.


Environment variables
---------------------

A second feature that CosmoSIS adds to ini files is the use of environment variables.
You can use environment variables within parameter files with this syntax::

    [my_likelihood]
    data_file = ${DATAFILE}

Then if before you run CosmoSIS you write this in the (bash) terminal::

    export DATAFILE=my_data_file.dat

Then it will replace DATAFILE with this value in the parameters.

If the environment variable :code:`$DATAFILE` is not set when you run the code then no replacement will be done and your pipeline will look for a presumably non-existent file called :code:`$DATAFILE`.

Interpolation
-------------

An in-built feature of ini files is called interpolation - parameters in the file can reference other parameters using this syntax::

    [section]
    name=xxxx
    data_dir: %(name)s/data

Then the parameter :code:`data_dir` will have the value :code:`xxxx/data`.  This only works within the same section, or using the default section described below.

The default section
-------------------

If you include a section called :code:`[DEFAULT]` then the code will fall back to that section if a particular option isn't found elsewhere.  This can be particularly useful in combination with the interpolation feature described above.

For example::

    [DEFAULT]
    NAME = v1

    [output]
    filename = %(NAME)s.txt

    [my_module]
    model=model-%(NAME)s

Would make the code use the value "v1" for the parameter "model" in the "my_module" section, and output the chain to "v1.txt" and use the parameter :code:`model=model-v1` in the my_module section.  

Looking up the parameter "NAME" in the "my_module" section would also give the value :code:`v1`.



