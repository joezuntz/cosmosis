The List sampler
------------------

Re-run existing chain samples

===========  =======================================
Name         list
Version      1.0
Author(s)    CosmoSIS Team
URL          https://bitbucket.org/joezuntz/cosmosis
Citation(s)
Parallelism  embarrassing
===========  =======================================

This is perhaps the second simplest sampler - it simply takes all its samples from a list in a file and runs them all with the new pipeline.

This could probably be replaced with an importance sampler, and may be merged into it in future.



Installation
============

No special installation required; everything is packaged with CosmoSIS




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Parameter
      - Type
      - Meaning
      - Default
    * - save
      - string
      - (empty)
      - if present the base-name to save the cosmology output from each sample
    * - filename
      - string
      - 
      - cosmosis-format chain of input samples

