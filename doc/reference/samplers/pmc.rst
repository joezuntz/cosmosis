The Pmc sampler
------------------

Adaptive Importance Sampling

===========  =======================================
Name         pmc
Version      1.0
Author(s)    CosmoSIS Team
URL          https://bitbucket.org/joezuntz/cosmosis
Citation(s)  MNRAS 405.4 2381-2390 (2010)
Parallelism  embarrassing
===========  =======================================

Population Monte-Carlo uses importance sampling with an initial  distribution that is gradually adapted as more samples are taken and their likelihood found.

At each iteration some specified number of samples are drawn from a mixed Gaussian distribution. Their posteriors are then evaluated and importance weights calculated.  This approximate distribution is then used to update the Gaussian mixture model so that it more closely mirrors the underlying distribution.

Components are dropped if they are found not to be necessary.

This is a python re-implementation of the CosmoPMC alogorithm in the  cited paper.



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
    * - components
      - integer
      - 5
      - Number of components in the Gaussian mixture
    * - iterations
      - integer
      - 30
      - Number of iterations (importance updates) of PMC
    * - student
      - boolean
      - F
      - Do not use this.  It is a not yet functional attempt to use a Student t mixture.
    * - final_samples
      - integer
      - 5000
      - Samples to take after the updating of the mixture is complete
    * - samples_per_iteration
      - integer
      - 1000
      - Number of samples per iteration of PMC
    * - nu
      - float
      - 2.0
      - Do not use this.  It is the nu parameter for the non-function Student t mode.

