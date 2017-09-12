
Samplers
--------

Samplers are the different methods that CosmoSIS uses to choose points in parameter spaces to evaluate.

Some are designed to actually explore likelihood spaces; others are useful for testing and understanding likelihoods.

Samplers for testing pipelines
===============================

.. toctree::
    :maxdepth: 1

     test: Evaluate a single parameter set <test>
     star: Posteriors along each dimension through a single point <star>
     list: Re-run existing chain samples <list>


Classic samplers
===============================

.. toctree::
    :maxdepth: 1

     metropolis: Classic Metropolis-Hastings sampling <metropolis>
     emcee: Ensemble walker sampling <emcee>
     importance: Importance sampling <importance>
     fisher: Fisher matrix calculation <fisher>


Ensemble samplers
===============================

.. toctree::
    :maxdepth: 1

     multinest: Nested sampling <multinest>
     kombine: Clustered KDE <kombine>
     pmc: Adaptive Importance Sampling <pmc>


Grid samplers
===============================

.. toctree::
    :maxdepth: 1

     grid: Simple grid sampler <grid>
     snake: Intelligent Grid exploration <snake>

Maximum likelihood samplers
===========================

.. toctree::
    :maxdepth: 1

     maxlike: Find the maximum likelihood using various methods in scipy <maxlike>
     minuit: Find the maximum posterior using the MINUIT library <minuit>
     gridmax: Naive grid maximum-posterior <gridmax>

Other samplers
===========================

.. toctree::
    :maxdepth: 1

     abcpmc: Approximate Bayesian Computing (ABC) Population Monte Carlo (PMC)  <abcpmc>
     apriori: Draw samples from the prior and evaluate the likelihood <apriori>

