
Samplers
--------

Samplers are the different methods that CosmoSIS uses to choose points in parameter spaces to evaluate.

Some are designed to actually explore likelihood spaces; others are useful for testing and understanding likelihoods.

.. toctree::
    :maxdepth: 1

     abcpmc: Approximate Bayesian Computing (ABC) Population Monte Carlo (PMC)  <abcpmc>
     apriori: Draw samples from the prior and evaluate the likelihood <apriori>
     emcee: Ensemble walker sampling <emcee>
     fisher: Fisher matrix calculation <fisher>
     grid: Simple grid sampler <grid>
     gridmax: Naive grid maximum-posterior <gridmax>
     importance: Importance sampling <importance>
     kombine: Clustered KDE <kombine>
     list: Re-run existing chain samples <list>
     maxlike: Find the maximum likelihood using various methods in scipy <maxlike>
     metropolis: Classic Metropolis-Hastings sampling <metropolis>
     minuit: Find the maximum posterior using the MINUIT library <minuit>
     multinest: Nested sampling <multinest>
     pmc: Adaptive Importance Sampling <pmc>
     snake: Intelligent Grid exploration <snake>
     star: Simple star sampler <star>
     test: Evaluate a single parameter set <test>
