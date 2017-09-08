# Demo 13: Fast grid analysis of the JLA Supernovae with the Snake sampler #

# This demo is only present in the development version of cosmosis#

We saw in [demo 4](Demo4) and [demo7](Demo7) how a grid sampler can produce an extremely clean and easy to analyze representation of the posterior.  But the grid sampler scales extremely badly as we increase the number of dimensions, as (number samples) ~ (grid points)^(number parameters).

The Snake sampler, introduced in [Mikkelsen et al](http://arxiv.org/abs/1211.3126), generates samples from a grid like the usual grid sampler, but rather than taking all samples regularly it moves gradually out from the peak of the likelihood, taking adjacent points to existing ones until it reaches a certain likelihood difference from the peak to the edge.

## Running

Run this demo with the command:

    cosmosis demos/demo13.ini

This should take a few minutes; if we had used the grid sampler it would have taken a few weeks.

Generate plots using:

    postprocess demos/demo13.ini -o plots -p demo13

This will make plots similar to those from [demo 5](Demo5), constraining the cosmological and supernova parameters, like this joint posterior on h and Omega_m:

![demo13_2D_cosmological_parameters--omega_m_cosmological_parameters--h0.png](https://bitbucket.org/repo/KdA86K/images/4189001617-demo13_2D_cosmological_parameters--omega_m_cosmological_parameters--h0.png)


You can also switch off the smoothing effect and get a better idea of what is actually calculated using:

    postprocess demos/demo13.ini -o plots -p demo13 --no-smooth

Which will yield a blockier version of the same plot:

![demo13_2D_cosmological_parameters--omega_m_cosmological_parameters--h0.png](https://bitbucket.org/repo/KdA86K/images/322211700-demo13_2D_cosmological_parameters--omega_m_cosmological_parameters--h0.png)


## Understanding


We use the same pipeline as in [demo 5](Demo5) - supernovae likelihoods with an H0 prior.
