# Demo 16:  Robust Planck 2015 best-fit using the Minuit Optimizer #

## Running ##

The Minuit2 library contains a pretty robust optimization algorithm that we will use here to get find the best-fit to one of the Planck 2015 data sets.

We will just leave four parameters free: Omega_m, H0, n_s, and A_s. Run the code like this:

```
#!bash

cosmosis demos/demo16.ini
```
This may take up to 600 or so calls to the likelihood, which will take up to ten minutes, so it's a good chance to get caught up on the arxiv.  Once it's finished you should see `demo16_output.ini` and `demo_16_covmat.txt`, as well as a directory called `demo16`.

The minuit sampler saves the cosmology calculations in the output directory, and parameter files and a covariance (which may be useful for sampling later) in the other files.


## Understanding ##

The Minuit optimizer, called Migrad, calculates numerical derivatives of the posterior and uses the BFGS rule to update its estimate of the Hessian matrix, and uses that to move towards the peak of the distribution. 

Minuit appears to be more robust than many other samplers, but you should still treat it with care and perhaps start a few chains from different places in parameter space.  You can use the output ini file as the values file for a new sampler, and if you are using metropolis or a similar sampler that can use a covariance matrix then you can use that output too.

The Planck 2015 data used here is the simplified version of the data set which includes only a single nuisance parameter, representing the overall calibration of the survey. This makes sampling the full space much easier.

