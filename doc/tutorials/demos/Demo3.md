# Demo 3:  Getting a slice in the BICEP likelihood in the r_T parameter #

In this example we will compute likelihoods along a line through our parameter space.
As an example we use the BICEP2 BB data and fit the tensor amplitude r.

## Running ##

After building CosmoSIS, run from the source directory:

```
#!bash

cosmosis demos/demo3.ini
```

This will take a little longer to run than the first two examples, as it is running 25 different cosmologies.  It will create the file demo3_output.txt, which you can plot either with your favorite plotting program or by running:

```
#!bash

postprocess demos/demo3.ini -o plots -p demo3
```

This will create plots/demo3_cosmological_parameters--r_t.png, which will look a little like this:
![demo3_cosmological_parameters--r_t.png](https://bitbucket.org/repo/KdA86K/images/958349557-demo3_cosmological_parameters--r_t.png)

The dashed vertical lines are 68% and 95% contours.


It will also print out some summary statistics on the constraints the BICEP likelihood gives you on the tensor amplitude:

```
#!bash
Marginalized mean, std-dev:
    cosmological_parameters--r_t = 0.228335 ± 0.0536721

Marginalized median, std-dev:
    cosmological_parameters--r_t = 0.213946 ± 0.0536721

Best likelihood:
    cosmological_parameters--r_t = 0.208333
    like = -3.24528
```

## Understanding ##
Let us look at the demos/demo3.ini file.

We are now using a grid sampler,
```
[runtime]
sampler = grid
```

rather than the simple test sampler we used before.  The grid sampler is gridding here over just a single dimension, the parameter r_T.  The file demos/demo3.ini shows how we are telling CosmoSIS to use the grid sampler instead:

```
#!ini

[runtime]
sampler = grid

[grid]
; The number of samples to take in each
; dimension in which the parameters vary
nsample_dimension = 25
```

In the values file demos/values3.ini

```
#!ini

[cosmological_parameters]
; Listing all these three numbers is how to specify
; that a parameter should vary.  The numbers are:
; lower limit,  starting value,  upper limit
r_t = 0.0  0.2  0.7
```

For the grid sampler, only the lower limit and upper limit are used (the starting value is ignored). 
Any parameter which has all these three values listed instead of just one will be included in the gridding, or in general, in whatever sampling process you are doing.  Otherwise it will be kept constant.

The postprocess script understands how to plot results and compute statistics for all the different samplers in cosmosis.