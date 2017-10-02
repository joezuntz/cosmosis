# Demo 7:  A 2D likelihood grid from BOSS DR9 CMASS Redshift-Space Distortion measurements of f * sigma_8

## Running ##

In this example we will generate a grid of likelihoods - we use the same grid sampler as in [Demo 3](Demo3), but this time with a 2D grid instead of 1D.


Run the demo:
```
#!bash

cosmosis demos/demo7.ini
```

This demo should take 1-2 minutes to run.

You should see some output like this:

    Will calculate f(z) and d(z) in 61 bins (0.000000:0.600000:0.010000)
    - CosmoSIS verbosity set to 40
    Growth parameters: z =  0.57 fsigma_8  =  0.215676036202  z0 =  0
    Growth parameters: z =  0.57 fsigma_8  =  0.231567954659  z0 =  0
    Growth parameters: z =  0.57 fsigma_8  =  0.247459873116  z0 =  0
    ....


The output file demo7.txt should contain 400 samples, representing the likelihoods on a 20 x 20 grid.

As always, we can post-process this file using the postprocess command:

```
#!bash

postprocess demos/demo7.ini -o plots/ -p demo7
```

or make plots in R with:
```
#!bash
 ./cosmosis/plotting/grid_plots.r -o plots -p demo7 -f demo7.txt
```

You will get a quick warning about Omega_m having large likelihoods at the edge of the field, and some summary statistics and three plots will be generated:

```
#!text
Marginalized mean, std-dev:
    cosmological_parameters--omega_m = 0.241955 ± 0.0915885
    cosmological_parameters--sigma_8 = 0.802542 ± 0.141301

Marginalized median, std-dev:
    cosmological_parameters--omega_m = 0.229923 ± 0.0915885
    cosmological_parameters--sigma_8 = 0.774402 ± 0.141301

Best likelihood:
    cosmological_parameters--omega_m = 0.178947
    cosmological_parameters--sigma_8 = 0.831579
    like = 1.79913

```

The 2D likelihood plot, plots/demo7_cosmological_parameters--omega_m_cosmological_parameters--sigma_8.png.  The color scale is for the likelihood, and the grey and black regions show 68% and 95% contours.

![demo7_2D_cosmological_parameters--sigma_8_cosmological_parameters--omega_m.png](https://bitbucket.org/repo/KdA86K/images/2931201929-demo7_2D_cosmological_parameters--sigma_8_cosmological_parameters--omega_m.png)

And the two 1D likelihoods, plots/demo7_cosmological_parameters--sigma_8.png and plots/demo7_cosmological_parameters--omega_m.png.  You can see that sigma_8 is well-constrained but the omega_m is much weaker.  The dotted lines bound 68% and 95% contours.

![demo7_cosmological_parameters--sigma_8.png](https://bitbucket.org/repo/KdA86K/images/3142514419-demo7_cosmological_parameters--sigma_8.png)

![demo7_cosmological_parameters--omega_m.png](https://bitbucket.org/repo/KdA86K/images/3738009061-demo7_cosmological_parameters--omega_m.png)

## Understanding ##

This example shows how the grid sampler works in more than one dimension.  In the ini file we set 20 samples per dimension:

   [grid]
   # We do a 20 x 20 grid for a total of 400 points
    nsample_dimension=20

as in demo 3, but in the values file we set two varying parameters:

    [cosmological_parameters]
    omega_m =  0.1    0.27  0.4
    sigma_8 = 0.5 0.8 1.2

In this case we sample over sigma_8 directly rather than deriving it from CAMB as in earlier examples.
If we were combining with CMB, for exampling, we could get sigma_8 from there instead.
Cosmosis doesn't care where we got sigma_8 from (sampled over or derived) as long as we have it.

We use some new modules in this example:

    modules = consistency growthfunction boss
    likelihoods = boss

The growth function module does the calculation of f(z) and d(z) directly, rather than using P(k) outputs.  The likelihood uses a measurement of f(z) * sigma_8(z) at z=0.57 from BOSS DR9 presented in Chuang et al 2013.

The postprocessor knows how to marginalize to make the 1D plots and interpolate smoothly to make the 2D plot.   You can also use the flag --no-smooth if you want a more grid-like plot.