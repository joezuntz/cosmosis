# Demo 6:  Getting a CFHTLens likelihood

## Running ##

In this example we will generate a likelihood of a cosmology using the CFHTLens tomographic data from Heymans et al.

We will use the test sampler again, so just a single cosmology.  I only just added the demo6.ini file, so if you can't see demos/demo6.ini then do "__git pull__" in your main directory to get these new files.

```
#!bash

cosmosis demos/demo6.ini
```

You should see some output like this:

    -- Setting up module camb --
    -- Setting up module halofit --
    -- Setting up module load_nz --
    Found 6 samples and 72 bins in redshift in file cosmosis-standard-    library/likelihood/cfhtlens/combined_nz.txt
    -- Setting up module shear_shear --
    -- Setting up module 2pt --
    -- Setting up module cfhtlens --
    Need to check the Anderson Hartlap when cutting matrix - cut first?
    xi_plus only? False
    Cut low thetas? True
    Setup all pipeline modules
    Pipeline ran okay.
    Likelihood -1.694817e+02
    Prior      =  0
    Likelihood =  -169.481692685
    Posterior  =  -169.481692685

The last number is the CFHTLens log-posterior of the parameters in demos/values6.ini.

Now let's make some plots.  For some variety, let's make them PDF plots instead of PNG:

```
#!bash

postprocess  demos/demo6.ini -o plots -p demo6 -f pdf

```

You will get some nice new plots in plots/demo6*.pdf.  Since we are now computing shear-shear functions we will have two new plots compared to demo 1: demo6_shear_power.pdf and demo6_shear_correlation.pdf.  Here's the shear power plot:

![shear_power.png](https://bitbucket.org/repo/KdA86K/images/2218571580-shear_power.png)

## Understanding ##

The weak lensing case is an example where there is a longer sequence of different modules.  The demos/demo6.ini file contains this line:

    modules = consistency camb halofit  load_nz  shear_shear  2pt cfhtlens

We have already looked at the consistency, camb and halofit modules in [demo one](Demo1).
The _load_nz_ module just loads a simple text file containing n(z) in its setup, and then each time it is executed provides the same n(z).

The _shear_shear_ module computes the Limber integral to go from P(k,z) and n(z) to C_ell for the different bins and the correlations between them.

The _2pt_ module integrates the C_ell with bessel functions J0 and J4 to get correlation function xi+ and x-.

The _CFHTLens_ module interpolates into the xi+ and xi- values to get their values at the CFHTLens observed values, and then gets a likelihood.

