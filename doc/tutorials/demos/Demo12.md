# Demo 12:  Getting the PDF for the maximum cluster mass using Extreme Value Statistics at a single redshift for a given cosmology#

This demo will calculate the probability distribution function for the maximum cluster mass at a single redshift of z=1.6 for a given cosmology using the Tinker et al 2008 mass function module. The likelihood given the XXMU J0044 cluster mass measurement at z=1.579 by Santos et al 2011 (as given in Table 1 of Harrison & Coles 2012) is also computed.

## Running ##

Run the demo with

```
#!bash

cosmosis demos/demo12.ini
```

The output in demo12_output/evs contains the PDF for the maximum cluster mass at this redshift.
You can plot this output together with the measurement of the XXMU J0044 cluster using:
```
#!bash

python demos/plot_demo12.py

```

![maxmass.png](https://bitbucket.org/repo/KdA86K/images/2348810901-maxmass.png)


## Understanding ##

There are five modules in the pipeline for this demo. Two are "physics" modules which, given a set of cosmological parameters, output useful things like the power spectrum and mass functions. The final two stages in the pipeline are "likelihood" modules which output likelihoods which can be used in sampling.

```
#!bash
[pipeline]
modules = consistency camb mf_tinker evs cluster_mass
likelihoods = evs maxmass
```
The 'camb` module performs the Boltzmann integration to calculate the complete linear  matter power spectrum in a narrow redshift range around the redshift of the cluster we are considering.

```
#!bash
[camb]
file = cosmosis-standard-library/boltzmann/camb/camb.so
zmin = 1.3
zmax = 1.9
nz = 40

```

The "mf_tinker" module calculates the mass function, using the Tinker et al 2008 formula, at each redshift that camb outputs  the power spectrum. This is done using the "redshift_zero = 0" setting. If only the z=0 mass function is required we could use 
"redshift_zero = 1". The mf_tinker module is based on Eiichiro Komatsu's Cosmology Routine Library.

The "evs" module computes the Extreme Value Statistics for the maximum cluster mass at a given redshift using the formula in Harrison & Coles 2012. In this test example we output the full pdf for a range of masses at a redshift of z=1.6, this is done using the "output_pdf = T" setting which we only recommend for the test sampler as it takes a long time to run.
The likelihood of a single cluster mass ("M_max" in the values.ini file) is always output (see the "Longer Demos" section of the wiki for an example of this being using in a sampling run)

The "cluster_mass" module outputs the Gaussian likelihood for M_max given the XXMU J0044 cluster mass measurement at z=1.579 by Santos et al 2011.






