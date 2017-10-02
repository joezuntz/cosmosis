# Demo 10:  Comparing Smith and Takahashi non-linear corrections, sampling over sigma_8

In this demo we will run cosmoSIS to compare two different non-linear corrections to the matter power spectrum, the Smith and Takahashi versions of the the `halofit` module (each taken from different versions of camb). The pipeline is based on [Demo 6](https://bitbucket.org/joezuntz/cosmosis/wiki/Demo6) using CFHTLenS likelihoods and the grid sampler. 

Instead of sampling over the primordial power spectrum amplitude, A_s, in this case we will sample over a measure of the late time amplitude measure sigma_8.  We'll also demonstrate how to use environment variables in ini files to reduce the number of files you need to copy around.

**This demo will take around 25 minutes when run on four cores. 
**
## Running ##
There are two commands to run, one for each version of Halofit:

```
#!bash

HALOFIT="halofit" cosmosis demos/demo10.ini
HALOFIT="halofit_takahashi" cosmosis demos/demo10.ini
```

We are setting the environment variable HALOFIT to either “halofit” or “halofit_takahashi” depending on which non-linear correction we are using in each run.

We run the post-process command on the two outputs at once to generate joint plots:
```
#!bash

postprocess demo10_output_halofit.txt demo10_output_halofit_takahashi.txt -o plots -p demo10 --tweaks demos/tweaks10.py
```

This will create two plots, including a one dimensional likelihood plot over sigma8 showing both corrections named plots/demo10_cosmological_parameters--sigma8_input.png.

![demo11_cosmological_parameters--sigma8_input.png](https://bitbucket.org/repo/KdA86K/images/3503097694-demo11_cosmological_parameters--sigma8_input.png)
 
## Understanding ##
The Smith correction is described in [Smith et al. (2003)](http://adsabs.harvard.edu/abs/2003MNRAS.341.1311S) and is implemented by the `halofit` module. The Takahashi correction is described in [Takahashi et al. (2012)](http://adsabs.harvard.edu/abs/2012ApJ...761..152T) and is implemented by the `halofit_takahashi` module. 

In the demo10.ini file the list of modules now reads:
```
#!ini

[pipeline]
modules = consistency camb sigma8_rescale ${HALOFIT} load_nz shear_shear 2pt cfhtlens 
```

`consistency`, `camb`, `load_nz`, `shear_shear`, `2pt`, and `cfhtlens` are all as described in [Demo 6](https://bitbucket.org/joezuntz/cosmosis/wiki/Demo6).

`sigma8_rescale` is a new module. It allows us to sample over sigma8 instead of A_s, and is run directly after camb to modify the latter's outputs.  We do this by setting in the values10.ini file a fiducial value for A_s for CAMB to use, and a new parameter sigma8_input for the rescaling:
```
#!ini
A_s = 2.1e-9
sigma8_input = 0.67 0.7 0.87

```


The `${HALOFIT}` module is replaced with the environment variable, which we set to either “halofit” or “halofit_takahashi” before running. Both modules are allocated separate sections in the demo10.ini file. 
The same syntax is used in the output section of demo10.ini, which now reads:
```
#!ini

[output]
format=text
filename=demo10_output_${HALOFIT}.txt
```

This results in two output text files being created, even though only one configuration file exists.
 
The post-processing script takes two arguments, one for each output file. The resulting plots each contain two curves corresponding to the two non-linear corrections. The sigma8 likelihood plot shows the two corrections overlapping over most of their sigma8 range but peaking at slightly different values.

A tweaks file named tweaks10.py is also called during postprocess to add a legend and y axis label to these plots - plot tweaks are described in detail in [Demo 8](https://bitbucket.org/joezuntz/cosmosis/wiki/Demo8).