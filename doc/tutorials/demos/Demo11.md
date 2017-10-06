# Demo 11: Modified gravity weak lensing with ISiTGR and CFHTLens and saving cosmological data during sampling

Various ways to modify gravity have been implemented in versions of the Boltzmann code CAMB.  CosmoSIS currently includes two of them: [ISiTGR: Integrated Software in Testing General Relativity](http://www.utdallas.edu/~jnd041000/isitgr/), and [MGCAMB version 2](http://www.sfu.ca/~aha25/MGCAMB.html).  These two codes use a phenomenological parameterization of modifications to the perturbed Einstein equations on large scales.

In this demo we generate a 1D slice in the CFHTLenS shear-shear likelihood to constrain one of the ISiTGR parameters, which controls modifications to the relation between the two metric potentials phi and psi.

##Running

If you have a machine with four or more cores, try running like this:

```
#!bash
mpirun -n 4 cosmosis --mpi demos/demo11.ini
```

if that doesn't work then you can fall back to the slower non-MPI mode:
```
#!bash
cosmosis demos/demo11.ini
```

And then either way generate plots and statistics with:

```
#!bash
postprocess demos/demo11.ini -o plots -p demo11
```

Which will tell you, among other things, that the PPF parameter D_0 is constrained for these cosmological parameters to D_0=0.982021±0.0364016.  

This will be illustrated in the plot that will be generated, plots/demo11_post_friedmann_parameters--d_0.png.  Boringly, the GR case D_0=1 is well within the 68% contours:

![demo12_post_friedmann_parameters--d_0.png](https://bitbucket.org/repo/KdA86K/images/754248129-demo12_post_friedmann_parameters--d_0.png)

A collection of tgz files will also appear in demo11/data_0.tgz, demo11/data_1.tgz, ...
These files are compressed versions of all the cosmological data computed for each model in the sample, and contain all the outputs that appeared in each directory for [Demo 6](Demo6), but one for each set of cosmological parameters.  If you want you can extract these using tar -zxvf demo11/data_0.tgz and explore the theory results in more detail.

------------------------------
##Understanding

Like (Demo 6)[Demo6] our pipeline computes cosmic shear correlation functions and compares them to CFHTLenS data.  See that demo for info on the basic pipeline; here we will go through the changes that add modified gravity effects, and show how to save all the data generated for all the samples.

------------------------------
### Science

Parametrized Post-Friedmann and related models use a phenomenological set of changes to gravity that tweak GR, rather than implementing a specific theory of modified gravity.  There are various different ways of making such changes, but all revolve around modifying the perturbed Einstein equations which describe the evolution of gravitational fluctuations.

For example in isItGR the (flat space) equations:

![Screen Shot 2014-11-20 at 14.58.10.png](https://bitbucket.org/repo/KdA86K/images/3848156670-Screen%20Shot%202014-11-20%20at%2014.58.10.png)

become:

![Screen Shot 2014-11-20 at 14.58.19.png](https://bitbucket.org/repo/KdA86K/images/2968548161-Screen%20Shot%202014-11-20%20at%2014.58.19.png)

The *Q* and *R* values, which generically are functions of time and scale, modify the equations. If they are unity then GR is recovered.  *Q* acts like a modified Newton's constant, and R creates a slip between the spatial and temporal perturbations *phi* and *psi*.  Isitgr also includes an alternative parametrization, where instead of R we use:

![Screen Shot 2014-11-20 at 15.09.50.png](https://bitbucket.org/repo/KdA86K/images/2322847912-Screen%20Shot%202014-11-20%20at%2015.09.50.png)

Choosing a modified gravity also involves selecting how *Q* and *R* (or *D*) vary with *k* and *z*.  In isitgr that choice is:

![Screen Shot 2014-11-20 at 15.11.13.png](https://bitbucket.org/repo/KdA86K/images/422418772-Screen%20Shot%202014-11-20%20at%2015.11.13.png)

The value D also modifies the lensing kernel in cosmic shear, scaling it by D^2.  A module with this modification is included in cosmosis too.

------------------------------

### Pipeline

Our pipeline is the a small tweak on the standard LCDM pipeline from Demo 6:

    modules = consistency isitgr sigma8_rescale halofit load_nz shear_shear_mg 2pt cfhtlens

We have changed ```camb``` to ```isitgr``` and ```shear_shear``` to ```shear_shear_mg```, and added the ```sigma8_rescale``` module from [Demo 10](Demo10).

The ```isitgr``` module takes these new parameters in demos/values11.ini:

```
#!ini
[post_friedmann_parameters]
d_0 = 0.85 1.0 1.1
d_inf = 1.0
q_0 = 1.0
q_inf = 1.0
s = 1.0
k_c = 0.01
```
These are the parameters in the equations shown above.  It also takes these new control parameters in the demos/demo11.ini file:

```
#!ini
scale_dependent = F
use_r_function = F
```

which determine whether to use the k-dependence in the equations above or switch it off, and whether it is R or D which has the ansatz shown above.  In either case it R_0 and R_inf are derived from the D and Q parameters.

As well as modifying the CMB and matter power spectra, the isitgr module saves grids of D, dD/dtau, Q, and dQ/dtau as functions of (k,z).  It also saves the derived parameters v_0, v_inf, r_0, and r_inf.  These are all saved in the modified_gravity section.

The shear_shear_mg module loads D(k,z) and uses it to scale the matter power spectrum used to calculate the lensing.

------------------------------

### Data saving

The demos/demo11.ini file contains these lines in the section [grid]:

```
#!ini
[grid]
save=demo11/data
```

The "save" option tells the grid sampler to save the complete datablock - all the saved cosmological information - to .tgz files saved with the given name - ```demo11/data_1.tgz```, ```demo11/data_2.tgz```, etc.

You can extract these files using the ```tar -zxvf``` command, or load them dynamically in python using the in-built ```tarfile``` module.  In future we will include a simpler cosmosis interface to this.