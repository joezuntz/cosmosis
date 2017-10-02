# Demo 2:  Getting a (Planck) likelihood for a single cosmology

When you want to do a Bayesian analysis of a model and some data, you need a likelihood - the probability of a data set (in this case the Planck CMB measurements) given a particular model (in this case the LCDM cosmological model).

The `test` sampler generates a single likelihood for one set of parameters.


## Running ##

```
#!bash

cosmosis demos/demo2.ini
```

You should seem some output from CAMB, just like in demo1.  At the end you will also see the total likelihood value of the cosmology.

This time the output directory is called demo_output_2.  You can look at the individual likelihood values by looking at the file:

```
#!bash

 cat  demo_output_2/likelihoods/values.txt

```

As before, you can also plot it using:

```
#!bash

 postprocess demos/demo2.ini -o plots -p demo2

```


You will now have a collection of png plots in the plots directory.  This time they will include a BB plot, which we need for BICEP:

![demo2_bb.png](https://bitbucket.org/repo/KdA86K/images/1594462742-demo2_bb.png)
## Understanding ##

Once again, the pipeline is defined in the section of that name

```
#!ini

[pipeline]
modules = consistency camb planck bicep2
values = demos/values2.ini
; ...
likelihoods = bicep planck

```

This time we are running four modules - consistency, camb, Planck, and Bicep2.  You can see the paths to them in the sections below.  The latter two of these are interesting in that they generate likelihood values, which can be used by MCMC samplers to explore the parameter space.  We have also told cosmosis what likelihoods it should look for by setting:
likelihoods = bicep planck

This means that CosmoSIS will look for two values, bicep_like and planck_like in the likelihood section when it comes to work out the total likelihood.  There may be other likelihoods calculated (for example, the file demo_output_2/likelihoods/values.txt contains all the separate Planck contributions as well as the total), but if they are not in this list they will *not* be included.


Now have a look at the **demos/values2.ini** file.  You will see that we now have to different sections to split up the parameters we use by type.  We have the regular cosmological parameters (now including r_T=0.2 as we are in the exciting post-BICEP world), but we also have the many nuisance parameters that Planck needs to know.