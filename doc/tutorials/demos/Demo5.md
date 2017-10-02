# Demo 5:  Running an MCMC analysis of supernova data #

## Running ##

In this example we will run a full MCMC analysis on data from the SDSS-II/SNLS3 Joint Light-curve Analysis (JLA).  See http://arxiv.org/abs/1401.4064 for more details about this data.

Run the sampler EITHER like this:


```
#!bash

cosmosis demos/demo5.ini
```

OR if you are feeling adventurous and have MPI set up:
```
#!bash

mpirun -n 4 cosmosis --mpi demos/demo5.ini
```



This will take a few minutes to run, depending on your machine. On a Mac you may be asked if python 2.7 should accept incoming network connections - click Allow. You should produce a file demo5.txt with 25600 samples:

```
#!bash

#cosmological_parameters--omega_m       cosmological_parameters--h0     supernova_params--deltam        supernova_params--alpha supernova_params--beta  LIKE
# ... [SNIP - all the parameters you ran with are saved here, along with other info like who to cite]
0.375349838504  0.543069181465  0.328431291151  0.124554370328  3.28955217497   -2227.10121001
0.277715836563  0.81440479776   6.95767929396   0.152362132558  8.60894281101   -27083.8222028
0.396752581802  0.664902793534  1.88367270414   0.127254401707  4.48364725723   -8182.54328716
0.215070227128  0.762109999036  5.29185049673   0.139453531438  6.46507792112   -26799.1925617
0.214404110344  0.781725633573  2.1142025756    0.141179467329  8.25565740506   -3161.17547089
# ...

```

(N.B. The actual numbers you get will be different to the above because it is a random process.) Now let's generate some plots to visualize the chain:

```
#!bash
postprocess  --burn 5000  -o plots -p demo5 demos/demo5.ini

```
or
```
#!bash
./cosmosis/plotting/marginal_densities.r -b 10000 -o plots/ demo5.txt -p demo5 --verbose --fill
```

Note that the _R_ script produces plots similar to those of the _Python_ script, but does not yet produce textual statistical summaries.

You will get output which will include summary information like this:

```
#!text
    Samples after cutting: 20600

    Marginalized mean, std-dev:
        cosmological_parameters--omega_m = 0.29728 ± 0.0328477
        cosmological_parameters--h0 = 0.738122 ± 0.0249374
        supernova_params--deltam = 0.040323 ± 0.0768359
        supernova_params--alpha = 0.137666 ± 0.00633048
        supernova_params--beta = 3.10945 ± 0.0814108
        like = -345.562 ± 1.55596

    Marginalized median, std-dev:
        cosmological_parameters--omega_m = 0.296391 ± 0.0328477
        cosmological_parameters--h0 = 0.73781 ± 0.0249374
        supernova_params--deltam = 0.0412422 ± 0.0768359
        supernova_params--alpha = 0.137889 ± 0.00633048
        supernova_params--beta = 3.10694 ± 0.0814108
        like = -345.255 ± 1.55596

    Best likelihood:
        cosmological_parameters--omega_m = 0.292377
        cosmological_parameters--h0 = 0.734679
        supernova_params--deltam = 0.0270577
        supernova_params--alpha = 0.137495
        supernova_params--beta = 3.09374
        like = -343.068
```

Also the plots directory should also now contain a collection of 1D and 2D histogram plots in pleasant shades of blue, like this one:

![2D_supernova_params--deltam_cosmological_parameters--h0.png](https://bitbucket.org/repo/KdA86K/images/2269836943-2D_supernova_params--deltam_cosmological_parameters--h0.png)

You should also find some files with the statistics above saved in a more structured form.


## Understanding ##

### The sampling ###


We are now using a proper MCMC sampler, called emcee.  This sampler is described in http://arxiv.org/abs/1202.3665 .    

Here's how we tell cosmosis to use it in demos/demo5.ini:

```
#!ini
[emcee]
walkers = 64
samples = 1000
nsteps = 100

```
You can find out more about the meaning of these parameters by looking in the ini file itself.


Setting up a cosmosis sampling run involves choosing (or writing) the modules that you want to run, making sure that each produces the quantities that later ones need, and then choosing what likelihoods to take out at the end.

We specify our pipeline like this in the same ini file:

```
#!ini

[pipeline]
; We use two likelihoods, the JLA (for high redshift) and
; Riess 2011 to anchor H0, which is otherwise degenerate
; with the nuisance parameter M
modules = consistency camb jla riess11
values = demos/values5.ini
likelihoods = jla riess

```
We take the sum of two likelihoods, *jla* and *riess* to get our constraints.

Once again each module in the list above is specified by an ini file section, with options in it.  The JLA module, for example, takes quite a lot of parameters pointing to the data files it needs:

```
#!ini

[jla]
file = cosmosis-standard-library/supernovae/jla_v3/jla.so
data_dir = cosmosis-standard-library/supernovae/jla_v3/data
data_file = jla_lcparams.txt
scriptmcut = 10.0
mag_covmat_file = jla_v0_covmatrix.dat
; ...
```



### The plotting ###

Our chain analysis program *postprocess* uses a technique called kernel density estimation (KDE) to smooth the 1D and 2D histograms and form smooth edges.

We use the --burn flag to burn the first 5000 samples.  We also specified the output directory and prefix.

You can get more information on postprocess by running it with the -h flag.
