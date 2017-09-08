# Demo 4:  Finding the best-fit parameters for Planck 2013 #

**Note that both the data and sampler used in this example are a little out of date.  If you are new to CosmoSIS you should still read it to get an idea of how things work, but you can see how to fit Planck 2015 data with the more robust Minuit sampler in [Demo 16](Demo16)**

## Running ##

In this example we will find the maximum likelihood cosmological parameters for Planck 2013.  For simplicity we keep the Planck nuisance parameters and the optical depth tau fixed, so we have five varying parameters.  Run like this:

```
#!bash

cosmosis demos/demo4.ini
```
This will take a little while, since we have to do several hundred different cosmologies.  Once done it will create two files, planck_best_fit.txt and planck_best_fit.ini.

 The first txt file planck_best_fit.txt is in the same form as the grid output from before - as column text.  But this time there is only one entry, the best-fit sample:

```
#!text

#cosmological_parameters--omega_m	cosmological_parameters--h0	cosmological_parameters--omega_b	cosmological_parameters--a_s	cosmological_parameters--n_s	LIKE
...
0.323118340227	0.663067835995	0.0481099681605	2.15244682388e-09	0.958526048347	-4033.15502856

```

(your numbers may be different from this - for speed we set the accuracy settings in the demo quite weakly - if you want a more accurate answer then see the note below).

The second output file planck_best_fit.ini is more interesting:
```
#!ini
[cosmological_parameters]
omega_m  =  0.2    0.32311834022735636    0.4
h0  =  0.6    0.66306783599489649    0.8
omega_b  =  0.02    0.048109968160540695    0.06
a_s  =  2e-09    2.152446823883425e-09    2.3e-09
n_s  =  0.92    0.95852604834714505    1.0
tau  =  0.08
omega_k  =  0.0
 
 ...

```
This is a new version of our ini file with the starting positions of all the varied parameters set to their best-fit values.  This can be used as a starting point for MCMC samplers to avoid the burn-in period at the start of their run.


## Understanding ##

This example introduces a third "sampler", which uses the Nelder-Mead simplex algorithm to find the maximum likelihood. (Notice that this algorithm does not draw samples from a probability distribution - but we refer to it as a sampler for code naming consistency.) There are a few parameters we set for this sampler:

```
#!ini

[runtime]
sampler = maxlike

[maxlike]
maxiter = 100
tolerance = 0.1
output_ini = planck_best_fit.ini

[output]
filename = planck_best_fit.txt
format = text
verbosity= debug

```

The tolerance and maxiter parameters are paraemters of the algorithm; reducing tolerance will improve the fit at the cost of time.  0.1 is really too low for this kind of parameter space - 0.01 will give you much better answers at the cost of more time (though still not too long).


We got the second ini output file because we set output_ini.  Setting the verbosity to debug meant we got lots of output during the optimization.

The parameters that we just gave a single value for in the values4.ini file were left fixed during the optimization, and the ones for which we gave a range were varied:

```
#!ini

[cosmological_parameters]

omega_m = 0.2  0.3  0.4
h0 = 0.6  0.7  0.8 
omega_b = 0.02  0.04  0.06
A_s = 2.0e-9   2.1e-9  2.3e-9
n_s = 0.92  0.96  1.0

; We will leave these parameters constant
; for speed
tau = 0.08
omega_k = 0.0
w = -1.0

 ...

```