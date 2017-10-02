Tutorial 2: Running an MCMC sampler
-----------------------------------

In the first tutorial we generated a single likelihood.  In this tutorial we we run an MCMC analysis to explore a parameter space and put constraints on some parameters.  There are lots of different MCMC algorithms available through CosmoSIS; in this example we will use emcee.

Running an MCMC
================

Have a look at :code:`demos/demo5.ini` and its values file :code:`demos/demo5.ini`.

Let's try using `MPI parallelism <https://en.wikipedia.org/wiki/Message_Passing_Interface> `_ to speed up this analysis.  Run this command::

    mpirun -n 4 cosmosis --mpi demos/demo5.ini

If that fails straight away then you may not have MPI installed (MPI should work automatically with the bootstrap and docker installation methods - let us know if not). If it fails then you can fall back to serial mode::

    cosmosis demos/demo5.ini


Demo 5 describes a supernova likelihood, using the JLA supernova sample, which measures the redshift-distance relation.