Command line flags for cosmosis
-------------------------------

You can view command line flags for cosmosis using the command::

    cosmosis --help


Parallelization
===============

The two flags :code:`--mpi` and :code:`--smp` run the code in parallel.

The MPI flag runs the code under MPI, which means different processes are launched and communicate with each other.  This requires the MPI runtime to be installed, and you must run cosmosis with the :code:`mpirun` command first.  This will depend on the details of your system, but usually a command like::

    mpirun -n 4 cosmosis --mpi params.ini

will run CosmoSIS using 4 processes.  All the parallel cosmosis samplers can run under MPI, and it can be used on large machines like supercomputers across nodes.


The SMP flag runs with the python multiprocessing module, which starts one process and then forks new ones. You would use it like this::

    cosmosis --smp 4 params.ini

Most of the cosmosis samplers can use SMP, but not the Multinest or Polychord samplers. It can only run on single nodes of larger machines.

Debugging
=========

CosmoSIS has two commands that help with debugging, :code:`--pdb` and :code:`--experimental-fault-handling`.

If your code fails and crashes while running any python module (not C, C++, or Fortran) then the :code:`--pdb` flag will mean rather than just crashing the code will stop at the point of the error and you will enter the python debugger, PDB.  You can read about how to use this debugger here: https://docs.python.org/3/library/pdb.html

The :code:`--experimental-fault-handling` requires the python module :code:`faulthandler` to be installed on your system, for example with :code:`pip install faulthandler`.  This command means that if your code crashes during a C, C++, or Fortran module you will get a traceback listing which functions were being run at the time.


Overriding Parameter Files
===========================

It can be useful to override parameters specified in the configuration files on the command line - this can let you launch a variety of different runs with the same file.  The :code:`-p` and :code:`-v` flags let you override parameters in the main (params.ini) and values files respectively.

You can override any number of parameters in the main parameter file like this::

    cosmosis params.ini -p section1.name1=value  section2.name2=value ...

For example, this command would change the sampler being used to emcee instead of its current value::

    cosmosis params.ini -p runtime.sampler=test

The :code:`-v` command is used exactly the same way but for the values file, for example, this would change one of the parameter ranges in demo 5::

    cosmosis demos/demo5.ini  -v cosmological_parameters.omega_m="0.2 0.3 0.5"

Note the quotations marks above, which are needed when there are spaces in the parameter value. 
