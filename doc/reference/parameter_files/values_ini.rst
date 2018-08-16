Values Files
============

The values file defines the parameters that the sampler varies over, and other fixed parameters that are put into the likelihood pipeline at the start.  All the values in the values file will be added to the data block before any modules are run.

The values file is required to run cosmosis. The path to it must be specified in the :code:`[pipeline]` section of the main parameter file, using the syntax :code:`values = path/to/values.ini`.


Like the other parameter files it is in the :code:`ini` format.  All parameters should be in named :code:`[section]s`.

Fixed parameters
-----------------

Fixed parameters are just given a single value, for example::

    [cosmological_parameters]
    w = -1.0

Fixed parameters can be integer or double types - if your code is expecting a double then make sure you don't write it as an integer - for example, writing the above example as :code:`w=-1` would cause an error.


Varied parameters
------------------

Parameters that should be varied in the run are given three values, representing, respectively, the parameter lower limit, a typical starting value, and an upper limit, for example::

    [cosmological_parameters]
    omega_m  =  0.15    0.3    0.4

Different samplers use these ranges in different ways - for example, the test sampler ignores everything except the starting point, the metropolis sampler starts a chain at the start point and won't let it stray outside the limits, and the grid sampler ignores the starting value and generates a linear grid between the lower and upper limits.  See the documentation on the samplers for more information.

Creating a parameter like this gives it an implicit uniform prior between the lower and upper limits (0.15 to 0.4).  You can add additiona priors in the priors file.