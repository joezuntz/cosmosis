The Multinest sampler
--------------------------------------------------------------------

Nested sampling

+--------------+------------------------------------------------------+
| | Name       | | multinest                                          |
+--------------+------------------------------------------------------+
| | Version    | | 3.7                                                |
+--------------+------------------------------------------------------+
| | Author(s)  | | Farhan Feroz,Mike Hobson                           |
+--------------+------------------------------------------------------+
| | URL        | | https://ccpforge.cse.rl.ac.uk/gf/project/multinest/|
+--------------+------------------------------------------------------+
| | Citation(s)| | arXiv:0809.3437, arXiv:0704.3704, arXiv:1306.2144  |
+--------------+------------------------------------------------------+
| | Parallelism| | parallel                                           |
+--------------+------------------------------------------------------+

Nested sampling is a method designed to calculate the Bayesian Evidence of a distribution, for use in comparing multiple models to see which fit the data better.

The evidence is the integral of the likelihood over the prior; it is equivalent to the probability of the model given the data (marginalizing over the specific parameter values): B = P(D|M) = \int P(D|Mp) P(p|M) dp

Nested sampling is an efficient method for evaluating this integral using members of an ensemble of live points and steadily replacing the lowest likelihood point with a new one  from a gradually shrinking proposal so and evaluating the integral in horizontal slices.

Multinest is a particularly sophisticated implementation of this which can cope  with multi-modal distributions using a k-means clustering algorithm and a proposal made from a collection of ellipsoids.

The output from multinest is not a set of posterior samples, but rather a set of weighted samples - when making histograms or parameter estimates these must be included.

The primary mulitnest parameter is the number of live points in the ensemble. If this number is too small you will get too few posterior samples in the result, and if it is too large the sampling will take a long time.  A few hundred seems to be reasonable for typical cosmology problems.

One odd feature of the multinest output is that it doesn't save any results until it has done a complete run through the parameter space.  It then starts again on a second run,  and sometimes a third depending on the parameters.  So don't worry if you don't see any lines in the output file for a while.



Installation
============

No special installation required; everything is packaged with CosmoSIS




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+-------------------------+----------+---------------------------------------------------------------+----------+
| | Parameter             | | Type   | | Meaning                                                     | | Default|
+-------------------------+----------+---------------------------------------------------------------+----------+
| | live_points           | | integer| | Number of live points in the ensemble                       |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | random_seed           | | integer| | Seed to use for random proposal; -1 to generate from current| | -1     |
|                         |          | | time.  Allows re-running chains exactly                     |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | feedback              | | bool   | | Print out progression information from multinest            | | T      |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | resume                | | bool   | | If you previously set multinest_outfile_root you can restart| | F      |
|                         |          | | an interrupted chain with this setting                      |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | wrapped_params        | | str    | | Space separated list of parameters (section--name) that     | | (empty)|
|                         |          | | should be given periodic boundary conditions. Can help      |          |
|                         |          | | sample params that hit edge of prior.                       |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | multinest_outfile_root| | str    | | In addition to CosmoSIS output, save a collection of        | | (empty)|
|                         |          | | multinest output files                                      |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | ins                   | | boolean| | Use Importance Nested Sampling (INS) mode - see papers for  | | True   |
|                         |          | | more info                                                   |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | efficiency            | | float  | | Target efficiency for INS - see papers                      | | 0.1    |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | update_interval       | | integer| | Frequency of printed output from inside multinest           | | 200    |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | max_iterations        | | integer| | Maximum number of samples to take                           |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | mode_ztolerance       | | float  | | If multi-modal, get separate stats for modes with this      | | 0.5    |
|                         |          | | evidence difference                                         |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | log_zero              | | float  | | Log-probabilities lower than this value are considered to be| | -1e5   |
|                         |          | | -infinity                                                   |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | cluster_dimensions    | | integer| | Look for multiple modes only on the first dimensions        | | -1     |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | max_modes             | | integer| | If multi-modal, maximum number of allowed modes             | | 100    |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | mode_separation       | | bool   | | Optimize for multi-modal or other odd likelihoods - split   | | N      |
|                         |          | | into different proposal modes                               |          |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | constant_efficiency   | | bool   | | Constant efficiency mode - see papers                       | | N      |
+-------------------------+----------+---------------------------------------------------------------+----------+
| | tolerance             | | float  | | Target error on evidence                                    | | 0.1    |
+-------------------------+----------+---------------------------------------------------------------+----------+
