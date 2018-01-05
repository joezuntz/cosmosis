The Maxlike sampler
--------------------------------------------------------------------

Find the maximum likelihood using various methods in scipy

+--------------+------------------------------------------------------------------------------------------+
| | Name       | | maxlike                                                                                |
+--------------+------------------------------------------------------------------------------------------+
| | Version    | | 1.0                                                                                    |
+--------------+------------------------------------------------------------------------------------------+
| | Author(s)  | | SciPy developers                                                                       |
+--------------+------------------------------------------------------------------------------------------+
| | URL        | | http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.minimize.html|
+--------------+------------------------------------------------------------------------------------------+
| | Citation(s)|                                                                                          |
+--------------+------------------------------------------------------------------------------------------+
| | Parallelism| | serial                                                                                 |
+--------------+------------------------------------------------------------------------------------------+

This sampler attempts to find the single point in parameter space with the highest likelihood.  It wraps a variety of samplers from the scipy.minimize package.

These methods are all iterative and local, not global, so they can only find the  nearest local maximum likelihood point from the parameter starting position.

Maximum likelihood using these kinds of methods can be something of a dark art. Results can be quite sensitive to the starting position and exact parameters used, so if you need precision ML then you should carefully explore the robustness of your results.

These samplers are wrapped in the current version of scipy:

- Nelder-Mead

- Powell

- CG

- BFGS

- Newton-CG

- Anneal (deprecated by scipy)

- L-BFGS-B

- TNC

- COBYLA

- SLSQP

- dogleg

- trust-ncg



Each has different (dis)advantages, and which works best will depend on your particular application.  The default in CosmoSIS is Nelder-Mead. See the references on the scipy URL above for more details.

Some methods can also output an estimated covariance matrix at the likelihood  peak.



Installation
============

Requires SciPy 0.14 or above.  This is installed by the CosmoSIS bootstrap, but if you are installing manually you can get it with the command:

pip install scipy  #to install centrally, may require sudo

pip install scipy --user #to install just for you




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+----------------+----------+---------------------------------------------------------------+--------------+
| | Parameter    | | Type   | | Meaning                                                     | | Default    |
+----------------+----------+---------------------------------------------------------------+--------------+
| | output_ini   | | string | | if present, save the resulting parameters to a new ini file | | (empty)    |
|                |          | | with this name                                              |              |
+----------------+----------+---------------------------------------------------------------+--------------+
| | maxiter      | | integer| | Maximum number of iterations of the sampler                 | | 1000       |
+----------------+----------+---------------------------------------------------------------+--------------+
| | tolerance    | | real   | | The tolerance parameter for termination.  Meaning depends on| | 1e-3       |
|                |          | | the sampler - see scipy docs.                               |              |
+----------------+----------+---------------------------------------------------------------+--------------+
| | output_covmat| | string | | if present and the sampler supports it, save the estimated  | | (empty)    |
|                |          | | covariance to this file                                     |              |
+----------------+----------+---------------------------------------------------------------+--------------+
| | method       | | string | | The minimization method to use.                             | | Nelder-Mead|
+----------------+----------+---------------------------------------------------------------+--------------+
