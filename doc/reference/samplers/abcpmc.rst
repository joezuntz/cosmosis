The Abcpmc sampler
--------------------------------------------------------------------

Approximate Bayesian Computing (ABC) Population Monte Carlo (PMC) 

+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Name       | | abcpmc                                                                                                                                                            |
+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Version    | | 0.1.1                                                                                                                                                             |
+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Author(s)  | | Joel Akeret and contributors                                                                                                                                      |
+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| | URL        | | http://abcpmc.readthedocs.org/en/latest/                                                                                                                          |
+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Citation(s)| | Akeret, J., Refregier, A., Amara, A, Seehars, S., and Hasner, C., JCAP (submitted 2015), Beaumont et al. 2009 arXiv:0805.2256, Fillippi et al 2012 arXiv:1106.6280|
+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Parallelism| | parallel                                                                                                                                                          |
+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+

abcpmc is a Python Approximate Bayesian Computing (ABC) Population Monte Carlo (PMC) implementation based  on Sequential Monte Carlo (SMC) with Particle Filtering techniques.  This likelihood free implementation estimates the posterior distribution using a model to simulate a  dataset given a set of parameters. A metric rho is used to determine a distance between the model and the data  and parameter values are retained if rho(model,data) < epsilon. This epsilon threshold can be fixed or linearly or exponentially modified every iteration in abcpmc.  abcpmc uses a set of N particles to explore parameter space (theta), on the first iteration, t=0, these are chosen from the prior. On subsequent iterations, t, another N particles are selected with a perturbation kernal K(theta(t) | theta(t-1)) using twice the weighted covariance matrix. It is extendable with k-nearest neighbour (KNN) or optimal local covariance matrix (OLCM)  pertubation kernels.

This implementation of abcpmc in CosmoSIS requires an understanding of how ABC sampling works and we recommend you contact the CosmoSIS team for specific implementaion questions; we would be very happy to help out!



Installation
============

pip install abcpmc  #to install centrally, may require sudo

pip install abcpmc --user #to install just for you




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+-----------------+----------+---------------------------------------------------------------+----------------+
| | Parameter     | | Type   | | Meaning                                                     | | Default      |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | niter         | | integer| | T - number of iterations                                    | |  2           |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | distance_func | | str    | | def func(x,y):\n\t do some calculation\n\t return           | | None         |
|                 |          | | dist_result                                                 |                |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | epimin        | | double | | epsilon at t=T                                              | |  1.0         |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | ngauss        | | int    | | dimension of multigaussian if run_multigauss                | |  4           |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | epimax        | | double | | epsilon at t=0                                              | |  5.0         |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | particle_prop | | string | | Particle proposal kernal, options = weighted_cov, KNN, OLCM | |  weighted_cov|
+-----------------+----------+---------------------------------------------------------------+----------------+
| | npart         | | integer| | number of particles                                         |                |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | run_multigauss| | boolean| | generate multigaussian data                                 | |  F           |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | metric_kw     | | str    | | mean, chi2 or other; if "other" then need to specify name of| |  "chi2"      |
|                 |          | | function "distance_func                                     |                |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | num_nn        | | integer| | number of neighbours if using particle_prop = KNN           | |  10          |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | threshold     | | string | | Various different threshold implementations, options =      | |  LinearEps   |
|                 |          | | LinearEps, ConstEps, ExpEps                                 |                |
+-----------------+----------+---------------------------------------------------------------+----------------+
| | set_prior     | | string | | prior, options = Gaussian, uniform                          | |  Gaussian    |
+-----------------+----------+---------------------------------------------------------------+----------------+
