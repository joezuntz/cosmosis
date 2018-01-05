The Grid sampler
--------------------------------------------------------------------

Simple grid sampler

+--------------+------------------------------------------+
| | Name       | | grid                                   |
+--------------+------------------------------------------+
| | Version    | | 1.0                                    |
+--------------+------------------------------------------+
| | Author(s)  | | CosmoSIS Team                          |
+--------------+------------------------------------------+
| | URL        | | https://bitbucket.org/joezuntz/cosmosis|
+--------------+------------------------------------------+
| | Citation(s)|                                          |
+--------------+------------------------------------------+
| | Parallelism| | embarrassing                           |
+--------------+------------------------------------------+

Grid sampling is the simplest and most brute-force way to explore a parameter space. It simply builds an even grid of points in parameter space and samples all of them.

As such it scales extremely badly with the number of dimensions in the problem: n_sample = grid_size ^ n_dim and so quickly becomes unfeasible for more than about 4 dimensions.

It is extremely useful, though, in lower dimensions, where it is perfectly parallel and provides a much smoother and more exact picture of the parameter space than MCMC methods do.  It is also useful for taking lines and planes through the parameter space.

The main parameter for the grid sampler is the number of sample points per dimension (grid_size above).



Installation
============

No special installation required; everything is packaged with CosmoSIS




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+--------------------+----------+--------------------------------------------------------------+----------+
| | Parameter        | | Type   | | Meaning                                                    | | Default|
+--------------------+----------+--------------------------------------------------------------+----------+
| | nsample_dimension| | integer| | The number of grid points along each dimension of the space|          |
+--------------------+----------+--------------------------------------------------------------+----------+
| | save             | | string | | If set, a base directory or .tgz name for saving the       | | (empty)|
|                    |          | | cosmology output for every point in the grid               |          |
+--------------------+----------+--------------------------------------------------------------+----------+
