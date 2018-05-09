The Snake sampler
--------------------------------------------------------------------

Intelligent Grid exploration

+--------------+------------------------------------------+
| | Name       | | snake                                  |
+--------------+------------------------------------------+
| | Version    | | 1.0                                    |
+--------------+------------------------------------------+
| | Author(s)  | | CosmoSIS Team                          |
+--------------+------------------------------------------+
| | URL        | | https://bitbucket.org/joezuntz/cosmosis|
+--------------+------------------------------------------+
| | Citation(s)| | ApJ 777 172 (2013)                     |
+--------------+------------------------------------------+
| | Parallelism| | parallel                               |
+--------------+------------------------------------------+

Snake is a more intelligent version of the grid sampler that avoids taking large number of samples with a low likelihood, which the naive grid sampler nearly always does.

It does ultimately have the same bad behaviour as you go to a higher number of dimensions, though you can push it higher than with the straight grid.

The Snake sampler maintains a list of samples on the interior and surface of the parameter combinations it has explored.  This allows it to first move gradually towards the maximum likelihood and then gradually diffuse outwards from that point in all the different dimensions.

Snake outputs can be postprocessed in exactly the same way as grid samples, with missing entries assumed to have zero posterior.



Installation
============

No special installation required; everything is packaged with CosmoSIS




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+--------------------+----------+---------------------------------------------------------+----------+
| | Parameter        | | Type   | | Meaning                                               | | Default|
+--------------------+----------+---------------------------------------------------------+----------+
| | threshold        | | float  | | Termination for difference betwen max-like and highest| | 4.0    |
|                    |          | | surface likelihood                                    |          |
+--------------------+----------+---------------------------------------------------------+----------+
| | nsample_dimension| | integer| | Number of grid points per dimension                   | | 10     |
+--------------------+----------+---------------------------------------------------------+----------+
| | maxiter          | | integer| | Maximum number of samples to take                     | | 100000 |
+--------------------+----------+---------------------------------------------------------+----------+
