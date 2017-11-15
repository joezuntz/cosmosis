The Kombine sampler
--------------------------------------------------------------------

Clustered KDE

+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Name       | | kombine                                                                                                                                              |
+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Version    | | 0.01                                                                                                                                                 |
+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Author(s)  | | Benjamin Farr                                                                                                                                        |
+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| | URL        | | https://github.com/bfarr/kombine                                                                                                                     |
+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Citation(s)| | Farr, B. and Farr, W.M., "kombine: a kernel-density-based, embarrassingly parallel ensemble sampler", in preparation., http://arxiv.org/abs/1309.7709|
+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| | Parallelism| | parallel                                                                                                                                             |
+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+

kombine is an ensemble sampler that uses a clustered kernel-density-estimate proposal density, which allows it to efficiently sample multimodal or non-gaussian posteriors. In between updates to the proposal density estimate, each member of the ensemble is sampled independently, allowing for massive parallelization.

The total number of samples generated will be walkers * samples.



Installation
============

kombine needs to be installed separately: it can be installed from github using pip:

pip install --user git+git://github.com/bfarr/kombine




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+------------------+----------+------------------------------------------------------------+----------+
| | Parameter      | | Type   | | Meaning                                                  | | Default|
+------------------+----------+------------------------------------------------------------+----------+
| | update_interval| | integer| | number of steps taken in between updating the posterior  |          |
+------------------+----------+------------------------------------------------------------+----------+
| | nsteps         | | integer| | number of sample steps taken in between writing output   |          |
+------------------+----------+------------------------------------------------------------+----------+
| | walkers        | | integer| | number of independent walkers in the ensemble            |          |
+------------------+----------+------------------------------------------------------------+----------+
| | samples        | | integer| | total sample steps taken                                 |          |
+------------------+----------+------------------------------------------------------------+----------+
| | start_file     | | string | | a file containing starting points for the walkers. If not| | (empty)|
|                  |          | | specified walkers are initialized randomly from the prior|          |
|                  |          | | distribution.                                            |          |
+------------------+----------+------------------------------------------------------------+----------+
