The Importance sampler
--------------------------------------------------------------------

Importance sampling

+--------------+------------------------------------------+
| | Name       | | importance                             |
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

Importance sampling is a general method for estimating quantities from one distribution, P', when what you have is samples from another, similar distribution, P. In IS a weight is calculated for each sample that depends on the difference between the likelihoods under the two distributions.

IS works better the more similar the two distributions are, but can also be useful for adding additional constraints to an existing data set.

There's a nice introduction to the general idea in Mackay ch. 29: http://www.inference.phy.cam.ac.uk/itila/book.html



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
| | input_filename   | | string | | cosmosis-format chain of input samples                |          |
+--------------------+----------+---------------------------------------------------------+----------+
| | nstep            | | integer| | number of samples to do between saving output         | | 128    |
+--------------------+----------+---------------------------------------------------------+----------+
| | add_to_likelihood| | bool   | | include the old likelihood in the old likelihood; i.e.| | N      |
|                    |          | | P'=P*P_new                                            |          |
+--------------------+----------+---------------------------------------------------------+----------+
