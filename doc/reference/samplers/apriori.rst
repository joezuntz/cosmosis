The Apriori sampler
--------------------------------------------------------------------

Draw samples from the prior and evaluate the likelihood

+--------------+---------------------------+
| | Name       | | apriori                 |
+--------------+---------------------------+
| | Version    | | 1.0                     |
+--------------+---------------------------+
| | Author(s)  | | CosmoSIS Team           |
+--------------+---------------------------+
| | URL        | | http://dan.iel.fm/emcee/|
+--------------+---------------------------+
| | Citation(s)|                           |
+--------------+---------------------------+
| | Parallelism| | parallel                |
+--------------+---------------------------+

This sampler draws samples from the prior and evaluates their likelihood. The main current use of this is to help test for misbehaviour in calculation modules in extreme regions of parameter space.



Installation
============

None required


Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+------------+----------+-------------------------------------------------------+----------+
| | Parameter| | Type   | | Meaning                                             | | Default|
+------------+----------+-------------------------------------------------------+----------+
| | save_name| | string | | If set, save sample data to directories save_name_0,| | (empty)|
|            |          | | save_name_1, etc.                                   |          |
+------------+----------+-------------------------------------------------------+----------+
| | nsample  | | integer| | number of samples to draw                           |          |
+------------+----------+-------------------------------------------------------+----------+
