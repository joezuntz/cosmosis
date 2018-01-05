The Star sampler
--------------------------------------------------------------------

Simple star sampler

+--------------+------------------------------------------+
| | Name       | | star                                   |
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
| | nsample_dimension| | integer| | The number of star points along each dimension of the space|          |
+--------------------+----------+--------------------------------------------------------------+----------+
| | save             | | string | | If set, a base directory or .tgz name for saving the       | | (empty)|
|                    |          | | cosmology output for every point in the star               |          |
+--------------------+----------+--------------------------------------------------------------+----------+
