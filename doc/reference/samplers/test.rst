The Test sampler
--------------------------------------------------------------------

Evaluate a single parameter set

+--------------+------------------------------------------+
| | Name       | | test                                   |
+--------------+------------------------------------------+
| | Version    | | 1.0                                    |
+--------------+------------------------------------------+
| | Author(s)  | | CosmoSIS Team                          |
+--------------+------------------------------------------+
| | URL        | | https://bitbucket.org/joezuntz/cosmosis|
+--------------+------------------------------------------+
| | Citation(s)|                                          |
+--------------+------------------------------------------+
| | Parallelism| | serial                                 |
+--------------+------------------------------------------+

This is the most trivial possible 'sampler' - it just runs on a single parameter sample. It is mainly useful for testing and for generating  cosmology results for plotting.

The test sampler uses the starting position defined in the value ini file, and runs the pipeline just on that.

At the end of the run it will print out the prior and likelihood (if there is one), and can optionally also save all the data saved along the pipeline, so that you can make plots of the useful cosmological quantities.

Experimental: we have a new test feature where you can plot your pipeline and the data flow through it as a graphical diagram.  This requires pygraphviz and the graphviz suite to make an image - use a command like: dot -Tpng -o graph.png graph.dot



Installation
============

No special installation required; everything is packaged with CosmoSIS. If you want to make a graphical diagram of your pipeline you need to have pygraphviz installed.  You can get this with:

pip install pygraphviz  #to install centrally, may require sudo

pip install pygraphviz --user #to install just for you

You also need graphviz to turn the result into an image.




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+---------------+---------+---------------------------------------------------------------+----------+
| | Parameter   | | Type  | | Meaning                                                     | | Default|
+---------------+---------+---------------------------------------------------------------+----------+
| | fatal_errors| | bool  | | Any errors in the pipeline trigger an immediate failure so  | | N      |
|               |         | | you can diagnose errors                                     |          |
+---------------+---------+---------------------------------------------------------------+----------+
| | graph       | | string| | Requires pygraphviz.  Save a dot file describing the        | | (empty)|
|               |         | | pipeline                                                    |          |
+---------------+---------+---------------------------------------------------------------+----------+
| | save_dir    | | string| | Save all the data computed in the pipeline to this directory| | (empty)|
|               |         | | (can also end with .tgz to get a zipped form)               |          |
+---------------+---------+---------------------------------------------------------------+----------+
