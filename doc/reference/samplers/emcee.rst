The Emcee sampler
--------------------------------------------------------------------

Ensemble walker sampling

+--------------+--------------------------------------+
| | Name       | | emcee                              |
+--------------+--------------------------------------+
| | Version    | | 2.1.0                              |
+--------------+--------------------------------------+
| | Author(s)  | | Dan Foreman-Mackey and contributors|
+--------------+--------------------------------------+
| | URL        | | http://dan.iel.fm/emcee/           |
+--------------+--------------------------------------+
| | Citation(s)| | PASP, 125, 925, 306-312            |
+--------------+--------------------------------------+
| | Parallelism| | parallel                           |
+--------------+--------------------------------------+

The emcee sampler is a form of Monte-Carlo Markov Chain that uses an ensemble of 'walkers' that explore the parameter space.  Each walker chooses another walker at random and proposes along the line connecting the two of them using the Metropolis acceptance rule. The proposal scale is given by the separation of the two walkers.

It is parallel, so multiple processes can be used to speed up the  running. It is also affine invariant, so that no covariance matrix or other  tuning is required for the proposal.

The burn-in behavior of emcee can sometimes be poor; it is much better to start the chain as near to the maximum posterior as possible.  The  maxlike or similar samplers can help you find this.

The total number of samples taken is walkers*samples.



Installation
============

emcee is included in the cosmosis bootstrap, but if you are installing manually you can get emcee using the command:

pip install emcee  #to install centrally, may require sudo

pip install emcee --user #to install just for you




Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

+---------------+----------+-------------------------------------------------------------+----------+
| | Parameter   | | Type   | | Meaning                                                   | | Default|
+---------------+----------+-------------------------------------------------------------+----------+
| | random_start| | bool   | | whether to start the walkers at random points in the prior| | N      |
|               |          | | instead of near the start.  Usually a bad idea            |          |
+---------------+----------+-------------------------------------------------------------+----------+
| | covmat      | | string | | a file containing a covariance matrix for initializing the| | (empty)|
|               |          | | walkers.                                                  |          |
+---------------+----------+-------------------------------------------------------------+----------+
| | samples     | | integer| | number of jumps to attempt per walker                     |          |
+---------------+----------+-------------------------------------------------------------+----------+
| | start_points| | string | | a file containing starting points for the walkers. If not | | (empty)|
|               |          | | specified walkers are initialized randomly from the prior |          |
|               |          | | distribution.                                             |          |
+---------------+----------+-------------------------------------------------------------+----------+
| | nsteps      | | integer| | number of sample steps taken in between writing output    |          |
+---------------+----------+-------------------------------------------------------------+----------+
| | walkers     | | integer| | number of walkers in the space                            |          |
+---------------+----------+-------------------------------------------------------------+----------+
