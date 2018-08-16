Priors Files
============

The priors file is optional, and if desired can be set in the :code:`[pipeline]` section of the main parameter file. The path to it may be specified in the :code:`[pipeline]` section of the main parameter file, using the syntax :code:`priors = path/to/values.ini`.

Like the other parameter files it is in the :code:`ini` format.  All parameters should be in named :code:`[section]s`.



Implicit Priors
---------------

Parameters that appear in the values file always have an implicit uniform prior between the lower and upper limits that you specify there.  

Priors that you specify in the priors ini file act as *additional* priors - for example, if you specify a Gaussian prior here for a parameter then the overall priors will be a Gaussian, truncated at the lower and upper bounds.


Additional Priors
-----------------

Priors to include in addition to the implicit ones are set in the priors ini file.  Here is an example::

    [cosmological_parameters]
    omega_m = uniform 0.0 0.2
    h0 = gaussian 0.72 0.08
    tau = exponential 0.05

As in the values file all parameters must be in a corresponding section.  Priors that are set here but refer to parameters not mentioned in the values file will be ignored.

Currently only independent priors can be specified in CosmoSIS.

Uniform Priors
--------------

Uniform priors specified in the priors file can be used to further restrict parameters compared to their allowed range in the values file.  Otherwise they are a little pointless.

They are specified in the form: :code:`param_name = uniform  <lower_limit>  <upper_limit>`.


Gaussian Priors
---------------

Gaussian priors specified here combine with the implicit priors in the values section to form truncated Gaussian priors. The normalization is correctly adjusted.

They are specified in the form: :code:`param_name = gaussian  <mean>  <std_dev>`.

Exponential Priors
------------------

Again, exponential priors are truncated by the limits in the values file.

The exponential distribution has the form :math:`P(X=x) = \frac{1}{\beta} \exp{(-x/\beta)}` for the parameter :math:`\beta`.

They are specified in the form: :code:`param_name = exponential  <beta>`.


