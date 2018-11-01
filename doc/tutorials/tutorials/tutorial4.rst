Tutorial 4: Building new pipelines
----------------------------------

In CosmoSIS a *pipeline* is a sequence of *modules* that computes one or more *likelihoods*.

You can build a new pipeline or modify an existing one by choosing which modules to include in the pipeline, and by selecting their configuration.

Extending a calculation by adding modules
=========================================

To know which module to add you have to understand the calculation you want to perform.  For example: if you want to compute a Redshift Space Distortion likelihood then you'll need a growth factor first. 

You can add modules into the middle of an existing pipeline, or at the end.