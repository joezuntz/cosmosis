from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline
import numpy as np
import pylab

#The easiest way to start a pipeline it from a parameter file.
#You load in the file, and then build a LikelihoodPipeline from it.
#You could modify things in the ini file object after loading 
#if you wanted.
ini = Inifile("demos/demo15.ini")
pipeline = LikelihoodPipeline(ini)

#You can modify which parameters you vary here.
#In the next cosmosis version there will be a 
#nicer method for doing this.
for parameter in pipeline.parameters:
    if parameter == ("cosmological_parameters", "omega_m"):
        parameter.limits = (0.2, 0.4)

#You can also override these properties if useful
pipeline.quiet = True
pipeline.debug = False
pipeline.timing = False

#Let's look through different values of omega_m
#and get a Galaxy Galaxy-Lensing spectrum for each of them
for omega_m in [0.2, 0.25, 0.3, 0.35, 0.4]:

    #In this method of running the pipeline we
    #pass it a value for each of the parameters 
    #we have told it to vary.
    #We could check what these are by looking at 
    #pipeline.varied_params
    data = pipeline.run_parameters([omega_m])

    #data is a DataBlock - can get things out of it as in any
    #cosmosis module:
    ell = data['ggl_cl', 'ell']
    cl  = data['ggl_cl', 'bin_1_1']

    #Make a plot for this value
    pylab.loglog(ell, np.abs(cl), label=str(omega_m))
    print "Done ", omega_m

#Save our plot.
pylab.legend()
pylab.savefig("test_scripting.png")
