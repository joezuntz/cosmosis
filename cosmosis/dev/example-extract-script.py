from cosmosis.postprocessing.postprocess import postprocessor_for_sampler
from cosmosis.runtime.config import Inifile
from cosmosis.runtime.parameter import Parameter
import sys

# Set up a quick test example
pp_class = postprocessor_for_sampler('emcee')
ini1=Inifile("./demos/demo5.ini")
postprocessor = pp_class(ini1, "demo5", 0)


# actual example
values_ini = postprocessor.extract_ini("VALUES")
params = Parameter.load_parameters(values_ini)

for param in params:
    print(param, param.limits)
