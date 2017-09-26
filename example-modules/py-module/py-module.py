
from builtins import object
from numpy import log, pi
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.runtime.declare import declare_module

class PyModule(object):

    cosmo = section_names.cosmological_parameters
    likes = section_names.likelihoods
    hst_h0_mean = 0.738
    hst_h0_sigma = 0.024
    
    def __init__(self,my_config,my_name):
        self.mod_name = my_name
        self.mean = my_config.get_double(my_name, "mean", default=PyModule.hst_h0_mean)
        self.sigma = my_config.get_double(my_name, "sigma", default=PyModule.hst_h0_sigma)
        self.norm = 0.5 * log(2*pi**2)
        # self.example = my_config[my_name, "value"]

    def execute(self, block):
        h0 = block[PyModule.cosmo, "h0"]
        like = -(h0-self.mean)**2 / self.sigma**2 / 2.0 - self.norm
        block[PyModule.likes, "RIESS_LIKE"] = like
        return 0
        
    def cleanup(self):
        return 0

declare_module(PyModule)

