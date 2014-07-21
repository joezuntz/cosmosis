import numpy as np
import argparse

class EverythingIsNan(object):
    def __getitem__(self, param):
        return np.nan

everythingIsNan = EverythingIsNan()

class ParseExtraParameters(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        result = {}
        for arg in values:
            section, param_value = arg.split('.',1)
            param,value = param_value.split('=',1)
            result[(section,param)] = value
        setattr(args, self.dest, result)
