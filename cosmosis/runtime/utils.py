import numpy as np


class EverythingIsNan(object):
    def __getitem__(self, param):
        return np.nan

everythingIsNan = EverythingIsNan()
