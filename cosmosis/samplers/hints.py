
from builtins import object
class Hints(object):
    def __init__(self):
        self._peak = None
        self._cov = None

    def has_peak(self):
        return self._peak is not None
    def set_peak(self, peak):
        self._peak = peak
    def get_peak(self):
        return self._peak
    def del_peak(self):
        self._peak = None

    def has_cov(self):
        return self._cov is not None
    def set_cov(self, cov):
        self._cov = cov
    def get_cov(self):
        return self._cov
    def del_cov(self):
        self._cov = None
    
    def update(self, other):
        if other.has_peak():
            self.set_peak(other.get_peak())
        if other.has_cov():
            self.set_cov(other.get_cov())