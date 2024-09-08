import numpy as np

class Hints(object):
    def __init__(self):
        self._peak = None
        self._cov = None
        self._peak_post = None

    def has_peak(self):
        return self._peak is not None
    def set_peak(self, peak, post):
        if self._peak_post is None or post > self._peak_post:
            self._peak = peak
            self._peak_post = post

    def set_peak_from_sample(self, samples, posts):
        assert len(samples) == len(posts)
        idx = np.argmax(posts)
        self.set_peak(samples[idx], posts[idx])
    def get_peak(self):
        return self._peak
    def del_peak(self):
        self._peak = None
        self._peak_post = None
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
            self.set_peak(other.get_peak(), other._peak_post)
        if other.has_cov():
            self.set_cov(other.get_cov())