from __future__ import print_function
from builtins import object
from . import lazy_pylab as pylab


class PostprocessProduct(object):
    def __init__(self, name, filename, value, info=None):
        self.name = name
        self.filename = filename
        self.value = value
        self.info = info

    def finalize(self):
        pass

class PostprocessPlot(PostprocessProduct):
    def finalize(self):
        pylab.figure(self.value.number)
        pylab.savefig(self.filename)
        pylab.close()

    def tweak(self, tweak):
        print("Tweaking", self.name)
        pylab.figure(self.value.number)
        tweak.info = self.info
        tweak.run()


class PostprocessText(PostprocessProduct):
    def finalize(self):
        self.value.close()
