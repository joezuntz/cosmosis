from __future__ import print_function
from builtins import object
import sys

"""
This thing is pretty cool - I found it on StackOverflow.

This is a lazily loaded module that provides an interface to pylab.

If you do:

import lazy_pylab as pylab

then you can use pylab in exactly the normal way:
pylab.plot(range(10))
etc., but pylab itself will not be loaded until the first
access of a pylab function or other attribute.

Since pylab can take a relatively long time to load this 
can save time if you never actually end up using it in a
particular run.

"""

class _LazyPylab(object):
    def __init__(self):
        self.first=True
    
    def initialize_matplotlib(self):
        try:
            import logging
            logger = logging.getLogger('matplotlib')
            # set WARNING for Matplotlib
            logger.setLevel(logging.WARNING)
            import matplotlib

        except ImportError:
            print()
            print("No matplotlib: plotting unavailable.")
            print("If you are using postprocess you can disable")
            print("plotting with the --no-plots option")
            print()
            print("Unable to continue cleanly so quitting now.")
            print()
            import sys
            sys.exit(1)
        #Some options are only available in newer
        #matplotlibs.
        try:
            matplotlib.rcParams['figure.max_open_warning'] = 100
        except:
            pass
        matplotlib.rcParams['figure.figsize'] = (8,6)
        matplotlib.rcParams['font.family']='serif'
        matplotlib.rcParams['font.size']=18
        matplotlib.rcParams['legend.fontsize']=15
        matplotlib.rcParams['xtick.major.size'] = 10.0
        matplotlib.rcParams['xtick.minor.size'] = 5.0
        matplotlib.rcParams['ytick.major.size'] = 10.0
        matplotlib.rcParams['ytick.minor.size'] = 5.0
        matplotlib.rcParams['figure.subplot.bottom'] = 0.125
        matplotlib.rcParams['figure.subplot.left'] = 0.175
        matplotlib.use("Agg")

    def __getattr__(self, name):
        if self.first:
            self.initialize_matplotlib()
            self.first=False
        import pylab
        return getattr(pylab, name)


sys.modules[__name__] = _LazyPylab()
