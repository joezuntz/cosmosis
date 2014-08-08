import sys


class _LazyPylab(object):
	def __init__(self):
		self.first=True
	
	def initialize_matplotlib(self):
		import matplotlib
		matplotlib.rcParams['figure.max_open_warning'] = 100
		matplotlib.rcParams['figure.figsize'] = (8,6)
		matplotlib.rcParams['font.family']='serif'
		matplotlib.rcParams['font.size']=18
		matplotlib.rcParams['legend.fontsize']=15
		matplotlib.rcParams['xtick.major.size'] = 10.0
		matplotlib.rcParams['xtick.minor.size'] = 5.0
		matplotlib.rcParams['ytick.major.size'] = 10.0
		matplotlib.rcParams['ytick.minor.size'] = 5.0
		matplotlib.use("Agg")

	def __getattr__(self, name):
		if self.first:
			self.initialize_matplotlib()
			self.first=False
		import pylab
		return getattr(pylab, name)


sys.modules[__name__] = _LazyPylab()
