"""
Stop!
This file is not ready yet.

I think I need to re-write the plotting code almost from scratch
as it's grown pretty organically.

"""

import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['font.size']=18
matplotlib.rcParams['legend.fontsize']=15
matplotlib.rcParams['xtick.major.size'] = 10.0
matplotlib.rcParams['xtick.minor.size'] = 5.0
matplotlib.rcParams['ytick.major.size'] = 10.0
matplotlib.rcParams['ytick.minor.size'] = 5.0
import pylab
import numpy as np
try:
	from cosmosis import output as output_module
except ImportError:
	print "Running without cosmosis: no pretty section names or running on ini files"


class GridPlotter(object):
	def __init__(self, *args, **kwargs):
		super(GridPlotter, self).__init__(*args, **kwargs)
		# convert the loaded chain data sets into grids
		# need the number of varied parameters
		# how do we get that?

	@classmethod
	def from_outputs(cls, options, **kw):
		column_names, chains, metadata, comments, final_metadata = output_module.input_from_options(options)
		chains = np.vstack(chains).T
		chains = dict(zip(column_names,chains))
		chain_data = {"Chains":chains}
		return cls(chain_data, **kw)

	@classmethod
	def from_chain_files(cls, filenames, **kw):
		nameset = [open(filename).readline().strip().strip('#').replace("/","--").split() for filename in filenames]
		dataset = [(np.loadtxt(filename).T) for filename in filenames]
		chain_data = collections.OrderedDict()
		for filename,names,data in zip(filenames,nameset,dataset):
			chain_datum = dict(zip(names,data))
			chain_data[filename] = chain_datum	
		return cls(chain_data, **kw)

	def _plot_1d(self, name):
		# for the 1D plot we need to marginalize by simply summing 
		# all the 
