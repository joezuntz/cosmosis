from __future__ import print_function
from builtins import zip
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['font.size']=18
matplotlib.rcParams['legend.fontsize']=15
matplotlib.rcParams['xtick.major.size'] = 10.0
matplotlib.rcParams['xtick.minor.size'] = 5.0
matplotlib.rcParams['ytick.major.size'] = 10.0
matplotlib.rcParams['ytick.minor.size'] = 5.0

import pylab
import scipy.stats
import numpy as np
import os
import sys
import collections

from .kde import KDE
from .plotter import Plotter

import scipy.optimize
try:
	from cosmosis import output as output_module
except ImportError:
	print("Running without cosmosis: no pretty section names or running on ini files")

class ChainPlotter(Plotter):

	@classmethod
	def from_outputs(cls, options, **kw):
		burn = kw.pop("burn", 0)
		thin = kw.pop("thin", 1)
		
		column_names, chains, metadata, comments, final_metadata = output_module.input_from_options(options)

		if burn==0:
			pass
		elif burn<1:
			for i,chain in enumerate(chains):	
				print("Burning fraction %f of chain %d, which is %d samples" %(burn,i,int(burn*len(chain[:,0]))))
			chains = [chain[int(burn*len(chain[:,0])):, :] for chain in chains]
		else:
			burn = int(burn)
			chains = [chain[burn:,:] for chain in chains]

	#In this case all the chains are assumed to be from a single
	#run.  So we should concatenate them all for a single 
		chains = np.vstack(chains).T
		chains = dict(list(zip(column_names,chains)))
		chain_data = {"Chains":chains}
		return cls(chain_data, **kw)


	@classmethod
	def from_chain_files(cls, filenames, burn=0, thin=1, **kw):
		nameset = [open(filename).readline().strip().strip('#').replace("/","--").split() for filename in filenames]
		dataset = [(np.loadtxt(filename).T) for filename in filenames]
		if burn==0:
			pass
		elif burn<1:

			for name,data in zip(filenames,dataset):	
				print("Burning fraction %f of chain %s, which is %d samples" %(burn,name,int(burn*len(data[0]))))
			dataset = [data[:,int(burn*len(data[0])):] for data in dataset]
		else:
			burn = int(burn)
			dataset = [data[:,burn:] for data in dataset]

		if thin!=1:
			dataset = [data[:,::thin] for data in dataset]

		chain_data = collections.OrderedDict()
		for filename,names,data in zip(filenames,nameset,dataset):
			chain_datum = dict(list(zip(names,data)))
			chain_data[filename] = chain_datum	
		return cls(chain_data, **kw)

	def _plot_1d(self, name, xmin_input=None, xmax_input=None, n=100, factor=2.0):
		for i,(filename, x) in enumerate(self.cols_for_name(name).items()):
			if xmin_input is None:
				xmin = x.min()
			else:
				xmin=xmin_input[i]
			if xmax_input is None:
				xmax = x.max()
			else:
				xmax=xmax_input[i]

			kde = KDE(x, factor=factor)
			x_axis, like = kde.grid_evaluate(n, (xmin,xmax) )
			#need to save plot_data called "like" here
			pylab.plot(x_axis, like, '-', color=self.colors[i], **self.plot_keywords_1d())
		pylab.xlabel("$"+self._display_names[name]+"$")
		if self.blind: self.blind_axes()
	
	@staticmethod
	def _find_contours_corrected(like, x, y, n, xmin, xmax, ymin, ymax, contour1, contour2):
		N = len(x)
		x_axis = np.linspace(xmin, xmax, n+1)
		y_axis = np.linspace(ymin, ymax, n+1)
		histogram, _, _ = np.histogram2d(x, y, bins=[x_axis, y_axis])


		def objective(limit, target):
			w = np.where(like>limit)
			count = histogram[w]
			return count.sum() - target
		target1 = N*(1-contour1)
		target2 = N*(1-contour2)
		level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
		level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
		return level1, level2, like.sum()


	@staticmethod
	def _find_contours(like, contour1, contour2):
		total_mass = like.sum()
		like_sorted = np.sort(like.flatten())
		like_cumsum = like_sorted.cumsum()
		height1 = np.interp(contour1/total_mass,like_cumsum,like_sorted)
		height2 = np.interp(contour2/total_mass,like_cumsum,like_sorted)
		return height1, height2, total_mass


	def _plot_2d(self, name1, name2, xmin_input=None, xmax_input=None, ymin_input=None, ymax_input=None, n=100, factor=2.0):
		cols1 = self.cols_for_name(name1)
		cols2 = self.cols_for_name(name2)
		fill = self.options.get('fill',True)
		for i,(filename,x) in enumerate(cols1.items()):
			if filename not in cols2: continue
			y = cols2[filename]
			if xmin_input is None:
				xmin = x.min()
			else:
				xmin=xmin_input[i]
			if xmax_input is None:
				xmax = x.max()
			else:
				xmax=xmax_input[i]

			if ymin_input is None:
				ymin = y.min()
			else:
				ymin=ymin_input[i]
			if ymax_input is None:
				ymax = y.max()
			else:
				ymax=ymax_input[i]

			kde = KDE([x,y], factor=factor)
			(x_axis,y_axis), like = kde.grid_evaluate(n, [(xmin,xmax),(ymin,ymax)] )
			like/=like.sum()
			contour1=1-0.68
			contour2=1-0.95
			level1, level2, total_mass = self._find_contours_corrected(like, x, y, n, xmin, xmax, ymin, ymax, contour1, contour2)
			level0 = 1.1
			levels = [level2, level1, level0]
			color=self.colors[i]
			if fill:
				pylab.contourf(x_axis, y_axis, like.T, [level2,level0], colors=[color], alpha=0.25)
				pylab.contourf(x_axis, y_axis, like.T, [level1,level0], colors=[color], alpha=0.25)
			else:
				pylab.contour(x_axis, y_axis, like.T, [level2,level1], colors=color)
		pylab.xlabel("$"+self._display_names[name1]+"$")
 		pylab.ylabel("$"+self._display_names[name2]+"$")
 		if "limits" in self.options and name1	in self.options["limits"]:
 			pylab.xlim(*self.options["limits"][name1])
 		if "limits" in self.options and name1	in self.options["limits"]:
 			pylab.ylim(*self.options["limits"][name2])
 		if "truth" in self.options and name1 in self.options["truth"] and name2 in self.options["truth"]:
 			xc = self.options["truth"][name1]
 			yc = self.options["truth"][name2]
			pylab.plot([xc],[yc],'k*', markersize=10)
 		if self.blind: self.blind_axes()
 		# self.command('pylab.xlabel("${0}$")',self._display_names[name2])




	def plot_keywords_1d(self):
		return dict(lw=5)

	def w0_wa_plot(self):
		print("Doing W0 - WA plot")
		w_name  = 'cosmological_parameters--w'
		wa_name = 'cosmological_parameters--wa'
		self._plot_2d(w_name, wa_name, fill=self.options.get('fill',True), factor=1.5)
		w_min,w_max = self.parameter_range(w_name)
		wa_min,wa_max = self.parameter_range(wa_name)
		w_axis = np.linspace(w_min,w_max,2)
		wa_axis = np.linspace(wa_min,wa_max,2)
		pylab.plot(w_axis,[0.0, 0.0],'--',color='gray',dashes=(20,20))
		pylab.plot([-1.0, -1.0], wa_axis,'--',color='gray',dashes=(20,20))
		pylab.plot([-1.0],[0.0],'k*',markersize=10)

		output_name = "%s/%sW0_WA.%s" % (self.root_dir,self.prefix,self.filetype,)
		pylab.savefig(output_name)
		pylab.close()
