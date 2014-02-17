#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex']=True
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
import ConfigParser
import pydesglue
import itertools
import cmd
if sys.version_info[1]==7:
	import collections
else:
	import ordereddict as collections

try:
	import argparse
except ImportError:
	sys.stderr.write("You need the argparse module to run plot_chains.\nTry 'pip install argparse' \n")
	sys.exit(1)

class KDE(scipy.stats.kde.gaussian_kde):
	def __init__(self, points, factor=1.0):
		points = np.array(np.atleast_2d(points))

		self._factor = factor
		d, n = points.shape
		self.norms = []
		normalized_points = []
		assert d<=2, "KDE is for 1D 2D only"
		for column in points:
			col_mean = column.mean()
			col_std = column.std()
			self.norms.append((col_mean,col_std))
			normalized_points.append((column-col_mean)/col_std)
		super(KDE,self).__init__(normalized_points)

	def covariance_factor(self):
		return self.scotts_factor() * self._factor

	def grid_evaluate(self, n, ranges):
		if isinstance(ranges,tuple):
			ranges = [ranges]
		slices = [slice(xmin,xmax,n*1j) for (xmin,xmax) in ranges]
		grids = np.mgrid[slices]
		axes = [ax.squeeze() for ax in np.ogrid[slices]]
		flats = [(grid.flatten()-norm[0])/norm[1] 
		         for (grid,norm) in zip(grids,self.norms)]

		shape = grids[0].shape
		flats = np.array(flats)
		like_flat = self.evaluate(flats)
		like = like_flat.reshape(*shape)
		if len(axes)==1:
			axes = axes[0]
		return axes,like


#KDE = 

class NoSuchParameter(Exception):
	pass

# class Plot(object):
# 	pass
# 	def save(self, filename):
# 		self.ax.

# class Plot1d(object):
# 	def 

def section_code(code):
	for att in dir(pydesglue.section_names):
		if getattr(pydesglue.section_names,att)==code:
			return att

class Plotter(object):
	colors=['blue','red','green','cyan','gray']
	def __init__(self, chain_data,latex_file=None, filetype="png", root_dir='.',**options):
		self._chain_data = chain_data
		all_names = set()
		for chain_datum in self._chain_data.values():
			for name in chain_datum.keys():
				all_names.add(name)
		self.all_names = sorted(list(all_names))
		self.load_latex(latex_file)
		self.nfile = len(chain_data)
		self.filetype=filetype
		self.options=options
		self.root_dir = root_dir

	def command(self, command, *args, **kwargs):
		cmd = command.format(*args, **kwargs) + '\n'
		self._output_file.write(cmd)

	def load_latex(self, latex_file):
		self._display_names = {}
		if latex_file is not None:
			latex_names = ConfigParser.ConfigParser()
			latex_names.read(latex_file)
		for i,col_name in enumerate(self.all_names):
			display_name=col_name
			if '--' in col_name:
				section,name = col_name.split('--')
				try:
					display_name = latex_names.get(section,name)
				except ConfigParser.NoSectionError:
					section = section_code(section)
				except ConfigParser.NoOptionError:
					pass					
				try:
					display_name = latex_names.get(section,name)
				except:
					pass
			else:
				if col_name in ["LIKE","likelihood"]:
					display_name=r"{\cal L}"
			self._display_names[col_name]=display_name



	@classmethod
	def from_chain_files(cls, filenames, burn=0, thin=1, **kw):
		nameset = [open(filename).readline().strip().strip('#').replace("/","--").split() for filename in filenames]
		dataset = [(np.loadtxt(filename).T) for filename in filenames]
		if burn==0:
			pass
		elif burn<1:

			for name,data in zip(filenames,dataset):	
				print "Burning fraction %f of chain %s, which is %d samples" %(burn,name,int(burn*len(data[0])))
			dataset = [data[:,int(burn*len(data[0])):] for data in dataset]
		else:
			burn = int(burn)
			dataset = [data[:,burn:] for data in dataset]

		if thin!=1:
			dataset = [data[:,::thin] for data in dataset]

		chain_data = collections.OrderedDict()
		for filename,names,data in zip(filenames,nameset,dataset):
			chain_datum = dict(zip(names,data))
			chain_data[filename] = chain_datum	
		return cls(chain_data, **kw)

	def cols_for_name(self, name):
		cols = collections.OrderedDict()
		for filename, chain_datum in self._chain_data.items():
			if name in chain_datum.keys():
				cols[filename] = chain_datum[name]
		if not cols:
			raise NoSuchParameter(name)
		return cols

	def parameter_range(self, name):
		cols = self.cols_for_name(name)
		xmin = 1e30
		xmax = -1e30
		for col in cols.values():
			if col.min() < xmin: xmin=col.min()
			if col.max() > xmax: xmax=col.max()
		if xmin==1e30 or xmax==-1e30:
			raise ValueError("Could not find col max/min - NaNs in chain?")
		return xmin,xmax


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
		level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,), xtol=1./N)
		level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,), xtol=1./N)
		return level1, level2, like.sum()


	@staticmethod
	def _find_contours(like, contour1, contour2):
		total_mass = like.sum()
		like_sorted = np.sort(like.flatten())
		like_cumsum = like_sorted.cumsum()
		height1 = np.interp(contour1/total_mass,like_cumsum,like_sorted)
		height2 = np.interp(contour2/total_mass,like_cumsum,like_sorted)
		return height1, height2, total_mass


	def _plot_2d(self, name1, name2, xmin_input=None, xmax_input=None, ymin_input=None, ymax_input=None, n=100, factor=2.0, fill=True):
		cols1 = self.cols_for_name(name1)
		cols2 = self.cols_for_name(name2)
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
 		# self.command('pylab.xlabel("${0}$")',self._display_names[name2])


 	def plot_1d_params(self, names):
 		if not names:
 			names = self.all_names
 		for name in names:
 			print "Plotting 1D curve for ", name
 			try:
	 			self._plot_1d(name)
 				pylab.savefig("%s/%s.%s"%(self.root_dir,name, self.filetype))
 			except Exception as error:
 				print "Unable to plot curve - may be only one value?"
 				print error
 			finally:
	 			pylab.close()

 	def plot_2d_params(self, names):
 		if not names:
 			names = self.all_names
 		for name1 in names:
 			for name2 in names:
 				if name1!=name2 and name1<name2:
		 			print "Plotting 2D curve for ", name1, "versus", name2
		 			try:
	 					self._plot_2d(name1,name2, fill=self.options.get('fill',True))
			 			pylab.savefig("%s/%s_%s.%s"%(self.root_dir,name1,name2,self.filetype))
		 			except Exception as error:
		 				print "Unable to plot contours - may be only one value?"
		 				print error
		 			finally:
			 			pylab.close()


	def plot_keywords_1d(self):
		return dict(lw=5)


class CosmologyPlotter(Plotter):
	def w0_wa_plot(self):
		print "Doing W0 - WA plot"
		w_name  = 'COSMOPAR--W'
		wa_name = 'COSMOPAR--WA'
		self._plot_2d(w_name, wa_name, fill=self.options.get('fill',True), factor=1.5)
		w_min,w_max = self.parameter_range(w_name)
		wa_min,wa_max = self.parameter_range(wa_name)
		w_axis = np.linspace(w_min,w_max,2)
		wa_axis = np.linspace(wa_min,wa_max,2)
		pylab.plot(w_axis,[0.0, 0.0],'--',color='gray',dashes=(20,20))
		pylab.plot([-1.0, -1.0], wa_axis,'--',color='gray',dashes=(20,20))
		pylab.plot([-1.0],[0.0],'k*',markersize=10)

		output_name = "%s/W0_WA.%s" % (self.root_dir,self.filetype,)
		pylab.savefig(output_name)
		pylab.close()

def main(args):
	if args.eps:
		filetype='eps'
	else:
		filetype='png'
	plotter = CosmologyPlotter.from_chain_files(
		args.filenames,
		burn=args.burn,
		thin=args.thin,
		latex_file=args.latex,
		filetype=filetype,
		fill=not args.nofill,
		root_dir=args.root_dir,
		)
	plotter.plot_1d_params(args.params)
	if not args.one_d:
		plotter.plot_2d_params(args.params)
	if args.w0_wa:
		plotter.w0_wa_plot()


description = "Plot marginalized constraints from a des-pipe MCMC output chain."
parser = argparse.ArgumentParser(description=description, add_help=True)

parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',
	help='Chain filenames')
parser.add_argument("-t","--thin", type=int, default=1, 
	help='Thinning to apply to chains (after burn)')
parser.add_argument("-b","--burn", type=float, default=0.2,
	help='Fraction or number of samples to cut at the start of the chain (before thinning)')
parser.add_argument("-l","--latex", type=str, default="",
	help='Ini file containing latex names for parameters')
parser.add_argument('-p', '--param', dest='params', action='append',
	help='List params to make plots of (e.g. COSMOPAR--H0).  Default is to do all.')
parser.add_argument('-1', '--one-d', dest='one_d', action='store_true',
	help='Make only 1D plots, not contours')
parser.add_argument('-w', '--w0-wa', dest='w0_wa', action='store_true',
	help='Make the special W0-Wa plot')
parser.add_argument('-e', '--eps', dest='eps', action='store_true',
	help='Save files as encapsulated postscript (eps)')
parser.add_argument('-L', '--no-fill', dest='nofill', action='store_true',
	help='Produce line contour plots instead of filled')
parser.add_argument('-r', '--root-dir', default='.', 
	help='The root directory in which to put plots')


if __name__=="__main__":
	args = parser.parse_args()
	main(args)
