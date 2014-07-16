from .elements import PostProcessorElement
from ..plotting import cosmology_theory_plots

class Plots(PostProcessorElement):
	def run(self):
		print "I do not know how to generate plots for this kind of data"
		return []

class GridPlots(Plots):
	pass

class MetropolisHastingsPlots(Plots):
	pass

class TestPlots(Plots):
	def run(self):
		ini=self.source.ini
		dirname = ini.get("test", "save_dir")
		output_dir = '.'
		prefix=''
		ftype='png'
		filenames = []
		for cls in cosmology_theory_plots.plot_list:
			try:
				cls.make(dirname, output_dir, prefix, ftype, quiet=True)
				filenames.append(cls.filename+"."+ftype)
			except IOError as err:
				print err
		return filenames


class MultinestPlots(Plots):
	pass
