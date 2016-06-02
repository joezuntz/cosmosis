from . import plots
from . import statistics
from .postprocess_base import PostProcessor, postprocessor_registry
import numpy as np

def postprocessor_for_sampler(sampler):
	return postprocessor_registry.get(sampler)

class MetropolisHastingsProcessor(PostProcessor):
	elements=[
		plots.MetropolisHastingsPlots1D,
		plots.MetropolisHastingsPlots2D,
		statistics.MetropolisHastingsStatistics,
		statistics.MetropolisHastingsCovariance,
		statistics.Citations,
	]
	def reduced_col(self, name, stacked=True):
		cols = self.get_col(name, stacked=False)
		burn = self.options.get("burn", 0)
		thin = self.options.get("thin", 1)

		if 0.0<burn<1.0:
			burn = len(cols[0])*burn
		else:
			burn = int(burn)
		cols = [col[burn::thin] for col in cols]
		if stacked:
			return np.concatenate(cols)
		else:
			return cols

	def posterior_sample(self):
		"""
		A posterior sample of MCMC is just all the samples.

		Return an array of Trues with the same length as the chain

		"""
		n = self.reduced_col(self.colnames[0]).size
		return np.ones(n, dtype=bool)



class EmceeProcessor(MetropolisHastingsProcessor):
	sampler="emcee"
	pass

class KombineProcessor(MetropolisHastingsProcessor):
	sampler="kombine"
	pass


class PymcProcessor(MetropolisHastingsProcessor):
	sampler="pymc"
	pass

class MetropolisProcessor(MetropolisHastingsProcessor):
	sampler="metropolis"
	elements=[
		plots.MetropolisHastingsPlots1D,
		plots.MetropolisHastingsPlots2D,
		statistics.MetropolisHastingsStatistics,
		statistics.MetropolisHastingsCovariance,
		statistics.DunkleyTest,
		statistics.Citations,		
		]


class WeightedMetropolisProcessor(MetropolisHastingsProcessor):
	sampler="weighted_metropolis"
	elements=[
		plots.WeightedMetropolisPlots1D,
		plots.WeightedMetropolisPlots2D,
		statistics.WeightedMetropolisStatistics,
		#statistics.WeightedMetropolisHastingsCovariance,
		#statistics.DunkleyTest,
		statistics.Citations,		
		]

	def weight_col(self):
		if hasattr(self, "_weight_col"):
			return self._weight_col
		if self.has_col("weight"):
			w = MetropolisHastingsProcessor.reduced_col(self, "weight").copy()
			logw = np.log(w)
		elif self.has_col("log_weight"):
			logw = MetropolisHastingsProcessor.reduced_col(self, "log_weight").copy()
		else:
			raise ValueError("No 'weight' or 'log_weight' column found in chain.")


		if self.has_col("old_weight"):
			old_w = MetropolisHastingsProcessor.reduced_col(self, "old_weight").copy()
			old_logw = np.log(old_w)
			logw += old_logw
			print "Including old_weight in weight"
		elif self.has_col("old_log_weight"):
			old_logw = MetropolisHastingsProcessor.reduced_col(self, "old_log_weight").copy()
			logw += old_logw
			print "Including old_log_weight in weight"
		logw-=np.nanmax(logw)
		self._weight_col = np.exp(logw)
		return self._weight_col    

	def posterior_sample(self):
		"""
		Weighted chains are *not* drawn from the posterior distribution
		but we do have the information we need to construct such a sample.

		This function returns a boolean array with True where we should
		use the sample at that index, and False where we should not.

		"""
		w = self.weight_col()
		w = w / w.max()
		u = np.random.uniform(size=w.size)
		return u<w


class ImportanceProcessor(WeightedMetropolisProcessor):
	sampler="importance"
	elements=[
		plots.WeightedMetropolisPlots1D,
		plots.WeightedMetropolisPlots2D,
		statistics.WeightedMetropolisStatistics,
		# statistics.DunkleyTest,
		statistics.Citations,		
		]

class GridProcessor(PostProcessor):
	elements=[
		plots.GridPlots1D,
		plots.GridPlots2D,
		statistics.GridStatistics,
		statistics.Citations,
		]
	sampler="grid"



class TestProcessor(PostProcessor):
	elements = [plots.TestPlots]
	sampler="test"
	cosmosis_standard_output=False

	def load(self, ini):
		if isinstance(ini, str) and os.path.isdir(ini):			
			self._sampler_options['save_dir'] = ini
		else:
			for key,val in ini.items("test"):
				self.sampler_options[key]=str(val)



class MultinestProcessor(WeightedMetropolisProcessor):
	elements = [
		plots.MultinestPlots1D, 
		plots.MultinestPlots2D,
		statistics.MultinestStatistics,
		statistics.MultinestCovariance,
		statistics.Citations,
		]
	sampler="multinest"
	def reduced_col(self, name, stacked=True):
		#we only use the last n samples from a multinest output
		#file.  And omit zero-weighted samples.
		n = int(self.final_metadata[0]["nsample"])
		col = self.get_col(name)
		w = self.get_col("weight")[-n:]
		return col[-n:][w>0]
		
	def weight_col(self):
		if hasattr(self, "_weight_col"):
			return self._weight_col
		n = int(self.final_metadata[0]["nsample"])
		w = self.get_col("weight")[-n:]
		w = w[w>0].copy()
		self._weight_col = w
		return self._weight_col

	def posterior_sample(self):
		"""
		Multinest chains are *not* drawn from the posterior distribution
		but we do have the information we need to construct such a sample.

		This function returns a boolean array with True where we should
		use the sample at that index, and False where we should not.

		"""
		w = self.weight_col()
		w = w / w.max()
		u = np.random.uniform(size=w.size)
		return u<w


class PMCPostProcessor(WeightedMetropolisProcessor):
	sampler="pmc"
	elements = [
		plots.WeightedMetropolisPlots1D,
		plots.WeightedMetropolisPlots2D,
		statistics.WeightedMetropolisStatistics,
		#statistics.WeightedMetropolisHastingsCovariance,
		#statistics.DunkleyTest,
		statistics.Citations,		
	]
	def reduced_col(self, name, stacked=True):
		#we only use the last n samples from a multinest output
		#file.  And omit zero-weighted samples.
		n = int(self.final_metadata[0]["nsample"])
		col = self.get_col(name)
		w = self.get_col("log_weight")[-n:]
		return col[-n:][np.isfinite(w)]
		
	def weight_col(self):
		if hasattr(self, "_weight_col"):
			return self._weight_col
		n = int(self.final_metadata[0]["nsample"])
		w = self.get_col("log_weight")[-n:]
		w = w[np.isfinite(w)].copy()
		self._weight_col = np.exp(w)
		return self._weight_col

	def posterior_sample(self):
		"""
		PMC chains are *not* drawn from the posterior distribution - they have weights.
		We do have the information we need to construct such a sample.

		This function returns a boolean array with True where we should
		use the sample at that index, and False where we should not.

		"""
		w = self.weight_col()
		w = w / w.max()
		u = np.random.uniform(size=w.size)
		return u<w


class SnakeProcessor(PostProcessor):
	sampler='snake'
	elements = [
		plots.GridPlots1D,
		plots.SnakePlots2D,
		statistics.Citations,		
	]

class FisherProcessor(PostProcessor):
	sampler = 'fisher'
	elements = [
		plots.CovarianceMatrixEllipse,
		plots.CovarianceMatrixGaussians,
		statistics.CovarianceMatrix1D,
		statistics.CovarianceMatrixEllipseAreas,
	]