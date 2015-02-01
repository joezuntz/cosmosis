from . import plots
from . import statistics
from .postprocess_base import PostProcessor, postprocessor_registry

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


class EmceeProcessor(MetropolisHastingsProcessor):
	sampler="emcee"
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


class MultinestProcessor(PostProcessor):
	elements = [
		plots.MultinestPlots1D, 
		plots.MultinestPlots2D,
		statistics.MultinestStatistics,
		statistics.MultinestCovariance,
		statistics.Citations,
		]
	sampler="multinest"


class PMCPostProcessor(PostProcessor):
	sampler="pmc"
	elements = [
		plots.WeightedMetropolisPlots1D,
		plots.WeightedMetropolisPlots2D,
		statistics.WeightedMetropolisStatistics,
		#statistics.WeightedMetropolisHastingsCovariance,
		#statistics.DunkleyTest,
		statistics.Citations,		
	]

class SnakeProcessor(PostProcessor):
	sampler='snake'
	elements = [
		plots.GridPlots1D,
		plots.SnakePlots2D,
	]
