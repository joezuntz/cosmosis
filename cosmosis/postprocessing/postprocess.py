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
		]


class EmceeProcessor(MetropolisHastingsProcessor):
	sampler="emcee"
	pass


class PymcProcessor(MetropolisHastingsProcessor):
	sampler="pymc"
	pass

class GridProcessor(PostProcessor):
	elements=[
		plots.GridPlots1D,
		plots.GridPlots2D,
		statistics.GridStatistics
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
		]
	sampler="multinest"

class SnakeProcessor(PostProcessor):
	sampler='snake'
	elements = [
		plots.GridPlots1D,
		plots.GridPlots2D,
	]