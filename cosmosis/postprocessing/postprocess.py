from . import plots
from . import statistics
from .postprocess_base import PostProcessor, postprocessor_registry

def postprocessor_for_sampler(sampler):
	return postprocessor_registry.get(sampler)

class MetropolisHastingsProcessor(PostProcessor):
	elements=[
		plots.MetropolisHastingsPlots,
		plots.MetropolisHastings2DPlots,
		statistics.MetropolisHastingsStatistics,
		]


class EmceeProcessor(MetropolisHastingsProcessor):
	sampler="emcee"
	pass


class PymcProcessor(MetropolisHastingsProcessor):
	sampler="pymc"
	pass

class GridProcessor(PostProcessor):
	elements=[plots.GridPlots1D, plots.GridPlots2D]
	sampler="grid"

class TestProcessor(PostProcessor):
	elements = [plots.TestPlots]
	sampler="test"
	cosmosis_standard_output=False


