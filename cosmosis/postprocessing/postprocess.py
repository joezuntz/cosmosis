from . import plots
from . import statistics
from .postprocess_base import PostProcessor, postprocessor_registry

def postprocessor_for_sampler(sampler):
	return postprocessor_registry.get(sampler)

class MetropolisHastingsProcessor(PostProcessor):
	plotClass=plots.MetropolisHastingsPlots
	statsClass=statistics.MetropolisHastingsStatistics


class EmceeProcessor(MetropolisHastingsProcessor):
	sampler="emcee"
	pass


class PymcProcessor(MetropolisHastingsProcessor):
	sampler="pymc"
	pass


class GridProcessor(PostProcessor):
	sampler="grid"
	plotClass=plots.GridPlots
	statsClass=statistics.GridStatistics

class TestProcessor(PostProcessor):
	sampler="test"
	plotClass=plots.TestPlots
	statsClass=statistics.TestStatistics
	cosmosis_standard_output=False


