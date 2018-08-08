# Import lots of bits from across cosmosis.
# TODO: Tidy this up
from cosmosis.samplers.fisher.fisher_sampler import FisherSampler
from cosmosis.runtime.pipeline import LikelihoodPipeline
from cosmosis.runtime.config import Inifile
from cosmosis.output.in_memory_output import InMemoryOutput
import numpy as np

# Create the configuration based on a file.
ini = Inifile("./demos/demo17.ini")

# Build the likelihood pipeline based on the config
pipeline = LikelihoodPipeline(ini)

# Fix a few parameters, just to make things faster
pipeline.set_fixed("shear_calibration_parameters", "m1", 0.0)
pipeline.set_fixed("shear_calibration_parameters", "m2", 0.0)
pipeline.set_fixed("shear_calibration_parameters", "m3", 0.0)

# tell cosmosis to keep the output in memory instead
# of writing to a text file
output = InMemoryOutput()

# Create the sampler.  In this case for speed
# we will just use a Fisher Matrix
sampler = FisherSampler(ini, pipeline, output)
sampler.config()

# Run the sampler.
# "convergence" doesn't necessarily mean the same
# thing for all the samplers - for fisher it just
# means until completion
while not sampler.is_converged():
    sampler.execute()

# The output object now contains the data
# that would have been output to file
fisher_matrix = np.array(output.rows)

# You could now do stuff with this matrix if you wanted.
print(fisher_matrix)
