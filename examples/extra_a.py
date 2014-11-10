"""
See demo 9 for more info on this.
"""

from cosmosis.postprocessing import plots

class ColorScatter(plots.MCMCColorScatterPlot):
    x_column = "cosmological_parameters--omega_m"
    y_column = "cosmological_parameters--h0"
    color_column = "cosmological_parameters--n_s"
    scatter_filename = "scatter_omm_h0_ns"
