"""
By default post-process does not make any
scatter plots, since there are so many possible
combinations you might want.

This shows how to add any color scatter plots that you want.

The same mechanism can be used to make any customized plots - 
an example of that is below too.

Instructions for busy people
----------------------------

Copy and paste the last seven lines of this file for
each scatter plot that you want.  Change the columns 
that you want for the x, y, and color, and the filename
for each one.  Don't mess up the indentation.


Explanation for non-python people
---------------------------------

All the types of plot that cosmosis makes are represented
by python "classes".  A class is a collection of functions
connected together with some space to store data - sort of 
like a mix between a variable with a Fortran derived type
and a Fortran "module".

Classes have a family tree, and inherit behaviour from their
parents.  This can be used when you want to have a range of
different possible behaviours without copying and pasing code.

We already supply with a class which makes scatter
plots with a third variable as the color.  You just need to
create children of this class which know specificially which
variables to use for x, y, and the color.  This demo contains
an example using Delta-M, h, and Omega_m as the variables.

We also make another plot to make our own custom multinest plot


Explanation for python people
-----------------------------

Subclasses here should inherit from a subclass of 
cosmosis.postprocessing.elements.Element.  Any such subclasses
are detected and instantiated as a postprocessor step.

The main behaviour is defined in the run method,
which in this case is inherited from ColorScatterPlotBase.

plots.MultinestColorScatterPlot has this class hierarchy:


                   Loadable
                      |
             PostProcessorElement
                /             \ 
               /              Plots
              /                 \  
MulitnestPostProcessorElement  ColorScatterPlotBase
               \                    /
              MultinestColorScatterPlot
                        | 
                   ColorScatter


The ColorScatterPlotBase class has class variables
that need over-riding here to define which variables
to plot and what the filename should be.

The second example, NestPlot, shows a completely custom
plot.
"""

from cosmosis.postprocessing import plots
from cosmosis.postprocessing import lazy_pylab as pylab
import numpy as np


class ColorScatter(plots.MultinestColorScatterPlot):
    #Column names to use:
    x_column = "supernova_params--deltam"
    y_column = "cosmological_parameters--h0"
    color_column = "cosmological_parameters--omega_m"
    #File name to create
    scatter_filename = "scatter_dm_h0_omm"



class NestPlot(plots.Plots, plots.MultinestPostProcessorElement):
    def run(self):
        like = self.reduced_col("like")
        weight = self.weight_col()

        filename = self.filename("nest_weight")
        figure = self.figure(filename)

        pylab.plot(like, 1e4*weight)
        pylab.xlim(-353, -341.5)
        pylab.xlabel("Likelihood")
        pylab.ylabel(r"Weight$\times  10^4$")
        #Add some helpful text
        pylab.title("Nested sampling progress")
        pylab.text(-351.6, 3, "Increasing likelihood,\n volume still large", size="small")
        pylab.text(-344.2, 3.5, "Volume\ndecreasing", size='small')
        pylab.text(-343.6, 0.5, "Final $n_\mathrm{live}$\npoints", size='small')

        return [filename]