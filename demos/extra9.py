"""
By default post-process does not make any
scatter plots, since there are so many possible
combinations you might want.

This shows how to add any color scatter plots that you want.

The same mechanism can be used to make any customized plots - 
an example of that is below too.

Instructions for busy people
----------------------------

Copy and paste lines 87-93 of this file for
each scatter plot that you want.  Change the columns 
that you want for the x, y, and color, and the filename
for each one.  Don't mess up the indentation.

If you use this for an MCMC instead of multinest then change
MultinestColorScatterPlot -> MCMCColorScatterPlot.

The other plot is a mulitnest example of creating a completely
new kind of plot.

Explanation for non-python people
---------------------------------

All the types of plot that cosmosis makes are represented
by python "classes".  A class is a collection of functions
connected together with some space to store data - sort of 
a mix between a variable with a Fortran derived type and a 
Fortran "module".

Classes have a family tree, and inherit behaviour from their
parents.  This can be used when you want to have a range of
different possible behaviours without copying and pasing code.

We already supply with a class which makes scatter
plots with a third variable as the color.  You just need to
create children of this class which know specificially which
variables to use for x, y, and the color.  This demo contains
an example using Delta-M, h, and Omega_m as the variables.

We also make another class to make our own custom plot
illustrating the concepts of multinest.


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
plot where the run method is over-ridden.
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
        #Get the columns we need, in reduced form 
        #since this is not MCMC
        weight = self.weight_col()
        weight = weight/weight.max()
        #Sort the final 500 samples, since they 
        #are otherwise un-ordered
        weight[-500:] = np.sort(weight[-500:])

        #x-axis is just the iteration number
        x = np.arange(weight.size)

        #Choose a filename and make a figure for your plot.
        figure, filename = self.figure("nest_weight")

        #Do the plotting, and set the limits and labels
        pylab.plot(x, weight/weight.max())
        #pylab.xlim(-353, -341.5)
        pylab.xlim(0,8000)
        pylab.xlabel("Iteration number $i$")
        pylab.ylabel(r"$p_i$")

        #Add some helpful text, on the title and the plot itself
        pylab.title("Normalised posterior weight $p_i$", size='medium')
        pylab.text(500, 0.4, "Increasing likelihood,\n volume still large", size="small")
        pylab.text(5500,0.65, "Volume\ndecreasing", size='small')
        pylab.text(6500, 0.12, "Final $n_\mathrm{live}$\npoints", size='small')

        #Return the list of filenames we made
        return [filename]