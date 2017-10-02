# Demo 9:  Bayesian evidence with Multinest; scatter and custom plots #

## Running ##

Bayesian evidence is a method for model selection - it tells us how likely an entire cosmological model is, rather than just a particular choice of parameters in that model.

In this demo we will use the sampler Multinest to calculate the Bayesian evidence.  As a by-product we can also construct all the usual parameter constraint plots, or get a set of posterior samples.  We'll use the same likelihood and pipeline as [demo 5](Demo5), the JLA SDSS supernova sample.

We'll also show you how to add additional custom plots as well as the standard ones that are produced by postprocess.  That way you can use the cosmosis system that understands how to load and interpret the output files with your own custom plotting.


Run the multinest sampler like this:


```
#!bash

cosmosis demos/demo9.ini
```

This should be a little faster than the emcee example - multinest is clever!

After a preamble about loading the supernova data files, you should see something like this:

```
#!text

 *****************************************************
 MultiNest v3.7
 Copyright Farhan Feroz & Mike Hobson
 Release June 2014

 no. of live points =  500
 dimensionality =    5
 *****************************************************
 Starting MultiNest
 generating live points
 live points generated, starting sampling
Acceptance Rate:                        0.994575
Replacements:                                550
Total Samples:                               553
Nested Sampling ln(Z):            -130739.431350
Importance Nested Sampling ln(Z):    -376.884711 +/-  0.999095

```

The last part will repeat until convergence, which should take a few minutes (about 15,000 samples are needed).  At the end you'll get a print out of some final statistics, which should look something like this:

```
#!text

Saving 8231 samples
 ln(ev)=  -356.16416856202284      +/-  0.14586312956518516     
 Total Likelihood Evaluations:        15063
 Sampling finished. Exiting MultiNest
```

The output file demo9.txt will contain this: 

```
#!bash

#cosmological_parameters--omega_m   cosmological_parameters--h0 supernova_params--deltam    supernova_params--alpha supernova_params--beta  like    weight
#sampler=multinest
# ... [SNIP extra info] ...
0.353661501408  0.885217994452  -9.97488260269  0.139410300255  2.40945339203   -255356.626132  0.0
0.240411996841  0.649673074484  9.91520762444   0.146602618694  2.34271526337   -244594.938757  0.0
0.351006817818  0.988167345524  -9.41894292831  0.140028219223  3.14296913147   -224902.169209  0.0
0.338572371006  0.633784323931  9.13187503815   0.14318883419   2.01251602173   -218580.219995  0.0
0.384171569347  0.941171020269  -8.48028302193  0.120274708271  2.03499937057   -206783.320879  0.0
# ...

```

Don't worry about the zero weight!  That's just for the early samples.  Let's generate some plots and stats, including some extra custom ones:

```
#!bash
postprocess -o plots -p demo9 demos/demo9.ini --extra demos/extra9.py

```

You'll get some summary stats, this time including the evidence value:

```
#!text

Marginalized mean, std-dev:
    cosmological_parameters--omega_m = 0.295753 ± 0.0336598
    cosmological_parameters--h0 = 0.738363 ± 0.0241566
    supernova_params--deltam = 0.0410873 ± 0.074628
    supernova_params--alpha = 0.136857 ± 0.00635137
    supernova_params--beta = 3.10989 ± 0.081146
    like = -345.526 ± 1.51149
    weight = 0.000348284 ± 0.000128698

Marginalized median, std-dev:
    cosmological_parameters--omega_m = 0.295503 ± 0.0336598
    cosmological_parameters--h0 = 0.738528 ± 0.0241566
    supernova_params--deltam = 0.0444039 ± 0.074628
    supernova_params--alpha = 0.136782 ± 0.00635137
    supernova_params--beta = 3.10814 ± 0.081146
    like = -345.222 ± 1.51149
    weight = 0.000390824 ± 0.000128698

Best likelihood:
    cosmological_parameters--omega_m = 0.299398
    cosmological_parameters--h0 = 0.740779
    supernova_params--deltam = 0.0486396
    supernova_params--alpha = 0.136235
    supernova_params--beta = 3.08996
    like = -343.073
    weight = 0.00589475

Bayesian evidence:
    log(Z) = -356.776 ± 0.145863

 ... 

```

The same plots as before will now be generated, in plots/demo9_*,  but with two custom ones that we specified in extra9.py.  One is a scatter plot using color to show a third variable; you can easily add more such plots for MCMC or Multinest runs.  The other is more specific to multinest: an annotated variant of figure 2 from [Allison & Dunkley](http://adsabs.harvard.edu/abs/2014MNRAS.437.3918A):


![scatter_dm_h0_omm.png](https://bitbucket.org/repo/KdA86K/images/4144257523-scatter_dm_h0_omm.png)

![nest_weight.png](https://bitbucket.org/repo/KdA86K/images/3968326800-nest_weight.png)

## Understanding ##

### The sampling ###

The multinest code and algorithm are described in [Feroz, Hobson & Bridges](http://adsabs.harvard.edu/abs/2009MNRAS.398.1601F).  There's also a nice summary and comparison to emcee and Metropolis in [Allison & Dunkeley](http://adsabs.harvard.edu/abs/2014MNRAS.437.3918A).

Here's how we tell cosmosis to use multinest:

```
[runtime]
sampler = multinest

[multinest]
max_iterations=50000
live_points=500
multinest_outfile_root=

```

In this case the sampler did not need the full maximum 50,000 iterations to reach a good level of accuracy - it suceeded long before then.  live_points is the size of the multinest set of points; increasing it will get you more samples from the posterior at the end at the cost of more time.  A few hundred is typical.  If we had set the outfile_root parameter multinest would have given us a large number of different output files of its own - if you're a mulitnest guru you can use those directly.

The rest of our ini file is the same as demo 5, since we have not changed the pipeline.


### The plotting ###

The chain files produced by multinest are a little more complicated than with emcee - they are not equally weighted posterior samples but have an extra "weight" column attached.  The postprocess code knows how to handle these weights and uses them in the kernel density estimation process to produce the constraints.



#### Extra plots ####

The --extra flag told cosmosis to look in demos/extra9.py for a python script containing extra plots to make.  You can create any new postprocessing step - it could be a plot, some statistics, or something else completely new.

postprocess searches the file you give it for any new post-processing steps, which must be python classes that inherit from a base post-processing step clas.  There are more details in extra9.py comments.

The first plot we make with this file is the scatter plot.  The cosmosis plotting tools already contain the code for this plot; all we have to do is tell it which columns to use for plotting. We define it like this:

```
#!python

class ColorScatter(plots.MultinestColorScatterPlot):
    x_column = "supernova_params--deltam"
    y_column = "cosmological_parameters--h0"
    color_column = "cosmological_parameters--omega_m"
    scatter_filename = "scatter_dm_h0_omm"

```

The first three lines show the columns to use for the x, y, and color data respectively.  They also provide the base filename to save to.

If you want a number of color scatter plots you could create multiple versions of this: ColorScatter1, ColorScatter2, etc.


The example is specific to multinest.  It shows how to create a completely new plot by writing the "run" method:

```
#!python

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
        filename = self.filename("nest_weight")
        figure = self.figure(filename)

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
```

In the run method you put ordinary pylab commands to draw whatever you want, except that the figure (matplotlib plotting space) should be generated using the commands shown below - we ask our parent class for a full filename to use (including the prefix and directory) and for the figure to generate.

To get the data for the plot we use method inherited from cosmosis - the reduced_col method gets a particular named column (and removes the burn-in in MCMC or preliminary samples in multinest).

Finally we return a list of any files we have created.