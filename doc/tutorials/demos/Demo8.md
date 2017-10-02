# Demo 8:  Comparing CAMB with Eisenstein & Hu; growth factors; plot tweaking  #

In this demo we will show one example of comparing predictions of different methods with cosmosis.  We will compare the Eisenstein & Hu approximation to the matter power spectrum (which is faster but does not include baryon wiggles) with the complete calculation with CAMB.

We will also show how to tweak plots a little to adjust details to make them publication-ready.


## Running ##

Run the demo with

```
#!bash

cosmosis demos/demo8.ini
```

The outputs should be similar to [demo 1](Demo1), with a little extra because we are now also computing the growth rate and applying the Eisenstein & Hu approximation.

This time we will add a new argument to post-processing:

```
#!bash

postprocess demos/demo8.ini -o plots -p demo8 --tweaks demos/tweaks8.py

```

You will have the same plots as in Demo 1, with one extra, and one with important modifications:

```
#!text

plots/demo8_growth.png        # New! 
plots/demo8_matter_power.png  # Modified!

```

![demo8_growth.png](https://bitbucket.org/repo/KdA86K/images/2682451871-demo8_growth.png)

![demo8_matter_power.png](https://bitbucket.org/repo/KdA86K/images/1170400040-demo8_matter_power.png)


## Understanding ##

There are now two "sub-pipelines" in the same pipeline, calculating slightly different things, that we can compare.  The list of modules now reads:

```
#!ini
[pipeline]
modules = consistency  growth  ehu  camb
```

The `growth` module calculates the linear-theory growth factor D(z) and the growth rate f(z).  See its entry on the [modules page](default_modules) for more information on definitions.

The `ehu` module uses Eiichiro Komatsu's Cosmology Routine Library implementation of the Eisenstein & Hu approximation to calculate a no-wiggles matter power spectrum at z=0.  This is then projected to higher redshifts using the growth factor.  The result is put in a section called matter_power_no_bao.

Finally, as before, `camb` performs the Boltzmann integration to calculate the more complete linear  matter power spectrum.

The post-processing script automatically makes a plot showing both of methods of calculating P(k), and we can see that they match really well, as they should.

### Tweaks ###

This example also introduces a method we can use to modify the plots that postprocess generates after they are complete.  This is mainly useful for cosmetic modifications, and in this example we use it to add a title to the plot ("CAMB vs Eisenstein & Hu") and to change the spacing slightly.  We refer to these small modifications as "Tweaks".

We tell postprocess where to find a python file containing our tweaks using the `--tweaks` argument.  When we do that postprocess reads that file and looks for any tweaks inside.  Here is the first tweak we make, to add the title:

```
#!python
class AddTitle(Tweaks):
    filename="matter_power"
    def run(self):
        pylab.title("CAMB vs Eisenstein & Hu")
```

The syntax "class AddTitle(Tweaks)" means that we are defining a new class (a blueprint for a bundle of functions and data), and that its parent, from which it inherits behaviour and variables, is "Tweaks".  Any classes defined in this file which inherit from Tweaks will be automatically run on the plots.

The rest of the syntax tells postprocess which file the tweak applies to, and then the "run" method tells it what tweak to make.  In this case the tweak is just one simple line, but it can more.


Here is the second tweak we make:

```
#!python

class MoreSpace(Tweaks):
    filename="all plots"
    def run(self):
        pylab.subplots_adjust(bottom=0.15, left=0.15)
```

This tweak uses "all plots" for the filename, to indicate, unsurprisingly, that this applies to all the different plots that we make.  We could also have used a list of files to do a particular set.   The tweak adds some more space at the edge of the plot for the labels, since sometimes they can fall off the edge.

Tweaks are mainly for small changes to plots.  To make an entirely new plot you can follow the example in [demo 9](Demo9).
