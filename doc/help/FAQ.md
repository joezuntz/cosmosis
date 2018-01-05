# FAQ


## Why does the bootstrap installer download so many things?

The auto-installation script downloads a complete stack down to the compiler.  We know this is a bit annoying and weird, and you can absolutely try and manage the dependency installation yourself, but we've found that with such a wide variety of codes collected together getting a consistent set of requirements can be a huge pain.  It's very easy to end up with, e.g. one lapack in scipy and another from source that are incompatible.

So our default install downloads a few hundred MB of compilers and other infrastructure to avoid this.

---------------------------------------


## When I run under MPI my output file comes out wrong with some lines too short and some repeated

To run cosmosis under MPI you also need to pass it the --mpi flag, not just use mpirun, e.g.:

    mpirun -n 4 cosmosis --mpi params.ini

---------------------------------------


## How do I create a new module?

https://bitbucket.org/joezuntz/cosmosis/wiki/modules


---------------------------------------

## Where can I find a list of modules?

https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules

---------------------------------------

## How does my code call CosmoSIS?

It doesn't.  CosmoSIS calls your code.  Any physics or likelihood calculation becomes a module, an pipeline element that can plug into cosmosis, and from there into other modules.

---------------------------------------

## CAMB says it's failing to find the parameter w but it's definitely there

You may need to change:

    w = -1

to:

    w = -1.0

------------------------------------------------

## When trying to run "xcode-select --install" on a mac I get the error "Can't download the software because of a network problem"

Please see the instructions on this page to install manually:

http://discuss.binaryage.com/t/aesepsis-1-4-1-issue-with-update-installing-xcode-command-line-tools/2013

---------------------------------------


## I get the message "Abort trap 6" when I try to post-process on a Mac.

Try this, e.g. for demo one:
DYLD_LIBRARY_PATH="" postprocess demos/demo1.ini

---------------------------------------


## _matplotlib_ still is not working for me. Can I make plots some other way?

If you encounter any trouble concerning _matplotlib_, please note that
_matplotlib_ is only used for producing plots, and is not required to
run _CosmoSIS_ itself. If you have, or can get, an installation of the
_R_ statistical environment, you can use it for generating plots as
well. _R_ can be obtained by following the instructions at the [R
project home page](http://www.r-project.org). Some additional _R_
packages are needed. Please _source_ the script
_cosmosis/plotting/install-r-packages_ to install them:
```
#!bash
$> source cosmosis/plotting/install-r-packages
```

---------------------------------------


## I used git to update cosmosis and now things don't work

There are two repositories, cosmosis and the subdirectory cosmosis-standard-library.  Make sure you update both and then do make clean and make afterwards.

---------------------------------------


## I'm worried that CosmoSIS will change and then my code won't work any more

In general we do aim to develop in a backward compatible way. Consider writing your own [regression tests for CosmoSIS](https://bitbucket.org/joezuntz/cosmosis/wiki/Regression%20tests%20for%20CosmoSIS) i.e. tests that you or the cosmosis core developers can run that checks if everything is still working the way you want. We can't promise that we won't break your regression test, but at least we will be able to try to keep things working for you and if not then we can let you know.

---------------------------------------

## I get an error with _thread in the "six" module on OSX

If you get an error like this:

    /Library/Python/2.7/site-packages/dateutil/rrule.py in <module>()
        14
        15 from six import advance_iterator, integer_types
    ---> 16 from six.moves import _thread
        17
        18 __all__ = ["rrule", "rruleset", "rrulestr",
    
    ImportError: cannot import name _thread

Then this is a general problem with the python-dateutil module. Fix it like this:

    sudo pip uninstall python-dateutil
    sudo pip install python-dateutil==2.2


---------------------------------------

## I'm getting a GSL interpolation error when running cosmic shear analyses

Your n(z) needs to go all the way down to z=0, and no higher than zmax that you gave CAMB.

---------------------------------------


## The axis labels look wrong - they have weird subscripts in the middle of words

You're using a parameter that cosmosis doesn't know the latex name for.    

For example, say you used a new parameter "m_max" in a section called "galaxies".

Make a new file with the parameters names in called a new file something like ```my-latex.ini```:

```
#!ini
[galaxies]
m_max = M_\mathrm{max}
```

Then run the postprocess command with the flag ```--more-latex=my-latex.ini```

---------------------------------------

## Can I use the CosmoSIS bootstrap with my forked repo?

Yes, you can do this by installing the normal cosmosis using the bootstrap code, but then change it to point instead to the new repository. E.g. use the bootstrap to install everything into a new directory called "my_cosmosis"

```
./cosmosis-bootstrap my_cosmosis
cd my_cosmosis
source config/setup-cosmosis
git remote set-url origin https://bitbucket.org/accountname/myforked_cosmosis
git pull
cd cosmosis-standard-library
git remote set-url origin https://bitbucket.org/accountname/myforked_cosmosis-standard-library
git pull
cd ..
make
```
---------------------------------------

## How can I customize my contour plot colors and line styles?

Use a "tweak", a set of commands which are run after the plotting is complete to customize one or more plots.  [Demo 8](Demo8) has an explanation of tweaks in general.  Here's a specific example for customizing a plot with two contours on.

Put this text in a file ``contour_tweaks.py`` and then run your postprocess command with the flag ``--extra contour_tweaks.py``:

    from cosmosis.postprocessing.plots import Tweaks
    import pylab

    class ModifyContours(Tweaks):
        #This could also be a list of files.  Just put the base part in here,
        #not the directory, prefix, or suffix.
        filename="2D_cosmological_parameters--omega_m_cosmological_parameters--h0"

        def run(self):
            ax = pylab.gca()
            #if you want to try this interactively you can add this:
            # from IPython import embed; embed() #press ctrl-D when finished playing
            #you don't unfortunately see the results until it finishes, I think, though you could
            # try adding a pylab.show?

            #each set has two contours in it, inner and outer 
            contour_set_1 = ax.collections[:2]
            contour_set_2 = ax.collections[2:4]

            #set the properties of the contour face and line
            for f in contour_set_1:
                f.set(linestyle=':', linewidth=3, facecolor='none', edgecolor='k', alpha=1.0)

            #you could do the same for contour set 2, etc.,  here.
            #just remember that 2 will always be drawn on top of 1; you may
            #need to choose the order of chain files on the command line accordingly

---------------------------------------

## How can I save a parameter that I marginalize over analytically, or generate in some other way

If you have an extra parameter that is derived from your chain, for example one marginalized analytically or derived from other parameters, you can save it in the output chains along with the sampled parameters

In the pipeline section of your parameter ini file, set:

    [pipeline]
    extra_output = section_name/param_name   section_name2/param_name2

This would save a parameter ``param_name`` that you write to the data block in the ``section_name`` section.


---------------------------------------

## How can I check convergence of the emcee sampler

One quick check for convergence of emcee is to plot each parameter the chain as points.  If it has converged then the various chains should all gradually diffuse out from the starting position and then all come to a similar deviation from the mean.  If the chains all still have a gradual drift across the chain, for example if they are all still moving outwards by the end of the chain, then that indicates non-convergence.

If you'd like you can also use the acor module to test convergence as in emcee.  Install acor using ``pip install acor`` and then you can use ``acor.acor(data)`` from python - you will need to reshape the chain to make it ``nwalker * nsample`` (or possibly the other way around!).


---------------------------------------

## How can I improve emcee convergence

There is an alpha parameter for emcee, but we do not currently expose it because it does not usually help convergence.  Instead the best way is usually to improve burn-in.  If you can guess a good distribution of starting points for the chain (one per walker; for example, from an earlier chain, or guessing) then you can set ``start_points`` to the name of a file with columns being the parameters and rows being the different starting points.


---------------------------------------

## What parameters does the cosmosis data block include

The data block does not include a fixed set of parameters. Instead it can contain anything you want to put into it. At the start of a pipeline (i.e. at the start of a single likelihood evaluation) it will contain just the parameters put into it from the values file; after each module is run more things will be added.

---------------------------------------

## During the bootstrap, installation of matplotlib has failed

If you are on a Linux machine, the main cause of this failure is lack of some system software that is required to build matplotlib. The package most commonly missing is the freetype development headers and libraries. The solution is to have someone with system management privileges install the appropriate package. On a RHEL-type system, the installation command is:

    yum install freetype-devel.x86_64

