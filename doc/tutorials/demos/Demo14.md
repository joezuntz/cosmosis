# Demo 14: Kombine sampler analysis of cluster gas fractions with advanced postprocessing options

# This demo is only present in the development version of cosmosis#


The ratio of gas to total mass in clusters in dynamically relaxed galaxy clusters
is sensitive to the expansion history of the universe and the background cosmological
baryon fraction.

The fraction is hard to measure, but a combination of lensing and X-ray data can give a handle on it. A series of papers by Mantz et al describe such a measurement.  The likelihood shown here include a number of nuisance parameters, some sampled and some fixed.


The Kombine sampler by Farr and Farr is a recently-released ensemble sampler which is particularly good at multi-modal sampling, though that particular feature will not be necessary here.  Kombine builds up a picture of the likelihood by dividing existing samples into modes (with the k-means algorithm) and building a proposal using kernel density estimation.


## Running

Before you run this demo you may need to seperately install the Kombine sampler.   There is currently a bug in the master version of the code, so you need to install a specific version:

pip install git+git://github.com/bfarr/kombine@e029360fa76bde5492cec6683eb90837dc8cbc14

or if this gives you a permissions error:

pip install --user git+git://github.com/bfarr/kombine@e029360fa76bde5492cec6683eb90837dc8cbc14


Then you can run this demo with the command:

    cosmosis demos/demo14.ini

This sampler can also be run in parallel, so to speed things up you can also do:

    mpirun -n 4 cosmosis --mpi demos/demo14.ini

With MPI this sampler will take about 90 seconds.

You Generate plots using:

    postprocess demos/demo14.ini  -o plots -p demo14 --thin 2 --blind-add  --more-latex demos/latex14.ini --tweaks demos/tweaks14.py

this will take a few minutes - longer than running the sampler in the first place. We will discuss in a moment the additional options used.  As well as the usual output you should see a series of lines like this at the start:

    Blinding additive value for cosmological_parameters--omega_b ~ 0.100000
    Blinding additive value for cosmological_parameters--hubble ~ 100.000000


A plot like the below will be made.  Note that this is not the real constraints from this data set!  The blinding option has been used to conceal the actual constraints; this will be constrained below.


![demo14_2D_cosmological_parameters--omega_b_cosmological_parameters--hubble.png](https://bitbucket.org/repo/KdA86K/images/2601990017-demo14_2D_cosmological_parameters--omega_b_cosmological_parameters--hubble.png)

## Understanding

### Pipeline

The pipeline in this case makes use of two new modules:

    [pipeline]
    modules = consistency bbn_consistency camb fgas
    likelihoods = fgas
    values = demos/values14.ini
    extra_output = cosmological_parameters/yhe

The BBN consistency module sets the helium mass fraction (YHe) from the mean baryon density (ombh2) and number of neutrinos (delta_neff), based on a table interpolation from Big Bang Nucleosynthesis calculations.  We also save the helium fraction that it calculates at the end of the pipeline.

The fgas module uses the likelihood described in detail in http://arxiv.org/abs/1402.6212 and derived from the expansion history and baryon fraction.  It has a large number of parameters which are described on the modules page and in detail in the paper.

### Postprocessing

We use a number of post-processing features for this demo.

**--thin 3**  

MCMC-style chains, including kombine, can take two options, "thin" and "burn" which shorten the chain by throwing away samples.  This options keeps only every 3rd sample in the chain.  We could also ignore some fraction at the start of the chain with the --burn option.

**--blind-add**

Experimenter bias is a significant problem in many areas of science, including cosmology.  It is the tendency to continue to experiment with changing different parts of your analysis only until you obtain the result you are expecting, often on a subconscious level.  In areas of cosmology with a strongly expected answer (such as agreement with a previous experiment or with w=-1, say) it is a very good idea to keep your final results blind until your analysis procedure is fixed.  

The --blind-add and --blind-mul add or multiply the chain values by secret numbers.  These values are derived from the name of the parameter, so you can compare the results of different chains with blinding switched on in both and get consistent results.  There are some edge cases where the scale of the blinding factor (which is based the mean of the parameter) might change if the parameter is close to a power of ten, so in the event of strange results pay close attention to the lines printed out above showing the scales of the added/multiplied constants.

**--more-latex demos/latex14.ini**

There are a number of additional parameters in this chain which don't have latex codes built into cosmosis.  The additional latex file describes extra latex strings for use as axis labels:

    [fgas_parameters]
    U_gas_0   = \Upsilon_0
    U_gas_1   =   \Upsilon_1
    fgas_scatter =   \sigma_f
    fgas_rslope  =   \eta
    cl_cal       =   K_0
    cl_calev     =   K_1



**--tweaks demos/tweaks14.py**

The matplotlib axis tick mark labeller is not very good, and so we need to tweak the labelling.  This tweak file, like the ones in earlier demos, modifies an existing set of plots - it removes every other tick label.

This is done by making a python class and telling it which files to work on with a class variable:

    class ThinLabels(Tweaks):
        filename=["2D_cosmological_parameters--omega_b_cosmological_parameters--hubble",
            "2D_cosmological_parameters--yhe_cosmological_parameters--omega_b",
            "2D_cosmological_parameters--yhe_cosmological_parameters--hubble"]


and then writing the actual behaviour in a method called "run":

        def run(self):
        	#Get the current axis
            ax=pylab.gca()
            #Get the numbers on the tick positions as a list
            labels = ax.get_xticks().tolist()
            #Change every second list entry to an empty string
            for i in xrange(0,len(labels),2):
                labels[i] = ''
            #Change the labels to a new list
            ax.set_xticklabels(labels)
