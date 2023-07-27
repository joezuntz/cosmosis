from .postprocess import postprocessor_for_sampler
from .inputs import read_input
from .plots import Tweaks
from ..utils import mkdir



def run_cosmosis_postprocess(inputs, **kwargs):
    """
    Run the postprocessing steps on the given input file(s).

    Parameters
    ----------
    inputs : list of str or output objects
        The input file(s) to run postprocessing on.


    """
    #Make the directory for the outputs to go in.
    outdir = kwargs.get("outdir", ".")
    legend = kwargs.get("legend")
    run_max_post = kwargs.get("run_max_post", False)
    text = kwargs.get("text", False)
    weights = kwargs.get("weights", False)
    extra = kwargs.get("extra")
    tweaks = kwargs.get("tweaks")

    mkdir(outdir)
    outputs = {}
    
    #Deal with legends, if any
    if legend:
        labels = legend.split("|")
        if len(labels)!=len(inputs):
            raise ValueError("You specified {} legend names but {} files to plot".format(len(labels), len(inputs)))
    elif isinstance(inputs[0], str):
        labels = inputs
    else:
        labels = [f"chain_{i}" for i in range(len(inputs))]

    if len(inputs)>1 and run_max_post:
        raise ValueError("Can only use the --run-max-post argument with a single parameter file for now")

    processors = []
    processor = None

    for i, ini_filename in enumerate(inputs):
        if "astropy" in str(type(ini_filename)):
            ini_filename.meta.setdefault("chain_name", labels[i])

        sampler, ini = read_input(ini_filename, text, weights)
        processor_class = postprocessor_for_sampler(sampler.split()[-1])

        #We do not know how to postprocess everything.
        if processor_class is None:
            print("I do not know how to postprocess output from the %s sampler"%sampler)
            sampler = None
            continue

        #Create and run the postprocessor

        processor = processor_class(ini, labels[i], i, **kwargs)

        #Inherit any plots from the previous postprocessor
        #so we can make plots with multiple datasets on
        processor.outputs.update(outputs)

        #We can load extra plots to make from a python
        #script here
        if extra:
            processor.load_extra_steps(extra)

        #Optionally add a step in which we re-run the best-fit
        if run_max_post:
            processor.add_rerun_bestfit_step(run_max_post)

        #Run the postprocessor and make the outputs for this chain
        processor.run()

        #Save the outputs ready for the next post-processor in case
        #they want to add to it (e.g. two constriants on the same axes)
        outputs.update(processor.outputs)

    # If there was no successul postprocessor then we are done
    if processor is None:
        return

    # Finalize all the elements - this adds legends to any plots
    # that need them. This final processor knows about all the
    # outputs that we made.
    processor.finalize()

    if sampler is None:
        return

    #Run any tweaks that the user specified
    if tweaks:
        tweaks = Tweaks.instances_from_file(tweaks)
        for tweak in tweaks:
            processor.apply_tweaks(tweak)

    return processor
