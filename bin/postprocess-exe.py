#!/usr/bin/env python
from __future__ import print_function
from cosmosis.postprocessing.postprocess import postprocessor_for_sampler
from cosmosis.postprocessing.inputs import read_input
from cosmosis.postprocessing.plots import Tweaks
from cosmosis.runtime.utils import mkdir
import sys
import argparse
import os


parser = argparse.ArgumentParser(description="Post-process cosmosis output")
parser.add_argument("inifile", nargs="+")
mcmc=parser.add_argument_group(title="MCMC", description="Options for MCMC-type samplers")
mcmc.add_argument("--burn", default=0.0, type=float, help="Fraction or number of samples to burn at the start")
mcmc.add_argument("--thin", default=1, type=int, help="Keep every n'th sampler in MCMC")
mcmc.add_argument("--weights", action='store_true', help="Look for a weight column in a generic MCMC file")

general=parser.add_argument_group(title="General", description="General options for controlling postprocessing")
general.add_argument("-o","--outdir", default=".", help="Output directory for all generated files")
general.add_argument("-p","--prefix", default="", help="Prefix for all generated files")
general.add_argument("--more-latex", default="", help="Load an additional latex file to the default")
general.add_argument("--no-latex", action='store_true', help="Do not use latex-style labels, just use the text")
general.add_argument("--blind-add", action='store_true', help="Blind results by adding adding a secret value to each parameter")
general.add_argument("--blind-mul", action='store_true', help="Blind results by scaling by a secret value for each parameter")
general.add_argument("--pdb", action='store_true', help="Run the debugger if any of the postprocessing stages fail")

inputs=parser.add_argument_group(title="Inputs", description="Options controlling the inputs to this script")
inputs.add_argument("--text", action='store_true', help="Tell postprocess that its argument is a text file, regardless of its suffix")
inputs.add_argument("--derive", default="", help="Read a python script with functions in that derive new columns from existing ones")

plots=parser.add_argument_group(title="Plotting", description="Plotting options")
plots.add_argument("--legend", help="Add a legend to the plot with the specified titles, separated by | (the pipe symbol)")
plots.add_argument("--legend-loc", default='best', help="The location of the legend: best, UR, UL, LL, LR, R, CL, CR, LC, UC, C (use quotes for the ones with two words.)")
plots.add_argument("--swap", action='store_true', help="Swap the ordering of the parameters in (x,y)")
plots.add_argument("--only", type=str, dest='prefix_only', help="Only make 2D plots where both parameter names start with this")
plots.add_argument("--either", type=str, dest='prefix_either', help="Only make 2D plots where one of the parameter names starts with this.")
plots.add_argument("--no-plots", action='store_true', help="Do not make any default plots")
plots.add_argument("--no-2d", action='store_true', help="Do not make any 2D plots")
plots.add_argument("--no-alpha", dest='alpha', action='store_false', help="No alpha effect - shaded contours will not be visible through other ones")
plots.add_argument("-f", "--file-type", default="png", help="Filename suffix for plots")
plots.add_argument("--no-smooth", dest='smooth', default=True, action='store_false', help="Do not smooth grid plot joint constraints")
plots.add_argument("--n-kde", default=100, type=int, help="Number of KDE smoothing points per dimension to use for MCMC 2D curves. Reduce to speed up, but can make plots look worse.")
plots.add_argument("--factor-kde", default=2.0, type=float, help="Smoothing factor for MCMC plots.  More makes plots look better but can smooth out too much.")
plots.add_argument("--no-fill", dest='fill', default=True, action='store_false', help="Do not fill in 2D constraint plots with color")
plots.add_argument("--extra", dest='extra', default="", help="Load extra post-processing steps from this file.")
plots.add_argument("--tweaks", dest='tweaks', default="", help="Load plot tweaks from this file.")
plots.add_argument("--no-image", dest='image', default=True, action='store_false', help="Do not plot the image in  2D grids; just show the contours")
plots.add_argument("--run-max-post", default="", help="Run the test sampler on maximum-posterior sample and save to the named directory.")

def main(args):
	#Read the command line arguments and load the
	#ini file that created the run
	args = parser.parse_args(args)

	for ini_filename in args.inifile:
		if not os.path.exists(ini_filename):
			raise ValueError("The file (or directory) {} does not exist.".format(ini_filename))

	#Make the directory for the outputs to go in.
	mkdir(args.outdir)
	outputs = {}

	#Deal with legends, if any
	if args.legend:
		labels = args.legend.split("|")
		if len(labels)!=len(args.inifile):
			raise ValueError("You specified {} legend names but {} files to plot".format(len(labels), len(args.inifile)))
	else:
		labels = args.inifile

	if len(args.inifile)>1 and args.run_max_post:
		raise ValueError("Can only use the --run-max-post argument with a single parameter file for now")

	for i,ini_filename in enumerate(args.inifile):
		sampler, ini = read_input(ini_filename, args.text, args.weights)
		processor_class = postprocessor_for_sampler(sampler.strip ())

		#We do not know how to postprocess everything.
		if processor_class is None:
			print("I do not know how to postprocess output from the %s sampler"%sampler)
			sampler = None
			continue

		#Create and run the postprocessor

		processor = processor_class(ini, labels[i], i, **vars(args))

		#Inherit any plots from the previous postprocessor
		#so we can make plots with multiple datasets on
		processor.outputs.update(outputs)

		#We can load extra plots to make from a python
		#script here
		if args.extra:
			processor.load_extra_steps(args.extra)

		#Optionally add a step in which we 
		if args.run_max_post:
			processor.add_rerun_bestfit_step(args.run_max_post)


		#Run the postprocessor and make the outputs for this chain
		processor.run()

		#Save the outputs ready for the next post-processor in case
		#they want to add to it (e.g. two constriants on the same axes)
		outputs = processor.outputs

	if sampler is None:
		return

	#Run any tweaks that the user specified
	if args.tweaks:
		tweaks = Tweaks.instances_from_file(args.tweaks)
		for tweak in tweaks:
			processor.apply_tweaks(tweak)

	#Save all the image files and close the text files
	processor.finalize()

if __name__=="__main__":
	main(sys.argv[1:])
