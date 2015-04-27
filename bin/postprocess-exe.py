#!/usr/bin/env python
from cosmosis.postprocessing.postprocess import postprocessor_for_sampler
from cosmosis.postprocessing.plots import Tweaks
from cosmosis.runtime.config import Inifile
from cosmosis.runtime.utils import mkdir
from cosmosis.output.text_output import TextColumnOutput
import sys
import argparse
import os


parser = argparse.ArgumentParser(description="Post-process cosmosis output")
parser.add_argument("inifile", nargs="+")
mcmc=parser.add_argument_group(title="MCMC", description="Options for MCMC-type samplers")
mcmc.add_argument("--burn", default=0.0, type=float, help="Fraction or number of samples to burn at the start")
mcmc.add_argument("--thin", default=1, type=int, help="Keep every n'th sampler in MCMC")

general=parser.add_argument_group(title="General", description="General options for controlling postprocessing")
general.add_argument("-o","--outdir", default=".", help="Output directory for all generated files")
general.add_argument("-p","--prefix", default="", help="Prefix for all generated files")
general.add_argument("--more-latex", default="", help="Load an additional latex file to the default")
general.add_argument("--no-latex", action='store_true', help="Do not use latex-style labels, just use the text")
general.add_argument("--blind-add", action='store_true', help="Blind results by adding adding a secret value to each parameter")
general.add_argument("--blind-mul", action='store_true', help="Blind results by scaling by a secret value for each parameter")

inputs=parser.add_argument_group(title="Inputs", description="Options controlling the inputs to this script")
inputs.add_argument("--text", action='store_true', help="Tell postprocess that its argument is a text file, regardless of its suffix")

plots=parser.add_argument_group(title="Plotting", description="Plotting options")
plots.add_argument("--swap", action='store_true', help="Swap the ordering of the parameters in (x,y)")
plots.add_argument("--no-plots", action='store_true', help="Do not make any default plots")
plots.add_argument("-f", "--file-type", default="png", help="Filename suffix for plots")
plots.add_argument("--no-smooth", dest='smooth', default=True, action='store_false', help="Do not smooth grid plot joint constraints")
plots.add_argument("--n-kde", default=100, type=int, help="Number of KDE smoothing points per dimension to use for MCMC 2D curves. Reduce to speed up, but can make plots look worse.")
plots.add_argument("--factor-kde", default=2.0, type=float, help="Smoothing factor for MCMC plots.  More makes plots look better but can smooth out too much.")
plots.add_argument("--no-fill", dest='fill', default=True, action='store_false', help="Do not fill in 2D constraint plots with color")
plots.add_argument("--extra", dest='extra', default="", help="Load extra post-processing steps from this file.")
plots.add_argument("--tweaks", dest='tweaks', default="", help="Load plot tweaks from this file.")
plots.add_argument("--no-image", dest='image', default=True, action='store_false', help="Do not plot the image in  2D grids; just show the contours")

def read_input(ini_filename, force_text):
	if ini_filename.endswith("txt") or force_text:
		output_info = TextColumnOutput.load_from_options({"filename":ini_filename})
		metadata=output_info[2][0]
		sampler = metadata.get("sampler")
		if sampler is None:
			print "This is not a cosmosis output file."
			print "So I will assume it in a generic MCMC file"
			sampler = "metropolis"
			ini = output_info
		else:
			ini = {"sampler":sampler, sampler:metadata, "data":output_info, "output":dict(format="text", filename=ini_filename)}
	elif os.path.isdir(ini_filename):
		ini = Inifile(None)
		ini.add_section("runtime")
		ini.add_section("test")
		sampler = "test"
		ini.set("runtime", "sampler", sampler)
		ini.set("test", "save_dir", ini_filename)
	else:
		#Determine the sampler and get the class
		#designed to postprocess the output of that sampler
		ini = Inifile(ini_filename)
		sampler = ini.get("runtime", "sampler")
	return sampler, ini


def main(args):
	#Read the command line arguments and load the
	#ini file that created the run
	args = parser.parse_args(args)
	ini_filename = args.inifile[0]

	sampler, ini = read_input(ini_filename, args.text)
	processor_class = postprocessor_for_sampler(sampler)

	#We do not know how to postprocess everything.
	if processor_class is None:
		print "I do not know how to postprocess output from the %s sampler"%sampler
		return

	#Make the directory for the outputs to go in.
	mkdir(args.outdir)

	#Create and run the postprocessor
	processor = processor_class(ini, **vars(args))
	if args.extra:
		processor.load_extra_steps(args.extra)
	try:
		processor.run()
	except:
		import traceback
		print "There was an error with one of the postprocessing steps:"
		traceback.print_exc()

	#Run it again for each subsequent file
	for ini_filename in args.inifile[1:]:
		sampler2, ini = read_input(ini_filename, args.text)
		if sampler2!=sampler:
			raise ValueError("Sorry - cannot currently process samples from two different samplers at once")
		processor.load(ini)
		try:
			processor.run()
		except:
			import traceback
			print "There was an error with one of the postprocessing steps:"
			traceback.print_exc()

	#Run any tweaks that the user specified
	if args.tweaks:
		tweaks = Tweaks.instances_from_file(args.tweaks)
		for tweak in tweaks:
			processor.apply_tweaks(tweak)

	#At some point we might run the processor on multiple chains
	#in that case we would call run more than once
	#and then finalize at the end
	processor.finalize()

if __name__=="__main__":
	main(sys.argv[1:])
