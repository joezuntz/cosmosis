from __future__ import print_function
from cosmosis.output.text_output import TextColumnOutput
from cosmosis.output.fits_output import FitsOutput
from cosmosis.runtime.config import Inifile
import os


def read_input(filename, force_text=False, weighted=False):
    """
    Read cosmosis output data, either by:
     - specifying a cosmosis .txt output file
     - specifying a cosmosis .ini input file that includes the output file specification
     - specifying a directory containing cosmosis test sampler output
     - specifying a non-cosmosis output file containing samples or weighted samples


    Params:
        filename: string, paths to any of the options described above
        force_text: bool, default=False - regardless of the file ending assume it is text columns
        weighted: bool, default=False - if a non-cosmosis file is passed, assume that its samples have weights
    Returns:
        sampler - string with the name of the sampler in
        ini - a dictionary of information containing the output and metadata, suitable for instantiating a postprocessor.
    """
    if filename.endswith("txt") or force_text:
        output_info = TextColumnOutput.load_from_options({"filename":filename})
        metadata=output_info[2][0]
        sampler = metadata.get("sampler")
        if sampler is None:
            print("This is not a cosmosis output file.")
            print("So I will assume it is a generic MCMC file")
            if weighted:
                sampler = "weighted_metropolis"
            else:
                sampler = "metropolis"
            ini = output_info
        else:
            ini = {"sampler":sampler, sampler:metadata, "data":output_info, "output":dict(format="text", filename=filename)}
    elif filename.endswith("fits"):
        output_info = FitsOutput.load_from_options({"filename":filename})
        metadata=output_info[2][0]
        sampler = metadata.get("sampler")
        if sampler is None:
            print("This is not a cosmosis output file.")
            print("So I will assume it is a generic MCMC file")
            if weighted:
                sampler = "weighted_metropolis"
            else:
                sampler = "metropolis"
            ini = output_info
        else:
            ini = {"sampler":sampler, sampler:metadata, "data":output_info, "output":dict(format="fits", filename=filename)}

    elif os.path.isdir(filename):
        ini = Inifile(None)
        ini.add_section("runtime")
        ini.add_section("test")
        sampler = "test"
        ini.set("runtime", "sampler", sampler)
        ini.set("test", "save_dir", filename)
    else:
        #Determine the sampler and get the class
        #designed to postprocess the output of that sampler
        ini = Inifile(filename)
        sampler = ini.get("runtime", "sampler")
    return sampler, ini
