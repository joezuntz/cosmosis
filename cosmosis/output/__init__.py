from .text_output import TextColumnOutput
from .cosmomc_output import CosmoMCOutput
from .null_output import NullOutput
from .fits_output import FitsOutput
from .in_memory_output import InMemoryOutput
from .astropy_output import AstropyOutput
from .output_base import output_registry, OutputBase


def output_from_options(options, resume=False):
	# figure out the type of output required
	format = options.get('format', 'text')
	if format not in output_registry:
		known = '\n'.join('    %s - %s' % (f,output_registry[f].__name__) for f in output_registry)
		message = """
I do not know what format you meant by '%s' in the [output] ini file section.
I know about these format names:
%s
		""" % (format, known)
		print(message)
		raise KeyError("Unknown format")

	output_class = output_registry[format]

	return output_class.from_options(options,resume)

def input_from_options(options):
    format = options['format']
    if format not in output_registry:
        known = '\n'.join('    %s - %s' % (f,output_registry[f].__name__) for f in output_registry)
        message = """
I do not know what format you meant by '%s' in the [output] ini file section.
I know about these format names:
%s
        """ % (format, known)
        print(message)
        raise KeyError("Unknown format")

    output_class = output_registry[format]
    return output_class.load_from_options(options)

def test():
	from . import test_output
	test_output.test_text()
