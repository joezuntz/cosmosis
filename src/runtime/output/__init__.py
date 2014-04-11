from . import text_output
from . import cosmomc_output
from . import multi_text_output
from .output_base import output_registry



def output_from_options(options):
	# figure out the type of output required


	format = options['format']
	if format not in output_registry:
		known = '\n'.join('    %s - %s' % (f,output_registry[f].__name__) for f in output_registry)
		message = """
I do not know what format you meant by '%s' in the [output] ini file section.
I know about these format names:
%s
		""" % (format, known)
		print message
		raise KeyError("Unknown format")

	output_class = output_registry[format]
	return output_class.from_options(options)


def test():
	from . import test_output
	test_output.test_text()