from __future__ import print_function
from . import text_output
from . import cosmomc_output
from . import null_output
from . import fits_output
from . import in_memory_output
from .output_base import output_registry
import logging


verbosity_levels = {
	"highest": 50-0,             #50
	"debug":50-logging.DEBUG,    #40
	"noisy":50-15,               #35
	"standard":50-logging.INFO,  #30
	"gentle":50-logging.WARNING, #20
	"quiet":50-logging.ERROR,    #10
	"silent":50-logging.FATAL,   #-1
}
#logging.basicConfig(format="- %(message)s",level=logging.DEBUG)

def set_verbosity(verb):
	try:
		verb = int(verb)
	except ValueError:
		pass
	if not isinstance(verb, int):
		try:
			verb = verbosity_levels[verb]
		except KeyError:
			valid_levels = ', '.join(list(verbosity_levels.keys()))
			message = """Error specifiying verbosity.
				You put: '{0}'.
				We want either an integer 0 (silent) - 50 (everything) 
				or one of: {1}""".format(verb, valid_levels)
			raise ValueError(message)
	level = 50 - verb
	logging.getLogger().setLevel(level)
	logging.debug("CosmoSIS verbosity set to %d"%(verb))

def output_from_options(options):
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

	verb = options.get("verbosity", "standard")
	set_verbosity(verb)
	return output_class.from_options(options)

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
