from .text_output import TextColumnOutput


def outputter_from_ini(ini):
	# figure out the type of output required
	# could do this from an explicit option
	# or by looking at the filename?

	if ini['format'] in ['text', 'txt']:
		output = TextColumnOutput.from_ini(ini)
	else:
		raise ValueError("I could not decide what kind of output you wanted.")
	return output


