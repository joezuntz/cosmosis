from __future__ import print_function

try:
	from cosmosis import names as section_names
except ImportError:
	print("Running without cosmosis: no pretty section names")
	section_names = None


class NoSuchParameter(Exception):
	pass

def section_code(code):
	if section_names:
		for att in dir(section_names):
			if getattr(section_names,att)==code:
				return att
	return code

