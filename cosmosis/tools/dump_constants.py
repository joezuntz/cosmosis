"""

You don't need to use this script.

It is for the developers to make header files.

"""

print __doc__
import sys
#You don't need to use this file - it is for the developers to 
try:
	import astropy.constants
	import tabulate
except ImportError:
	print "I can tell you aren't really a developer because you don't have"
	print "the dependencies (tabulate, astropy)."

def find_constants():
	consts = []
	for x in dir(astropy.constants):
		c = getattr(astropy.constants, x)
		if isinstance(c, astropy.constants.constant.Constant):
			consts.append(c)
	return consts

class ConstantFile(object):
	def write(self, constants):
		lines = []
		for c in constants:
			parts = [
				"{self.prefix}{c.abbrev}".format(self=self,c=c),
				"=",
				"{}".format(float(c.value)),
				"{self.comment} {c.unit}".format(self=self,c=c),
				"{self.comment} {c.name}".format(self=self,c=c),
			]
			lines.append(parts)
		f = open(self.filename, 'w')
		f.write(self.header)
		f.write(tabulate.tabulate(lines,tablefmt="plain", numalign='left',floatfmt="e"))
		f.write("\n")
		f.write(self.footer)
		f.write("\n")
		f.close()



class Python(ConstantFile):
	filename = "cosmosis/constants.py"
	header = ""
	prefix = ""
	comment = "#"
	footer = ""

class C(ConstantFile):
	filename = "cosmosis/datablock/cosmosis_constants.h"
	header = "#ifndef COSMOSIS_CONSTANTS_H\n#define COSMOSIS_CONSTANTS_H\n"
	prefix = "static const double constant_"
	comment = "//"
	footer = "#endif /* COSMOSIS_CONSTANTS_H */"

class Fortran(ConstantFile):
	filename = "cosmosis/datablock/cosmosis_constants.fh"
	header = ""
	prefix = "real(8), parameter ::  constant_"
	comment = "!"
	footer = ""




def main():
	constants = find_constants()
	for cls in [Python, C, Fortran]:
		cls().write(constants)


if __name__ == '__main__':
	main()