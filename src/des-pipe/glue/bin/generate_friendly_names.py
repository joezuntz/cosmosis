#!/usr/bin/env python

def load_name_list(filename):
	output = []
	for line in open(filename):
		line = line.strip()
		if not line: continue
		if line.startswith('#'):
			continue
		code, friendly = line.split()
		code = code.upper()
		friendly = friendly.lower()
		output.append((friendly,code))
	return output

def generate_fortran_names(names, outfile):
	f = open(outfile,"w")
	fmt = '    character(*), parameter :: %s_section = "%s"'
	text = '\n'.join([fmt % pair for pair in names])
	f.write("""
! These code/friendly names are generated
! automatically by the script 
! des-pipe/glue/bin/generate_friendly_names.py
! from the file des-pipe/glue/section_names.txt

! If you want to add friendly section names please
! modify that latter file. 
 


module des_section_names
%s
end module des_section_names
"""%text)
	f.close()



def generate_c_names(names, outfile):
	f = open(outfile,"w")
	fmt = '#define %s_SECTION "%s"'
	text = '\n'.join([fmt % (name.upper(), code) for (name,code) in names])
	f.write("""
/* These code/friendly names are generated
   automatically by the script 
   des-pipe/glue/bin/generate_friendly_names.py
   from the file des-pipe/glue/section_names.txt

   If you want to add friendly section names please
   modify that latter file. 
 */

#ifdef _H_DES_SECTION_NAME
#else
#define _H_DES_SECTION_NAME

%s

#endif

""" % text)
	f.close()

def generate_py_names(names, outfile):
	f = open(outfile,"w")
	fmt = '    %s = "%s"'
	text = '\n'.join([fmt % pair for pair in names])
	f.write("""
# These code/friendly names are generated
# automatically by the script 
# des-pipe/glue/bin/generate_friendly_names.py
# from the file des-pipe/glue/section_names.txt

# If you want to add friendly section names please
# modify that latter file. 

class section_names(object):
    "Friendly-code name translations"
%s
"""%text)
	f.close()

def main():
	fortran_outfile = "fortran/des_section_names.f90"
	c_outfile = "C/des_section_names.h"
	py_outfile = "python/pydesglue/section_names.py"
	names = load_name_list("section_list.txt")
	generate_fortran_names(names, fortran_outfile)
	generate_c_names(names, c_outfile)
	generate_py_names(names, py_outfile)


if __name__=="__main__":
	main()


