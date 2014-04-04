fortran_template = """
! This module is auto-generated from the file section_list.txt.
! Edit that and then re-make to add your own pre-defined section names.

module cosmosis_section_names


    implicit none
{0}

end module

"""

c_template = """
// This header file is auto-generated from the file section_list.txt.
// Edit that and then re-make to add your own pre-defined section names.

{0}

"""

python_template = """
# This module is auto-generated from the file section_list.txt.
# Edit that and then re-make to add your own pre-defined section names.

{0}

"""

def generate_python(section_names, filename):
	sections = "\n".join('{0} = "{0}"'.format(name) for name in section_names)
	open(filename,'w').write(python_template.format(sections))

def generate_fortran(section_names, filename):
	sections = "\n".join('    character(*), parameter :: {0}_section = "{0}"'.format(name) for name in section_names)
	open(filename,'w').write(fortran_template.format(sections))

def generate_c(section_names, filename):
	sections = "\n".join('#define {0}_SECTION {1}'.format(name.upper(),name) for name in section_names)
	open(filename,'w').write(c_template.format(sections))

def generate(section_list_filename):
	section_names = []
	for line in open(section_list_filename):
		line=line.strip()
		if line.startswith('#') or not line:
			continue
		line=line.split('#')[0].strip()
		section_names.append(line)
	generate_c(section_names, 'section_names.h')
	generate_python(section_names, 'cosmosis_py/section_names.py')
	generate_fortran(section_names, 'cosmosis_f90/cosmosis_section_names.F90')


if __name__ == '__main__':
	generate("section_list.txt")