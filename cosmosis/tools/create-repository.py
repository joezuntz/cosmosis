#!/usr/bin/env python
import os
import sys
import shutil

try:
	cosmosis_dir=os.environ['COSMOSIS_SRC_DIR']
except KeyError:
	sys.stderr.write("""
Please set up your cosmosis environment with source config/setup-cosmosis
or source setup-my-cosmosis before running this script
 """)
	sys.exit(1)

module_makefile_text = """
# If you already have your own Makefile you can 
# replace all of this, but you need to keep this line
# at the top:
include ${COSMOSIS_SRC_DIR}/config/compilers.mk


#For python, leave this as is.  For C/C++/Fortran, 
#comment the first line and uncomment the second:
all: 
#all %s.so

# ... and then provide the command to compile it, perhaps
# like this for a simple example in C:
#%s.so: my_files.c
#	$(CC) -shared -o %s.so my_files.c


#Replace this to put in some kind of test command for
#your module, if you have one.
test:
	@echo "Alas, %s/%s module has no tests"

#Add anything else to clean here
clean:
	rm -f *.o *.mod

"""




def create_library(library_name, *module_names):
	project_dir=os.path.join(cosmosis_dir, "modules", library_name)

	#Create the project directory
	if os.path.exists(project_dir):
		sys.stderr.write("A directory (or file) already exists for the project dir you specified\n")
		sys.exit(1)
	print "Making ", project_dir
	os.mkdir(project_dir)

	#Use modules/Makefile as a template
	old_makefile=os.path.join(cosmosis_dir, "modules", "Makefile")
	old_makefile_text=open(old_makefile).read()

	#Create the project Makefile
	if module_names:
		module_text = " ".join(module_names)
	else:
		module_text=""
	new_makefile=os.path.join(project_dir, "Makefile")
	new_makefile_text=old_makefile_text.replace("SUBDIRS =", "SUBDIRS = {0} \n#".format(module_text))
	open(new_makefile,"w").write(new_makefile_text)
	print "Writing ", new_makefile

	#Create directories for each module in the project and 
	#Give them Makefiles
	for module_name in module_names:
		module_dir = os.path.join(project_dir, module_name)
		os.mkdir(module_dir)
		module_makefile=os.path.join(module_dir, "Makefile")
		open(module_makefile,"w").write(module_makefile_text%(module_name,module_name,module_name,library_name,module_name))
		print "Writing ", module_makefile

	os.chdir(project_dir)
	print "Setting up repository:"
	print '--------------------'
	os.system("git init")
	os.system("git add -A")
	os.system("git commit -m 'Initial commit of %s'"%library_name)
	print '--------------------'

	#Update that parent makefile with the new text
	old_makefile_new_text=old_makefile_text.replace("SUBDIRS =", "SUBDIRS = "+library_name)
	open(old_makefile,"w").write(old_makefile_new_text+"\n")
	print "Updating ", old_makefile

import argparse
parser = argparse.ArgumentParser(description="Set up a new repository for cosmosis modules.")
parser.add_argument("repository_name", help="Name of repository to create")
parser.add_argument("module_names", nargs="+", help="Names of any initial modules to create in the repository (any number okay)")

if __name__ == '__main__':
	args=parser.parse_args()
	create_library(args.repository_name, *args.module_names)
