include config/compilers.mk
include config/subdirs.mk

welcome_message=echo Welcome


all::
	@echo 
	@echo "Cosmosis compilation complete!"
	@echo "If this is your first time why not run Demo 1 now like this:"
	@echo "cosmosis demos/demo1.ini"
	@echo

SUBDIRS=cosmosis example-modules cosmosis-standard-library modules
