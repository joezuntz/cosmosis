include config/compilers.mk
include config/subdirs.mk

all:: config/first

config/first:
	@echo 
	@echo "Cosmosis compilation complete!"
	@echo "If this is your first time why not run Demo 1 now like this:"
	@echo "cosmosis demos/demo1.ini"
	@echo
	@touch config/first

SUBDIRS=cosmosis example-modules cosmosis-standard-library modules


ifneq ($(wildcard cosmosis-des-library/*),)
SUBDIRS+=cosmosis-des-library
$(info )
$(info    Compiling DES code in cosmosis-des-library)
$(info )
endif
