ifneq (clean,$(MAKECMDGOALS))
include cosmosis/compilers.mk
else
ifeq (${COSMOSIS_SRC_DIR},)
COSMOSIS_SRC_DIR=${PWD}
$(info here we are)
export COSMOSIS_SRC_DIR
else
COSMOSIS_SRC_DIR=${COSMOSIS_SRC_DIR}
endif
endif

include cosmosis/subdirs.mk



all:: config/first

config/first:
	@echo 
	@echo "Cosmosis compilation complete!"
	@echo "If this is your first time why not run Demo 1 now like this:"
	@echo "cosmosis demos/demo1.ini"
	@echo
	@touch config/first

SUBDIRS=cosmosis
