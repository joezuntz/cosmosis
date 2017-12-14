ifneq (clean,$(MAKECMDGOALS))
include cosmosis/compilers.mk
endif

$(info ${COSMOSIS_SRC_DIR})

include ${COSMOSIS_SRC_DIR}/cosmosis/subdirs.mk

SUBDIRS=datablock samplers runtime
