#SUBDIRS = src cosmosis-standard-library
SUBDIRS = src

# Include the standard variables and rules.
include ${COSMOSIS_DIR}/config/vars.mk
include ${COSMOSIS_DIR}/config/rules.mk
