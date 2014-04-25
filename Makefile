ifeq (,$(COSMOSIS_SRC_DIR))
$(error "You must run config/setup_for_dev.sh <path-to-ups-product-directory> before building.")
endif

include config/compilers.mk
include config/subdirs.mk


SUBDIRS=cosmosis example-modules cosmosis-standard-library
