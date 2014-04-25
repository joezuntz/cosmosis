# --- user defined environment

TARGETS =
USER_CXXFLAGS =
USER_LDFLAGS =
USER_LDLIBS =

SUBDIRS = cosmosis_f90 cosmosis_py

include ${COSMOSIS_DIR}/config/vars.mk

# We override the library name for this directory's library, which is
# special.
LIBNAME = cosmosis

section_names.h: $(SOURCE_DIR)/datablock/generate_sections.py $(SOURCE_DIR)/datablock/section_names.txt
	$(PYTHON) $+ "c" $@

c_datablock.o: section_names.h

include ${COSMOSIS_DIR}/config/rules.mk
