# --- user defined environment

TARGETS =
USER_CXXFLAGS =
USER_LDFLAGS =
USER_LDLIBS =

include ${COSMOSIS_DIR}/config/vars.mk

cosmosis_section_names.F90: $(SOURCE_DIR)/datablock/generate_sections.py $(SOURCE_DIR)/datablock/section_names.txt
	$(PYTHON) $+ "fortran" $@

include ${COSMOSIS_DIR}/config/rules.mk

library: cosmosis_section_names.F90
