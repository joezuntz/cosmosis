# --- user defined environment

TARGETS =
USER_CXXFLAGS =
USER_LDFLAGS =
USER_LDLIBS =

include ${COSMOSIS_DIR}/config/vars.mk

section_names.py: $(SOURCE_DIR)/datablock/generate_sections.py $(SOURCE_DIR)/datablock/section_names.txt
	$(PYTHON) $+ "python" $@

include ${COSMOSIS_DIR}/config/rules.mk

library: section_names.py

