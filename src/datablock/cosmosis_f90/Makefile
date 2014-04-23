# --- user defined environment

TARGETS =
USER_CXXFLAGS =
USER_LDFLAGS =
USER_LDLIBS =

include ${COSMOSIS_DIR}/config/vars.mk

cosmosis_section_names.F90: $(SOURCE_DIR)/datablock/generate_sections.py $(SOURCE_DIR)/datablock/section_names.txt
	$(PYTHON) $+ "fortran" $@

cosmosis_wrappers.o: cosmosis_types.mod
cosmosis_test.o: cosmosis_modules.mod
cosmosis_modules.o: cosmosis_section_names.mod cosmosis_types.mod cosmosis_wrappers.mod

cosmosis_section_names.mod: cosmosis_section_names.F90
	$(COMPILE.f) -o /dev/null $<

cosmosis_types.mod: $(SOURCE_DIR)/datablock/cosmosis_f90/cosmosis_types.F90
	$(COMPILE.f) -o /dev/null $<

cosmosis_wrappers.mod: $(SOURCE_DIR)/datablock/cosmosis_f90/cosmosis_wrappers.F90
	$(COMPILE.f) -o /dev/null $<

cosmosis_modules.mod: $(SOURCE_DIR)/datablock/cosmosis_f90/cosmosis_modules.F90
	$(COMPILE.f) -o /dev/null $<

include ${COSMOSIS_DIR}/config/rules.mk

USER_LDLIBS=-lgfortran -lcosmosis

library: cosmosis_section_names.mod cosmosis_types.mod cosmosis_wrappers.mod cosmosis_modules.mod \
	cosmosis_section_names.F90
