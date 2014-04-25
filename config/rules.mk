$(error "rules.mk is obsolete; you should not be using it")

#define subdirs_macro
#	@for d in $(SUBDIRS); \
#	do \
#	  (mkdir -p $$d && cd $$d && make -f $(SRC_AREA)$$d/Makefile $@); \
#	done
#endef
define subdirs_macro
   @for d in $(SUBDIRS); \
   do \
     (cd $$d && make -f $(SRC_AREA)$$d/Makefile $@); \
   done
endef

# ----- internal rules

all: library $(MODULES) $(EXE_TARGETS) build_tests
	$(subdirs_macro)

# $(BUILD_TOP)/lib is the directory into which we build all dynamic libraries
$(BUILD_TOP)/lib:
	@echo "Creating $(BUILD_TOP)/lib"
	@mkdir -p $(BUILD_TOP)/lib

library: $(LIBRARY)
	$(subdirs_macro)

$(LIBRARY): $(OBJ_LIB) $(BUILD_TOP)/lib
	@echo Linking library $(LIBRARY) from $(OBJ_LIB)
	$(LINK.cc) -shared -o $@ $(OBJ_LIB) $(USER_LDLIBS)

build: ${COSMOSIS_DIR}/config/rules.mk
	@echo "#!/usr/bin/env bash" > $@
	@echo "make -f $(SRC_AREA)Makefile \"\$$@\""  >> $@
	@chmod +x $@
	@echo "local build file generated in $${PWD}"
	$(subdirs_macro)

print:
	@echo "EXE_TARGETS = $(EXE_TARGETS)"
	@echo "LIBRARY = $(LIBRARY)"
	@echo "MODULES = $(MODULES)"
	@echo "TESTS = $(TESTS)"
	@echo "SRC_ALL = $(SRC_ALL)"
	@echo "BLD_AREA = $(BLD_AREA)"
	@echo "SRC_AREA = $(SRC_AREA)"
	@echo "SRC_MODULES = $(SRC_MODULES)"
	@echo "OBJ_MODULES = $(OBJ_MODULES)"
	@echo "SRC_TESTS = $(SRC_TESTS)"
	@echo "OBJ_TESTS = $(OBJ_TESTS)"
	@echo "SRC_TARGETS = $(SRC_TARGETS)"
	@echo "OBJ_TARGETS = $(OBJ_TARGETS)"
	@echo "SRC_LIB = $(SRC_LIB)"
	@echo "OBJ_LIB = $(OBJ_LIB)"
	@echo "MAKEFILE_LIST = $(MAKEFILE_LIST)"
	@echo "MAKECMDGOALS = $(MAKECMDGOALS)"
	@echo "DEPDIR = $(DEPDIR)"
	@echo "LDLIBS = $(LDLIBS)"
	@echo "-----diving into subdirectories-----"
	$(subdirs_macro)

clean:
	rm -f *.o *.d $(EXE_TARGETS) *.P *.so $(TESTS) *.log
	rm -rf *.dSYM/
	$(subdirs_macro)

build_tests: $(TESTS)
	$(subdirs_macro)

test: $(patsubst %_test,test_%,$(TESTS))
	$(subdirs_macro)

# Each test program should have source code ending in _test. We make a
# test_ target for each, to run the program.
#
test_% : %_test
	@echo -n Running $<
	@LD_LIBRARY_PATH=.:$(LD_LIBRARY_PATH) $(MEMCHECK_CMD) ./$< > $<.log 2>&1
	@echo ...  passed	

% : %.o $(LIBRARY)
	@echo Building executable from $<
	$(LINK.cc) -o $@ $< $(LDLIBS)
	@echo done with $@ $<

%_test : $(SRC_AREA)%_test.cc $(LIBRARY)
	@echo Building test from $<
	$(LINK.cc) -o $@ $< $(LDLIBS)
	@echo done with $@ $<

%_test : $(SRC_AREA)%_test.c $(LIBRARY)
	@echo Building test from $<
	$(LINK.c) -o $@ $< $(LDLIBS)
	@echo done with $@ $<

%_test : $(SRC_AREA)%_test.F90 $(LIBRARY)
	@echo Building test from $<
	$(LINK.f) -o $@ $< $(LDLIBS)
	@echo done with $@ $<

%_test : $(SRC_AREA)%_test.f90 $(LIBRARY)
	@echo Building test from $<
	$(LINK.f) -o $@ $< $(LDLIBS)
	@echo done with $@ $<

%_module.so: %_module.o $(LIBRARY)
	@echo Building module from $<
	$(LINK.cc) -shared -o $@ $+ $(USER_LDLIBS)
	@echo done with module library $@, using $<

# lib%_module.so: $(SRC_AREA)%_module.c $(LIBRARY)
# 	@echo Building module from $+
# 	$(LINK.c) -shared -o $@ $+ $(LDLIBS)
# 	@echo done with module library $@, using $<

# Macro for post-processing dependency files.
# Thanks, SRT.
# We include some fixes for old infelicities.
# Thanks, Chris.
define postprocess_d
	@test -f $(dir $@)$(basename $(notdir $<)).d && \
	cat $(dir $@)$(basename $(notdir $<)).d | \
	sed 's?$*\.o?$(dir $@)$*.o ?g' > \
	dep_tmp.$$$$ ; \
	mv dep_tmp.$$$$ $(dir $@)$(basename $(notdir $<)).d
endef

%.o : $(SRC_AREA)%.cc
	@echo Compiling source $<
	$(COMPILE.cc) -MMD -o $@ $<
	$(postprocess_d)

%.o : $(SRC_AREA)%.c
	@echo Compiling source $<
	$(COMPILE.c) -MMD -o $@ $<
	$(postprocess_d)

%.o : $(SRC_AREA)%.F90
	@echo Compiling source $<
	$(COMPILE.F) -MMD -o $@ $<

%.o : $(SRC_AREA)%.f90
	@echo Compiling source $<
	$(COMPILE.f) -MMD -o $@ $<

-include $(foreach x,$(notdir $(basename $(SRC_ALL))),$(DEPDIR)$(x).d)
