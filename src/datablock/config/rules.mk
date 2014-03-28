
# ----- internal rules

all: $(LIBRARY) $(MODULES) $(TESTS) $(EXE_TARGETS) test

$(LIBRARY): $(OBJ_LIB)
	$(LINK.cc) -shared -o $@ $+ $(USER_LDLIBS)
	@echo done with library $@, using $<

build: $(SRC_AREA)config/rules.mk
	@echo "#!/usr/bin/env bash" > $@
	@echo "echo SOURCE_DIR is: ${SOURCE_DIR}" >> $@
	@echo "make -f ${SOURCE_DIR}/Makefile \"\$$@\""  >> $@
	@chmod +x $@
	@echo "local build file generated"

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

clean:
	rm -f *.o *.d $(EXE_TARGETS) *.P *.so $(TESTS)

build_tests: $(TESTS)

test: $(patsubst %_test,test_%,$(TESTS))

# Each test program should have source code ending in _test. We make a
# test_ target for each, to run the program.
#
test_% : %_test
	@echo -n Running $<
	@ $(MEMCHECK_CMD) $< > $<.log 2>&1
	@echo ...  passed	

% : %.o $(LIBRARY)
	$(LINK.cc) -o $@ $< $(LDLIBS)
	@echo done with f $@ $<

%_test : $(SRC_AREA)%_test.cc $(LIBRARY)
	$(LINK.cc) -o $@ $< $(LDLIBS)
	@echo done with f $@ $<

%_test : $(SRC_AREA)%_test.c $(LIBRARY)
	$(LINK.c) -o $@ $< $(LDLIBS)
	@echo done with f $@ $<

lib%_module.so: %_module.o
	$(LINK.cc) -shared -o $@ $+ $(LDLIBS)
	@echo done with module library $@, using $<

# Macro for post-processing dependency files.
# Thanks, SRT.
# We include some fixes for old infelicities.
# Thanks, Chris.
define postprocess_d
test -f $(dir $@)$(basename $(notdir $<)).d && \
cat $(dir $@)$(basename $(notdir $<)).d | \
sed 's?$*\.o?$(dir $@)$*.o ?g' > \
dep_tmp.$$$$ ; \
mv dep_tmp.$$$$ $(dir $@)$(basename $(notdir $<)).d
endef

# %.o : $(SRC_AREA)%.c
# 	$(COMPILE.c) -MMD -o $@ $<
# 	@cp $*.d $*.P; \
# 		sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
# 		-e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
# 	rm -f $*.d

# %.o : $(SRC_AREA)%.cc
# 	$(COMPILE.cc) -MMD -o $@ $<
# 	@cp $*.d $*.P; \
# 		sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
# 		-e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
# 	rm -f $*.d

%.o : $(SRC_AREA)%.cc
	$(COMPILE.cc) -MMD -o $@ $<
	$(postprocess_d)

%.o : $(SRC_AREA)%.c
	$(COMPILE.c) -MMD -o $@ $<
	$(postprocess_d)

#-include $(SRCS:%.c=$(DEPDIR)/%.P)
#-include $(SRC_ALL:%.c=$(DEPDIR)/%.d)
-include $(foreach x,$(notdir $(basename $(SRC_ALL))),$(DEPDIR)$(x).d)
