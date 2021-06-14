ifneq (clean,$(filter clean,$(MAKECMDGOALS)))
include ${COSMOSIS_SRC_DIR}/config/compilers.mk
endif

USER_LDFLAGS=-L/Users/jzuntz/src/cosmosis/conda_env3/lib -lcosmosis -Wl,-rpath,/Users/jzuntz/src/cosmosis/conda_env3/lib
USER_CFLAGS+=-I${COSMOSIS_SRC_DIR}/datablock
USER_FFLAGS+=-I${COSMOSIS_SRC_DIR}/datablock
USER_CXXFLAGS+=-I${COSMOSIS_SRC_DIR}/datablock

TEST_COMMANDS=ndarray_t datablock_t c_datablock_t c_datablock_int_array_t c_datablock_double_array_t \
			  c_datablock_complex_array_t c_datablock_multidim_double_array_t c_datablock_multidim_int_array_t \
			  c_datablock_multidim_complex_array_t section_t entry_t fortran_t



clean: 
	rm -f ${TEST_COMMANDS}

test:  test_entry test_section test_datablock test_c_datablock \
	test_ndarray \
	test_c_datablock_int_array test_c_datablock_double_array test_c_datablock_complex_array \
	test_c_datablock_multidim_double_array \
	test_c_datablock_multidim_int_array \
	test_c_datablock_multidim_complex_array \
	test_fortran

test_ndarray: ndarray_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_datablock: datablock_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock: c_datablock_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_int_array: c_datablock_int_array_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_double_array:  c_datablock_double_array_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_complex_array: c_datablock_complex_array_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_multidim_double_array: c_datablock_multidim_double_array_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_multidim_int_array: c_datablock_multidim_int_array_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_multidim_complex_array: c_datablock_multidim_complex_array_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_section: section_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

# entry_t doesn't depend oncosmosis
test_entry: entry_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_fortran: fortran_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed


fortran_t: cosmosis_test.F90
	$(FC) $(LDFLAGS)  $(FFLAGS)  -o $@ cosmosis_test.F90  -L. -lcosmosis

datablock_t: datablock_test.cc 
	$(CXX) $(LDFLAGS)  $(CXXFLAGS) -o $@ datablock_test.cc -L . -lcosmosis

c_datablock_t: c_datablock_test.c 
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ c_datablock_test.c -L . -lcosmosis

c_datablock_int_array_t: c_datablock_int_array_test.c 
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_double_array_t: c_datablock_double_array_test.c
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_complex_array_t: c_datablock_complex_array_test.c 
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_multidim_double_array_t: c_datablock_multidim_double_array_test.c 
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_multidim_int_array_t: c_datablock_multidim_int_array_test.c 
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_multidim_complex_array_t: c_datablock_multidim_complex_array_test.c 
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $< -L . -lcosmosis

entry_t: entry_test.cc
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $<

section_t: section_test.o
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $^

ndarray_t: ndarray_test.o
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $<

clean:
	rm -f *.o *.d *.so *.log *.mod *.mod 
	rm -f c_datablock_complex_array_t c_datablock_double_array_t c_datablock_int_array_t
	rm -f c_datablock_t
	rm -f c_datablock_multidim_double_array_t
	rm -f c_datablock_multidim_int_array_t
	rm -f c_datablock_multidim_complex_array_t
	rm -f datablock_t entry_t fortran_t ndarray_t section_t
	rm -rf  *.dSYM/
