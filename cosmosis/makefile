#----------------------------------------------------------------------
#
# Preliminary version of make-based build. This version does an
# in-source build. Users must be aware to never add a build-product to
# the repository.
#
# To add a new test takes two steps:
#   1) define the new test target. Follow the example of test_datablock
#   2) add the new test target to the dependencies of 'test'
#
# To run tests using valgrind (which only works if you have valgrind
# available!), use:
#    $>  MEMCHECK=1 make test
# If you always want to run tests with valgrind, you can just set
# MEMCHECK in your environment. This is not recommended, because
# testing will then be slow, always.
#
# Bugs and bad features:
#   1) dependencies for targets are not automatically generated.
#
#----------------------------------------------------------------------
#CXXFLAGS=-std=c++11 -O3 -Wall -Wextra -pedantic -fPIC -g
CXXFLAGS=-std=c++11 -O0 -Wall -Wextra -pedantic -fPIC -g
#CCFLAGS=-std=c99 -O3 -Wall -Wextra -pedantic -fPIC -g
CCFLAGS=-std=c99 -O0 -Wall -Wextra -pedantic -fPIC -g

ifdef MEMCHECK
  MEMCHECK_CMD=valgrind --error-exitcode=1 --leak-check=yes --errors-for-leak-kinds=definite --track-origins=yes --suppressions=cosmosis_tests.supp
else
  MEMCHECK_CMD=
endif

all: verify_setup lib

lib: libcosmosis.so

verify_setup:
	@if [ -z "$(SETUP_UPS)" ]; then echo "You must setup UPS first" && exit 1; fi

test:  test_entry test_section test_datablock test_c_datablock

test_datablock: lib datablock_t
	@echo -n Running datablock_t
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./datablock_t > datablock_t.log
	@echo  ... passed

test_c_datablock: lib c_datablock_t
	@echo -n Running c_datablock_t
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./c_datablock_t > c_datablock_t.log
	@echo  ... passed

test_section: lib section_t
	@echo -n Running section_t
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./section_t >  section_t.log
	@echo  ... passed

test_entry: lib entry_t
	@echo -n Running entry_t
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./entry_t >  entry_t.log
	@echo  ... passed

libcosmosis.so: datablock.o entry.o section.o c_datablock.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $+

datablock_t: datablock_t.cc datablock.hh libcosmosis.so
	$(CXX) $(CXXFLAGS) -o $@ datablock_t.cc -L . -lcosmosis

c_datablock_t: c_datablock_t.c c_datablock.h libcosmosis.so
	$(CC) $(CCFLAGS) -o $@ c_datablock_t.c -L . -lcosmosis

entry_t: entry_t.o libcosmosis.so
	$(CXX) $(CXXFLAGS) -o $@ $< -L . -lcosmosis

section_t: section_t.o libcosmosis.so
	$(CXX) $(CXXFLAGS) -o $@ $< -L . -lcosmosis

datablock.o: datablock.cc datablock.hh c_datablock.h entry.hh datablock_status.h
c_datablock.o: c_datablock.cc datablock.hh c_datablock.h entry.hh datablock_status.h
entry.o: entry.cc entry.hh datablock_status.h
entry_t.o: entry_t.cc entry.hh
section.o: section.cc section.hh entry.hh datablock_status.h
section_t.o: section_t.cc section.hh

clean:
	rm -f *.o *~ *.so c_datablock_t *.log entry_t section_t
	rm -rf *.dSYM
