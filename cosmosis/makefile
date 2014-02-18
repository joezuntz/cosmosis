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
CXXFLAGS=-std=c++11 -O3 -Wall -Wextra -pedantic -fPIC
CCFLAGS=-std=c99 -O3 -Wall -Wextra -pedantic -fPIC

ifdef MEMCHECK
  MEMCHECK_CMD=valgrind --leak-check=full
else
  MEMCHECK_CMD=
endif

all: libcosmosis.so
test: test_datablock

test_datablock: all c_datablock_t
	@echo -n Running c_datablock_t
	@LD_LIBRARY_PATH=. $(MEMCHECK_CMD) ./c_datablock_t > c_datablock_t.log
	@echo  ... passed

libcosmosis.so: datablock.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

c_datablock_t: datablock.h c_datablock_t.c libcosmosis.so
	$(CC) $(CCFLAGS) -o $@ c_datablock_t.c -L . -lcosmosis

datablock.o: datablock.cc datablock.h

clean:
	rm -f *.o *~ *.so c_datablock_t *.log
	rm -rf *.dSYM
