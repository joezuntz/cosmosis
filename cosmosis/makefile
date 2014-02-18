CXXFLAGS=-std=c++11 -O3 -Wall -Wextra -pedantic -fPIC
CCFLAGS=-std=c99 -O3 -Wall -Wextra -pedantic -fPIC

#MEMCHECK=valgrind --leak-check=full
MEMCHECK=

all: libcosmosis.so
test: test_datablock

test_datablock: all c_datablock_t
	@echo -n Running c_datablock_t
	@LD_LIBRARY_PATH=. $(MEMCHECK) ./c_datablock_t > c_datablock_t.log
	@echo  ... passed

libcosmosis.so: datablock.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

c_datablock_t: datablock.h c_datablock_t.c libcosmosis.so
	$(CC) $(CCFLAGS) -o $@ c_datablock_t.c -L . -lcosmosis

datablock.o: datablock.cc datablock.h

clean:
	rm -f *.o *~ *.so c_datablock_t *.log
	rm -rf *.dSYM
