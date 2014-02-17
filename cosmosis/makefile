CXXFLAGS=-std=c++11 -O3 -Wall -Wextra -pedantic -fPIC
CCFLAGS=-std=c99 -O3 -Wall -Wextra -pedantic -fPIC

all: libcosmosis.so c_datablock_t
test: all
	@LD_LIBRARY_PATH=. ./c_datablock_t > c_datablock_t.log || echo Failed c_datablock_t

libcosmosis.so: datablock.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $<

c_datablock_t: datablock.h c_datablock_t.c
	$(CC) $(CCFLAGS) -o $@ c_datablock_t.c -L . -lcosmosis

datablock.o: datablock.cc datablock.h

clean:
	rm -f *.o *~ *.so c_datablock_t *.log
