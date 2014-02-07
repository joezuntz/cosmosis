CXXFLAGS=-std=c++11 -O3 -Wall -Wextra -pedantic -fPIC

ALL: libsample.so c_prog

libsample.so: sample.o
	$(CXX) -shared -o $@ $<

c_prog: sample.h c_prog.c
	$(CC) -o $@ c_prog.c -L . -lsample

sample.o: sample.cc sample.h

clean:
	rm -f *.o *~ *.so
