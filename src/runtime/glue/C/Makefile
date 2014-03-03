CC=gcc
INCL=-I../inih 

CFLAGS=$(INCL) -m64 -std=c99 -O0 -g



#Make the very small library from this code.
all: lib


lib:
	$(CC) $(CFLAGS) -c internal_fits.c
	$(CC) $(CFLAGS) -c des_options.c

	ar rc libcfitswrap.a internal_fits.o des_options.o


test_option_read: test_option_read.c lib
	$(CC) $(CFLAGS) -o test_option_read test_option_read.c -L. -L../lib -lini -lcfitswrap

python:
	$(CC) $(CFLAGS) `../python_compilation_flags include` -c des_python_caller.c
	$(CC) $(CFLAGS) `../python_compilation_flags include` -c des_pipeline.c
	$(CC) $(CFLAGS) `../python_compilation_flags include` `../python_compilation_flags link` -L../lib -lini -lcfitswrap -lcfitsio des_python_caller.o des_pipeline.o test_python_caller.c -o test_python_caller


clean:
	rm -rf *.o libcfitswrap.a test_option_read test_option_read.dSYM

.PHONY: clean
