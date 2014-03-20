CC=gcc
CFLAGS=-fpic

DATABLOCK_DIR=../../datablock
module.so:
	$(CC) $(CFLAGS) -shared -o module.so  module.c -I $(DATABLOCK_DIR) -L $(DATABLOCK_DIR) -lcosmosis

clean:
	rm -f module.so
