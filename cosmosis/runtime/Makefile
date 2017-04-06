include ${COSMOSIS_SRC_DIR}/config/compilers.mk

USER_CFLAGS=$(PEDANTIC_C_FLAGS) -Werror
USER_CXXFLAGS=$(PEDANTIC_C_FLAGS)


all: experimental_fault_handler.so

experimental_fault_handler.so:
	-$(CC) -fPIC -shared handler.c -o experimental_fault_handler.so

clean:
	rm experimental_fault_handler.so
