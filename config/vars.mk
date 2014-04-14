
# --- internal environment

DOLLAR := $$
VAR := (MAKECMDGOALS)
#BLD_AREA:=./
BLD_AREA:=$(PWD)
SRC_AREA:=$(firstword $(dir $(MAKEFILE_LIST)))

SRC_TARGETS:=$(TARGETS)
SRC_ALL:=$(wildcard $(SRC_AREA)*.c) $(wildcard $(SRC_AREA)*.c*)
SRC_MODULES:=$(wildcard $(SRC_AREA)*_module.c*)
SRC_TESTS:=$(wildcard $(SRC_AREA)*_test.c*)
SRC_LIB:=$(filter-out $(SRC_MODULES) $(SRC_TESTS) $(SRC_TARGETS), $(SRC_ALL))

OBJ_MODULES:=$(patsubst %.cc,%.o,$(notdir $(SRC_MODULES)))
OBJ_MODULES:=$(patsubst %.c,%.o,$(notdir $(OBJ_MODULES)))

OBJ_TESTS:=$(patsubst %.cc,%.o,$(notdir $(SRC_TESTS)))
OBJ_TESTS:=$(patsubst %.c,%.o,$(notdir $(OBJ_TESTS)))

OBJ_TARGETS:=$(patsubst %.cc,%.o,$(notdir $(SRC_TARGETS)))

OBJ_LIB:=$(patsubst %.cc,%.o,$(notdir $(SRC_LIB)))
OBJ_LIB:=$(patsubst %.c,%.o,$(notdir $(OBJ_LIB)))

# It looks like it may not be possible to ever have DEPDIR be something
# other than '.'. This is because we're generating the .d files with the
# -MMD flag of GCC at the same time we're generating the .o file, the .d
# file is put into the same directory as the .o file. If there is a way
# to redirect this file, then the macro for postprocessing the .d file
# will have to be modfied to make use of DEPDIR.
# We could use -MF <file> in conjunction with -MMD to control the location
# of the .d file.
#
# TODO: Consider eliminating DEPDIR.

DEPDIR:=./
DEPFILE = $(DEPDIR)$(*F)

ifneq ($(OBJ_LIB),)
LIBNAME = $(lastword $(subst /, ,$(SRC_AREA)))
LIBRARY = lib$(LIBNAME).so
endif
MODULES = $(patsubst %.o,%.so,$(OBJ_MODULES))
TESTS = $(basename $(OBJ_TESTS))
EXE_TARGETS = $(basename $(OBJ_TARGETS))

CXX=g++
CC=gcc
COMMON_C_FLAGS=-O3 -g -fPIC -Wall -Wextra -pedantic -I$(SRC_AREA) -I$(BUILD_TOP) -I$(COSMOSIS_DIR)/src $(USER_CXXFLAGS)
CXXFLAGS=$(COMMON_C_FLAGS) -std=c++11
CFLAGS=$(COMMON_C_FLAGS) -std=c99
LDFLAGS=$(USER_LDFLAGS) -L.
LDLIBS=$(USER_LDLIBS) -l$(LIBNAME)

PYTHON=python

ifdef MEMCHECK
	MEMCHECK_CMD=valgrind --error-exitcode=1 --leak-check=yes --errors-for-leak-kinds=definite --track-origins=yes --suppressions=$(SOURCE_DIR)/cosmosis_tests.supp
else
  MEMCHECK_CMD=
endif
