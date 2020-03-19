ifeq (,$(COSMOSIS_SRC_DIR))
$(error "You must source config/setup-cosmosis before building.")
endif

ifeq (1,$(COSMOSIS_DEBUG))
  COMMON_FLAGS=-O0 -g -fPIC  -fno-omit-frame-pointer
else
  COMMON_FLAGS=-O3 -g -fPIC
endif

COMMON_C_FLAGS=$(COMMON_FLAGS) -I${COSMOSIS_SRC_DIR}/..
OS=$(shell uname -s)
PEDANTIC_C_FLAGS=-Wall -Wextra -pedantic
CXXFLAGS=$(COMMON_C_FLAGS) $(USER_CXXFLAGS) -std=c++14
CFLAGS=$(COMMON_C_FLAGS) $(USER_CFLAGS) -std=c99
FFLAGS=$(COMMON_FLAGS) -I${COSMOSIS_SRC_DIR}/datablock $(USER_FFLAGS) -std=gnu -ffree-line-length-none
LDFLAGS=$(USER_LDFLAGS) -L${COSMOSIS_SRC_DIR}/datablock -Wl,-rpath,$(COSMOSIS_SRC_DIR)/datablock
PYTHON=python
MAKEFLAGS += --print-directory

ifeq (1,$(COSMOSIS_DEBUG))
LDFLAGS+=
endif

ifeq (1,${COSMOSIS_OMP})
COMMON_FLAGS+= -fopenmp
LDFLAGS+=-lgomp
endif

ifeq (Darwin, $(OS))
  LDFLAGS+=-headerpad_max_install_names
endif
