ifeq (,$(COSMOSIS_SRC_DIR))
$(error "You must source config/setup-cosmosis before building.")
endif

ifeq (1,$(COSMOSIS_DEBUG))
  COMMON_FLAGS=-O0 -g -fPIC  -fno-omit-frame-pointer
else
  COMMON_FLAGS=-O3 -g -fPIC
endif

COMMON_C_FLAGS=$(COMMON_FLAGS) -I${COSMOSIS_SRC_DIR}/
OS=$(shell uname -s)
PEDANTIC_C_FLAGS=-Wall -Wextra -pedantic
CXXFLAGS=$(COMMON_C_FLAGS) $(USER_CXXFLAGS) -std=c++14 
CFLAGS=$(COMMON_C_FLAGS) -std=c99 $(USER_CFLAGS)
FFLAGS=$(COMMON_FLAGS) -I${COSMOSIS_SRC_DIR}/datablock -std=gnu -ffree-line-length-none $(USER_FFLAGS)
LDFLAGS=$(USER_LDFLAGS) -L${COSMOSIS_SRC_DIR}/datablock -Wl,-rpath,$(COSMOSIS_SRC_DIR)/datablock
PYTHON=python
MAKEFLAGS += --print-directory

ifeq (1,$(COSMOSIS_DEBUG))
LDFLAGS+=
endif


# Might be using 
ifeq (1,${COSMOSIS_OMP})
	ifeq (, $(COSMOSIS_OMP_FLAGS))
		COMMON_FLAGS+=$(COSMOSIS_OMP_FLAGS)
		LDFLAGS+=$(COSMOSIS_OMP_LDFLAGS)
	else ifeq (Darwin, $(OS))
		COMMON_FLAGS+= -fopenmp
		LDFLAGS+= -lomp
	else
		COMMON_FLAGS+= -fopenmp
		LDFLAGS+=-lgomp
	endif
endif

ifeq (Darwin, $(OS))
  LDFLAGS+=-headerpad_max_install_names
endif


# CXXFLAGS+= -isystem ${CONDA_PREFIX}/include 
# CFLAGS+= -isystem ${CONDA_PREFIX}/include
# LDFLAGS+= -L${CONDA_PREFIX}/lib