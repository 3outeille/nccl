# First, source the environment setup and get the variables
CUDA_HOME ?= $(shell source ../setup_env.sh && echo $$CUDA_HOME)
NCCL_HOME ?= $(shell source ../setup_env.sh && echo $$NCCL_HOME)

CC := gcc
NVCC := $(CUDA_HOME)/bin/nvcc

# Include directories
INCLUDES := -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include

# Library directories and libraries
LDFLAGS := -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LIBS := -lnccl -lcudart

# Compile flags
# -std=gnu99: Use GNU C99 standard
# -D_POSIX_C_SOURCE=200112L: Define POSIX C source to ensure necessary POSIX features are available (e.g. gethostname from unistd.h)
CFLAGS := -O3 -std=gnu99 -D_POSIX_C_SOURCE=200112L
NVCCFLAGS := -O3

# Target executables
TARGETS := single-process-init multiple-processes-init config-init

# Print paths for debugging
$(info CUDA_HOME is $(CUDA_HOME))
$(info NCCL_HOME is $(NCCL_HOME))

# Rules
all: $(TARGETS)

single-process-init: single-process-init.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(LIBS)

multiple-processes-init: multiple-processes-init.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(LIBS)

with-config-init: with-config-init.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGETS)
