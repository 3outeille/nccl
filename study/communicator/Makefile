# First, source the environment setup and get the variables
CUDA_HOME ?= $(shell source ../setup_env.sh && echo $$CUDA_HOME)
NCCL_HOME ?= $(shell source ../setup_env.sh && echo $$NCCL_HOME)
MPI_HOME ?= $(shell echo $$MPI_PATH)

CC := mpicc
NVCC := $(CUDA_HOME)/bin/nvcc

# Include directories
INCLUDES := -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -I$(MPI_HOME)/include

# Library directories and libraries
LDFLAGS := -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -L$(MPI_HOME)/lib
LIBS := -lnccl -lcudart -lmpi

# Compile flags
# -std=gnu99: Use GNU C99 standard
# -D_POSIX_C_SOURCE=200112L: Define POSIX C source to ensure necessary POSIX features are available (e.g. gethostname from unistd.h)
CFLAGS := -O3 -std=gnu99 -D_POSIX_C_SOURCE=200112L
NVCCFLAGS := -O3

# Target executables
TARGETS := single-process-multiple-devices single-process-single-device multiple-processes-multiple-devices

# Print paths for debugging
$(info CUDA_HOME is $(CUDA_HOME))
$(info NCCL_HOME is $(NCCL_HOME))
$(info MPI_HOME is $(MPI_HOME))

# Rules
all: $(TARGETS)

single-process-multiple-devices: single-process-multiple-devices.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(LIBS)

single-process-single-device: single-process-single-device.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(LIBS)

multiple-processes-multiple-devices: multiple-processes-multiple-devices.c
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(LIBS)

# test: test.c
# 	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGETS)
