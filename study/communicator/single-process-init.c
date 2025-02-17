#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <unistd.h> // for gethostname
#include <stdlib.h>    // for exit() and EXIT_FAILURE

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed: NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[]) {

    /*
    ncclCommInitAll (Single Process, Multiple Devices):
    - Used when one process manages multiple GPUs
    - Simplest method for single-node scenarios
    - Cannot be used for inter-node communication
    - This is what your code is using1
    */

    const int nDev = 4; // Number of GPUs to use
    ncclComm_t comms[nDev];
    int devs[nDev];
    char hostname[1024];
    gethostname(hostname, 1024);

    // Initialize devices
    for (int i = 0; i < nDev; i++) {
        devs[i] = i;
        CUDA_CHECK(cudaSetDevice(i));
        printf("Initializing device %d on host %s\n", i, hostname);
    }

    // Initialize NCCL communicators
    // For single process, multiple devices
    NCCL_CHECK(ncclCommInitAll(comms, nDev, devs));
    printf("Successfully initialized NCCL communicators\n");

    // Clean up
    for (int i = 0; i < nDev; i++) {
        ncclCommDestroy(comms[i]);
    }
    
    printf("Successfully destroyed NCCL communicators\n");
    return 0;
} 