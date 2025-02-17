/*
mpirun -np 1 ./single-process-multiple-devices <num_gpus>
*/
#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <unistd.h> // for gethostname
#include <stdlib.h>    // for exit() and EXIT_FAILURE
#include <mpi.h>

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

#define MPI_CHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));
    
    int rank, worldSize;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    if (worldSize != 1) {
        printf("This program should be run with exactly 1 MPI process\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get number of devices from command line
    if (argc != 2) {
        printf("Usage: %s <num_gpus>\n", argv[0]);
        printf("Example: %s 4\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int nDev = atoi(argv[1]);
    
    // Check if requested number of GPUs is available
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (nDev <= 0 || nDev > deviceCount) {
        printf("Error: Requested %d GPUs but only %d are available\n", 
               nDev, deviceCount);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ncclComm_t comms[nDev];
    int size = 32*1024*1024;
    int devs[nDev];

    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
    printf("Process %d managing %d GPUs\n", rank, nDev);

    // Initialize devices
    for (int i = 0; i < nDev; i++) {
        devs[i] = i;
        CUDA_CHECK(cudaSetDevice(i));
        printf("\tInitializing GPU %d\n", i);
        CUDA_CHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
        
        // Initialize with different values for each GPU
        float* hostbuff = (float*)malloc(size * sizeof(float));
        for(int j = 0; j < size; j++) hostbuff[j] = i + 1.0f;
        CUDA_CHECK(cudaMemcpy(sendbuff[i], hostbuff, size * sizeof(float), cudaMemcpyHostToDevice));
        free(hostbuff);
        
        CUDA_CHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(s+i));
    }

    // Initialize NCCL communicators
    NCCL_CHECK(ncclCommInitAll(comms, nDev, devs));
    printf("Successfully initialized NCCL communicators\n");

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
    {
        NCCL_CHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
        printf("Launched NCCL AllReduce on device %d\n", i);
    }
    NCCL_CHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(s[i]));
        
        // Verify results
        float* results = (float*)malloc(size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(results, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Results on GPU %d: First element = %f (should be %f)\n", 
               i, results[0], (float)(nDev * (nDev + 1) / 2));
        free(results);
    }

    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(sendbuff[i]));
        CUDA_CHECK(cudaFree(recvbuff[i]));
        CUDA_CHECK(cudaStreamDestroy(s[i]));
        printf("Successfully freed device buffers for device %d\n", i);
    }

    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);
    
    free(sendbuff);
    free(recvbuff);
    free(s);

    MPI_CHECK(MPI_Finalize());
    printf("All done!\n");
    return 0;
}