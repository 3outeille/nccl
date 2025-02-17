/*
mpirun -np <num_processes> ./multiple-processes-multiple-devices <num_gpus_per_proc>
Example: mpirun -np 2 ./multiple-processes-multiple-devices 2  # Each process handles 2 GPUs
*/
#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <unistd.h> // for gethostname
#include <stdlib.h>    // for exit() and EXIT_FAILURE
#include <mpi.h>
#include <stdint.h>

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

static uint64_t getHostHash(const char* string) {
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));
    
    int rank, worldSize;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    // Get number of devices per process from command line
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <num_gpus_per_proc>\n", argv[0]);
            printf("Example: %s 2\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int nDevPerProc = atoi(argv[1]);
    const int totalGPUs = nDevPerProc * worldSize;
    
    // Check if requested number of GPUs is available
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (totalGPUs > deviceCount) {
        if (rank == 0) {
            printf("Error: Total GPUs requested (%d) exceeds available GPUs (%d)\n", 
                   totalGPUs, deviceCount);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Add local rank calculation
    uint64_t* hostHashs = (uint64_t*)malloc(worldSize * sizeof(uint64_t));
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                           hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    
    int localRank = 0;
    for (int p = 0; p < worldSize; p++) {
        if (p == rank) break;
        if (hostHashs[p] == hostHashs[rank]) localRank++;
    }
    free(hostHashs);

    // Modify GPU selection to use localRank
    const int startGpu = localRank * nDevPerProc;  // Changed from rank to localRank
    
    ncclComm_t comms[nDevPerProc];
    int size = 32*1024*1024;
    int devs[nDevPerProc];

    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDevPerProc * sizeof(float*));
    float** recvbuff = (float**)malloc(nDevPerProc * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDevPerProc);

    // Initialize devices
    for (int i = 0; i < nDevPerProc; i++) {
        int globalGpuIdx = startGpu + i;
        devs[i] = globalGpuIdx;
        CUDA_CHECK(cudaSetDevice(globalGpuIdx));
        printf("Process %d initializing GPU %d\n", rank, globalGpuIdx);
   
        CUDA_CHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
        
        // Initialize with different values for each GPU
        float* hostbuff = (float*)malloc(size * sizeof(float));
        for(int j = 0; j < size; j++) hostbuff[j] = globalGpuIdx + 1.0f;
        CUDA_CHECK(cudaMemcpy(sendbuff[i], hostbuff, size * sizeof(float), cudaMemcpyHostToDevice));
        free(hostbuff);
        
        CUDA_CHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(s+i));
    }

    // Create NCCL unique ID and broadcast it
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Initialize NCCL communicators
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < nDevPerProc; i++) {
        NCCL_CHECK(ncclCommInitRank(&comms[i], totalGPUs, id, startGpu + i));
    }
    NCCL_CHECK(ncclGroupEnd());
    
    // Make sure all processes are ready before proceeding
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) printf("All processes initialized NCCL communicators\n");

    // AllReduce operation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));  // Synchronize before communication
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < nDevPerProc; i++) {
        NCCL_CHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], 
                                size, ncclFloat, ncclSum, comms[i], s[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // Make sure all processes complete their operations
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) printf("All processes completed AllReduce operations\n");

    // Synchronize and check results
    for (int i = 0; i < nDevPerProc; i++) {
        int globalGpuIdx = startGpu + i;
        CUDA_CHECK(cudaSetDevice(globalGpuIdx));
        CUDA_CHECK(cudaStreamSynchronize(s[i]));
        
        float* results = (float*)malloc(size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(results, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Process %d, GPU %d: First element = %f (should be %f)\n", 
               rank, globalGpuIdx, results[0], (float)(totalGPUs * (totalGPUs + 1) / 2));
        free(results);
    }

    // Cleanup
    for (int i = 0; i < nDevPerProc; i++) {
        CUDA_CHECK(cudaSetDevice(startGpu + i));
        CUDA_CHECK(cudaFree(sendbuff[i]));
        CUDA_CHECK(cudaFree(recvbuff[i]));
        CUDA_CHECK(cudaStreamDestroy(s[i]));
        ncclCommDestroy(comms[i]);
    }
    
    free(sendbuff);
    free(recvbuff);
    free(s);

    MPI_CHECK(MPI_Finalize());
    if (rank == 0) printf("All done!\n");
    return 0;
}