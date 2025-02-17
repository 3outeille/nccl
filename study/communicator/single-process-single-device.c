/*
mpirun -np 4 ./single-process-single-device
*/

#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdlib.h>    // for exit() and EXIT_FAILURE
#include <mpi.h>
#include <stdint.h>  // for uint64_t

#define MPI_CHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

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

int main(int argc, char* argv[])
{  
  int size = 32*1024*1024;

  // Initialize MPI
  MPI_CHECK(MPI_Init(&argc, &argv));
  
  // Get my rank and world size
  int rank, nGpus;  // nGpus is same as worldSize
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nGpus));

  printf("Process %d using GPU %d of %d GPUs\n", rank, rank, nGpus);

  // Initialize device
  CUDA_CHECK(cudaSetDevice(rank));  // Use rank as GPU ID

  // Create NCCL unique ID and broadcast it
  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Initialize NCCL communicator
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, nGpus, id, rank));
  
  // Allocate device buffers
  float *sendbuff, *recvbuff;
  CUDA_CHECK(cudaMalloc((void**)&sendbuff, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&recvbuff, size * sizeof(float)));

  // Initialize data
  float *hostbuff = (float*)malloc(size * sizeof(float));
  for(int i=0; i<size; i++) hostbuff[i] = rank + 1.0f;  // Use rank instead of myGpuId
  CUDA_CHECK(cudaMemcpy(sendbuff, hostbuff, size * sizeof(float), cudaMemcpyHostToDevice));
  free(hostbuff);

  // Create CUDA stream
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // Perform AllReduce
  printf("Performing AllReduce on GPU %d (rank %d)\n", rank, rank);
  NCCL_CHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, 
                          comm, s));

  // Synchronize stream
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Check results
  float *results = (float*)malloc(size * sizeof(float));
  CUDA_CHECK(cudaMemcpy(results, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Results on GPU %d (rank %d): First element = %f (should be %f)\n", 
         rank, rank, results[0], (float)(nGpus * (nGpus + 1) / 2));
  free(results);

  // Cleanup
  CUDA_CHECK(cudaFree(sendbuff));
  CUDA_CHECK(cudaFree(recvbuff));
  CUDA_CHECK(cudaStreamDestroy(s));
  ncclCommDestroy(comm);

  MPI_CHECK(MPI_Finalize());
  printf("[GPU %d] Success \n", rank);
  return 0;
}