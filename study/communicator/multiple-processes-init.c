#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

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
    char hostname[1024];
    gethostname(hostname, 1024);

    // Get rank and number of ranks from command line
    if (argc != 3) {
        printf("Usage: %s <rank> <nranks>\n", argv[0]);
        exit(0);
    }
    int rank = atoi(argv[1]);
    int nranks = atoi(argv[2]);

    // Generating NCCL unique ID at rank 0 and sharing it with other ranks
    ncclUniqueId id;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
        // Write the ID to a file
        FILE* fp = fopen("/tmp/nccl_id", "wb");
        if (fp == NULL) {
            printf("Failed to open file for writing\n");
            exit(EXIT_FAILURE);
        }
        size_t written = fwrite(&id, sizeof(id), 1, fp);
        if (written != 1) {
            printf("Failed to write NCCL ID to file\n");
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        fclose(fp);
    }
    
    // Make sure rank 0 has written the file
    sleep(1);
    
    // Other ranks read the ID
    if (rank != 0) {
        FILE* fp = fopen("/tmp/nccl_id", "rb");
        if (fp == NULL) {
            printf("Failed to open file for reading\n");
            exit(EXIT_FAILURE);
        }
        size_t read = fread(&id, sizeof(id), 1, fp);
        if (read != 1) {
            printf("Failed to read NCCL ID from file\n");
            fclose(fp);
            exit(EXIT_FAILURE);
        }
        fclose(fp);
    }

    // Initialize NCCL communicator
    ncclComm_t comm;
    CUDA_CHECK(cudaSetDevice(rank));
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));
    printf("Rank %d initialized NCCL on %s\n", rank, hostname);

    // Clean up
    ncclCommDestroy(comm);
    
    // Rank 0 removes the temporary file
    if (rank == 0) {
        remove("/tmp/nccl_id");
    }
    
    return 0;
} 