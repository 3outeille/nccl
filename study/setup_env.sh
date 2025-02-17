#!/bin/bash

# vim /usr/share/modules/modulefiles/cuda -> then replace with export. Dont forget to `source setup_env.sh` in .bashrc

# CUDA paths
export PATH=/usr/local/cuda/bin:${PATH:-}
export CPATH=/usr/local/cuda/targets/x86_64-linux/include:/usr/local/cuda/include
export MANPATH=/usr/local/cuda/share/man
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/extras/CUPTI/lib64
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/extras/CUPTI/lib64
export CUDA_HOME=/usr/local/cuda

# NCCL paths
export CPATH=/fsx/ferdinandmom/ferdinand-hf/nccl/build/include:${CPATH:-}
export LD_LIBRARY_PATH=/fsx/ferdinandmom/ferdinand-hf/nccl/build/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/fsx/ferdinandmom/ferdinand-hf/nccl/build/lib:${LIBRARY_PATH:-}
export NCCL_HOME=/fsx/ferdinandmom/ferdinand-hf/nccl/build

# AWS EFA and OFI paths
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/opt/aws-ofi-nccl/lib:${LIBRARY_PATH:-}
export AWS_OFI_NCCL_HOME=/opt/aws-ofi-nccl:${AWS_OFI_NCCL_HOME:-}

# EFA settings
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export FI_EFA_ENABLE_SHM_TRANSFER=1
export NCCL_PROTO=simple
export NCCL_SOCKET_IFNAME=enp

# MPI paths
export MPI_PATH=/opt/amazon/openmpi
export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/efa/test-cuda:${PATH:-}

# Add an echo to test
echo "CPATH is: $CPATH"
echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"
echo "LIBRARY_PATH is: $LIBRARY_PATH"
echo "CUDA_HOME is: $CUDA_HOME"
echo "NCCL_HOME is: $NCCL_HOME"