#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:2
#SBATCH --partition=gpu
#SBATCH --mail-user=zgp2ps@virginia.edu
#SBATCH --mail-type=ALL
#SBATCH --account=mamalakislab
#SBATCH --output=./.slurmlogs/%j.out
#SBATCH --job-name="diffusion"

source $HOME/.bashrc &&
source $HOME/miniconda3/etc/profile.d/conda.sh &&
conda activate s2s-cesm2 &&
export PYTHONPATH=$HOME/s2s-cesm2 &&

export NCCL_IB_DISABLE=1
export NCCL_COLLNET_ENABLE=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_LEVEL=SYS
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Force IPv4 localhost for single-node runs to avoid AF errors
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
# torchrun --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nproc_per_node=${SLURM_GPUS_ON_NODE} train.py --model=diffusion --dataset=cesm2
# torchrun --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nproc_per_node=${SLURM_GPUS_ON_NODE} train.py --model=diffusion --dataset=era5
torchrun --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nproc_per_node=${SLURM_GPUS_ON_NODE} train.py --model=diffusion --dataset=era5 --finetune
