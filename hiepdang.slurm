#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:4
#SBATCH --partition=gpu
#SBATCH --mail-user=zgp2ps@virginia.edu
#SBATCH --mail-type=ALL
#SBATCH --account=mamalakislab
#SBATCH --output=./.slurmlogs/%j.out
#SBATCH -J "ddpm"

source $HOME/.bashrc &&
source $HOME/miniconda3/etc/profile.d/conda.sh &&
conda activate s2s-cesm2 &&
export PYTHONPATH=$HOME/s2s-cesm2 &&

# Force IPv4 localhost for single-node runs to avoid AF errors
export MASTER_ADDR=127.0.0.1 &&
export MASTER_PORT=29505 &&
torchrun --master_addr=${MASTER_ADDR} --nproc_per_node=${SLURM_GPUS_ON_NODE} --master_port=${MASTER_PORT} train.py --model=ddpm --dataset=cesm2
