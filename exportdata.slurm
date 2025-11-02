#!/bin/bash

#SBATCH --time=20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=gpu
#SBATCH --mail-user=zgp2ps@virginia.edu
#SBATCH --mail-type=ALL
#SBATCH --account=mamalakislab_paid
#SBATCH --output=./.slurmlogs/%j.out
#SBATCH --job-name="cesm2"

source $HOME/.bashrc &&
source $HOME/miniconda3/etc/profile.d/conda.sh &&
conda activate s2s-cesm2 &&
export PYTHONPATH=$HOME/s2s-cesm2 &&

python exportdata.py --dataset=cesm2 --resume
