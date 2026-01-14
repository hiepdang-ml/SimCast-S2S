#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --mail-user=zgp2ps@virginia.edu
#SBATCH --mail-type=ALL
#SBATCH --account=mamalakislab
#SBATCH --output=./.slurmlogs/%j.out
#SBATCH --job-name="downloadcesm2"

source $HOME/.bashrc &&
source $HOME/miniconda3/etc/profile.d/conda.sh &&
conda activate s2s-cesm2 &&
export PYTHONPATH=$HOME/s2s-cesm2 &&

python cesm2download/download.py

