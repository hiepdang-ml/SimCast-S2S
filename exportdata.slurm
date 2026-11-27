#!/bin/bash

#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --partition=standard
#SBATCH --mail-user=zgp2ps@virginia.edu
#SBATCH --mail-type=ALL
#SBATCH --account=mamalakislab
#SBATCH --output=./.slurmlogs/%j.out
#SBATCH --job-name="cesm2"

source $HOME/.bashrc &&
source $HOME/miniconda3/etc/profile.d/conda.sh &&
conda activate SimCast-S2S &&
export PYTHONPATH=$HOME/SimCast-S2S &&

python exportdata.py --dataset=cesm2 --fresh
