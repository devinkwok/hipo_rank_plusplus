#!/bin/bash
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=hpp-%j.out
#SBATCH --error=hpp-%j.err

SRC_DIR=$HOME/proj/hiporank_plusplus

# load modulels
module load python/3.7
module load python/3.7/cuda/10.1/cudnn/8.0/pytorch/1.6.0

# set up python environment
virtualenv $SLURM_TMPDIR/env/
source $SLURM_TMPDIR/env/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r $SRC_DIR/requirements.txt
pyrouge_set_rouge_path $SRC_DIR/ROUGE-1.5.5/

python $SRC_DIR/hpp_test.py
# python $SRC_DIR/hpp_baseline.py
# python $SRC_DIR/hpp_clustering.py
