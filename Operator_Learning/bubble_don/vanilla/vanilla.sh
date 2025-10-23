#!/bin/bash
#SBATCH -J vanilla
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=80:00:00
#SBATCH --mem=64GB
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o /users/jdtoscan/data/jdtoscan/RBA+Operators/bubble_don/Output/vanilla-%j.out
#SBATCH -e /users/jdtoscan/data/jdtoscan/RBA+Operators/bubble_don/Error/vanilla-%j.err

cd /users/jdtoscan/data/jdtoscan/RBA+Operators/bubble_don/vanilla/|| exit

nvidia-smi
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate Pytorch_conda

rm -rf ./__pycache__/

python3 -u train_don.py 