#!/bin/bash
#SBATCH -J rba_sample
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=80:00:00
#SBATCH --mem=64GB
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o /users/jdtoscan/data/jdtoscan/RBA+Operators/wave_tcunet/Output/rba_sample-%j.out
#SBATCH -e /users/jdtoscan/data/jdtoscan/RBA+Operators/wave_tcunet/Error/rba_sample-%j.err

cd /users/jdtoscan/data/jdtoscan/RBA+Operators/wave_tcunet/rba_sample/|| exit

nvidia-smi
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate Pytorch_conda

rm -rf ./__pycache__/

python3 -u rba_sample.py 