#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=32g
#SBATCH -p 3090-gcondo 
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpu2252

#SBATCH -J train_don
#SBATCH -o train_don.out
#SBATCH -e train_don.err
## get tunneling info

module load cuda

source ~/apps/torch_env/bin/activate

python3 -u train_don.py
