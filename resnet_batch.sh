#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=resnet_finetuning
#SBATCH --mem=4G

module purge
module load torchvision/0.13.1-foss-2022a

python3 ResNet.py