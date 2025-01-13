#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=resnet_finetuning
#SBATCH --mem=2G

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.8.2

python3 loss_ResNet.py