#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=resnet_finetuning
#SBATCH --mem=2G

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

python3 ResNet.py