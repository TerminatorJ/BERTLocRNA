#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --job-name=training
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=8 --mem=100000M
#We expect that our program should not run longer than 3 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=3-00:00:00
#SBATCH --output=train_model_%j.out
#SBATCH --array=1  # Specify the range of job array tasks (corresponding to 10 layers)


~/miniconda3/envs/deeploc_torch/bin/python /home/sxr280/BERTLocRNA/scripts/train_model.py

