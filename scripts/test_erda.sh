#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu --gres=gpu:1
#SBATCH --job-name=erda
#number of independent tasks we are going to start in this script
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=1 --mem=1000M
#We expect that our program should not run longer than 3 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=03:00:00
#SBATCH --output=test_erda_%j.out
sh ./mount_erda.sh
ls -al /tmp/erda
sh ./unmount_erda.sh