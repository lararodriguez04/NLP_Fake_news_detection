#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/lnlpG08/nlp/RESULTS
#SBATCH -t 0-02:00
#SBATCH -p dcca40
#SBATCH --mem 32768
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err

python3 /home/lnlpG08/nlp/train_random_augmentation_data.py
