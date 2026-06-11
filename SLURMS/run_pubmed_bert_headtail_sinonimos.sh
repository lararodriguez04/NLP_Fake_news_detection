#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/lnlpG08/nlp/RESULTS
#SBATCH -t 0-12:00
#SBATCH -p dcca40
#SBATCH --mem 32768
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err
#SBATCH --gres gpu:1

python3 /home/lnlpG08/nlp/train_pubmed_bert_headtail_sinonimos.py
