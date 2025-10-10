#!/bin/bash

#SBATCH --job-name=add-languages
#SBATCH --output=### # Add as needed
#SBATCH --error=###  # Add as needed
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --export=ALL
#SBATCH --time=48:00:00



python add_languages.py --language french --download_dumps

