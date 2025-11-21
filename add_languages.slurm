#!/bin/bash
#SBATCH --job-name=add-language-turkish
#SBATCH --error=/scratch2/nlan/jlm-llm/logs/add-language-turkish_%j.err
#SBATCH --output=/scratch2/nlan/jlm-llm/logs/add-language-turkish_%j.out
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --gres=gpu:1          # Ask for 1 GPU
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --mem=32G


source /home/nlan/.virtualenvs/jlm-llm/bin/activate
module load cuda/11.8.0-r465
nvidia-smi

cd /scratch2/nlan/jlm-llm/
export CUDA=1


python add_languages.py --language turkish --download_dumps
