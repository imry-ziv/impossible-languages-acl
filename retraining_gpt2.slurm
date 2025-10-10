#!/bin/bash
#SBATCH --job-name= # Add as required
#SBATCH --error= # Add as required
#SBATCH --output= # Add as required
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

languages=("english" "italian" "russian" "danish" "finnish" "greek" "hebrew" "german" "french")
method="no-perturb"
LOG_DIR="./gpt2_logs"

for language in "${languages[@]}"; do
    EXPERIMENT_IDENTIFIER="${language}-${method}"
    exec > ${LOG_DIR}/retraining-il-${EXPERIMENT_IDENTIFIER}.out 2> ${LOG_DIR}/retraining-il-${EXPERIMENT_IDENTIFIER}.err
    python impossible_languages/gpt2_retraining.py --language "$language" --method "$method"
done
