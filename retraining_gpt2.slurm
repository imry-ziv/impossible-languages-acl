#!/bin/bash
#SBATCH --job-name=retraining_2011
#SBATCH --error=logs/retraining_2011-%j.err
#SBATCH --output=logs/retraining_2011-%j.out
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

languages=("english" "hebrew")
method="no-perturb"
LOG_DIR="./gpt2_logs"

mkdir -p "$LOG_DIR"

for language in "${languages[@]}"; do
    EXPERIMENT_IDENTIFIER="${language}-${method}"
    python gpt2_retraining.py --language "$language" --method "$method" \
        > "${LOG_DIR}/retraining-il-${EXPERIMENT_IDENTIFIER}.out" \
        2> "${LOG_DIR}/retraining-il-${EXPERIMENT_IDENTIFIER}.err"
done
