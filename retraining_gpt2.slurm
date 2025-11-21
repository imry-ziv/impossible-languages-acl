#!/bin/bash
#SBATCH --job-name=gpt2-retraining-multi-lang-no-perturb
#SBATCH --error=/scratch2/nlan/impossible-languages-acl/logs_downsampled/gpt2-retraining-no-perturb_%j.err
#SBATCH --output=/scratch2/nlan/impossible-languages-acl/logs_downsampled/gpt2-retraining-no-perturb_%j.out
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --gres=gpu:1          # Ask for 1 GPU
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --mem=32G

source /home/nlan/.virtualenvs/jlm-llm/bin/activate
module load cuda/11.8.0-r465
nvidia-smi

cd /scratch2/nlan/impossible-languages-acl/
export CUDA=1

languages=("english" "hebrew")
method="no-perturb"
LOG_DIR="/scratch2/nlan/impossible-languages-acl/logs_downsampled"

for language in "${languages[@]}"; do
    EXPERIMENT_IDENTIFIER="${language}-${method}"
    exec > ${LOG_DIR}/retraining-il-${EXPERIMENT_IDENTIFIER}.out 2> ${LOG_DIR}/retraining-il-${EXPERIMENT_IDENTIFIER}.err

    python gpt2_retraining.py --language "$language" --method "$method"
done
