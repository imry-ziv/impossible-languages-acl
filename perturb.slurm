#!/bin/bash
#SBATCH --job-name= # Add as required
#SBATCH --output= # Add as required
#SBATCH --error= # Add as required
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --export=ALL
#SBATCH --time=48:00:00


methods=("reverse-baseline" "no-hop" "full-reverse" "partial-reverse" "shuffle-global" "shuffle-local-2" "switch-indices" "token-hop")
languages=("danish" "english" "finnish" "french" "german" "greek" "hebrew" "italian" "russian")

for method in "${methods[@]}"; do
  for lang in "${languages[@]}"; do
    echo "Running $method for language: $lang"
    python impossible_languages/perturb.py --language "$lang" --method "$method"
  done
done

