#!/bin/bash
#SBATCH --job-name=perturb-cpu-2111
#SBATCH --output=/scratch2/nlan/impossible-languages-acl/logs_downsampled/perturb-cpu-2111-%j.output
#SBATCH --error=/scratch2/nlan/impossible-languages-acl/logs_downsampled/perturb-cpu-2111-%j.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --export=ALL
#SBATCH --time=48:00:00




source /home/nlan/.virtualenvs/jlm-llm/bin/activate
cd /scratch2/nlan/impossible-languages-acl
pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_lg/resolve/main/tr_core_news_lg-1.0-py3-none-any.whl
methods=("reverse-baseline" "no-hop" "full-reverse" "partial-reverse" "shuffle-global" "shuffle-local-2" "switch-indices" "token-hop")

languages=("turkish")

for method in "${methods[@]}"; do
  for lang in "${languages[@]}"; do
    echo "Running $method for language: $lang"
    python perturb.py --language "$lang" --method "$method"
  done
done

