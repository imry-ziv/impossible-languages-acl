# Impossible Languages: Data and Scripts

This repository contains code and data for reproducing the experiments in our paper  
**_Biasless Language Models Learn Unnaturally: How LLMs Fail to Distinguish the Possible from the Impossible_**.

---

## 📦 Repository Structure

```
impossible_languages/
│
├── perturb.py                 # Generates impossible-language datasets
├── gpt2_retraining.py         # Fine-tunes GPT-2 on the datasets
├── analysis/                  # Scripts for validation and visualization
│
├── data/
│   ├── original/              # Unmodified datasets
│   └── perturbed/             # Generated perturbed datasets
│
└── logs/                      # Output logs from training and perturbations
```

---

## ⚙️ Setup

### 1. Environment

```bash
source /home/nlan/.virtualenvs/jlm-llm/bin/activate
cd /scratch2/nlan/jlm-llm
```

### 2. Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🧪 Running Perturbations

The script `perturb.py` generates "impossible" versions of language datasets using various perturbation methods.

Available methods:

```
reverse-baseline
no-hop
full-reverse
partial-reverse
shuffle-global
shuffle-local-2
switch-indices
token-hop
```

To run perturbations for all languages and methods:

```bash
methods=("reverse-baseline" "no-hop" "full-reverse" "partial-reverse" "shuffle-global" "shuffle-local-2" "switch-indices" "token-hop")
languages=("danish" "english" "finnish" "french" "german" "greek" "hebrew" "italian" "russian")

for method in "${methods[@]}"; do
  for lang in "${languages[@]}"; do
    echo "Running $method for language: $lang"
    python impossible_languages/perturb.py --language "$lang" --method "$method"
  done
done
```

---

## 🔁 Fine-Tuning GPT-2

After generating the datasets, fine-tune GPT-2 with:

```bash
for language in "${languages[@]}"; do
  EXPERIMENT_IDENTIFIER="${language}-${method}"
  exec > ${LOG_DIR}/retraining-il-${EXPERIMENT_IDENTIFIER}.out 2> ${LOG_DIR}/retraining-il-${EXPERIMENT_IDENTIFIER}.err
  python impossible_languages/gpt2_retraining.py --language "$language" --method "$method"
done
```

This will redirect the standard output and error streams to log files for each run.

---

## 📊 Analysis

Use the scripts in `analysis/` to:
- Plot validation perplexity across languages and perturbations
- Compute within- and across-language variances
- Evaluate the model’s ability to distinguish possible vs. impossible languages

Example:

```bash
python analysis/plot_ppl.py --input_dir logs/ --output_dir plots/
```

---

## 📚 Citation

If you use this repository, please cite:

```
@article{lan2025biasless,
  title={Biasless Language Models Learn Unnaturally: How LLMs Fail to Distinguish the Possible from the Impossible},
  author={Lan, Nur and Ziv, Imry and Chemla, Emmanuel and Katzir, Roni},
  journal={arXiv preprint arXiv:2510.07178},
  year={2025}
}
```

---

## 🪪 License

This repository is shared under the **MIT License**.  
Parts of the code and methodology are based on:
- [facebookresearch/colorlessgreenRNNs](https://github.com/facebookresearch/colorlessgreenRNNs)
- [jkallini/mission-impossible-language-models](https://github.com/jkallini/mission-impossible-language-models)
