# Impossible Languages: Data and Scripts

# What is this project?

1. **Research Question**  
Are LLMs sensitive to the distinction between humanly possible languages and humanly impossible languages?  
Contrary to previous results, we find that they are **not**.  

2. **Impossible Languages**  
Impossible languages are hypothetical languages that humans cannot acquire in principle.  
They violate universal properties of human language — constraints induced by the innate machinery humans are born with.
For example, all human languages are "hierarchical"

3. **Example for Impossibility**

All human languages are **hierarchical**, not just linear.  

- **Linear view:** Words are simply a chain, one after another.  
  Example:  
  ```
  The cat sat on the mat
  ```
  In a linear view, each word only relates to its neighbors.

- **Hierarchical view:** Words group into phrases, which group into larger phrases, like nested boxes.  
  Example:  
  ```
  [The cat] [sat [on [the mat]]]
  ```
  - `[The cat]` → noun phrase (NP)  
  - `[on [the mat]]` → prepositional phrase (PP)  
  - `[sat [on [the mat]]]` → verb phrase (VP)  

This hierarchy allows humans to create **complex sentences with embedded ideas**, something a purely linear system cannot capture.  
A language that does not rely on this hierarchical structure -  would be an impossible language.

4. **Innately Human Task**  
Distinguishing the possible from the impossible is thus an innately human task.  
If LLMs are able to make this distinction, it might indicate that their learning biases resemble human ones.

5. **Methodology**  
We test LLM sensitivity to impossibility using the methodology from Kallini et al. (2024) "Mission: Impossible Languages", extending it cross-linguistically.  
We apply perturbation functions to datasets from 9 languages, creating "impossible" counterparts for each attested language.

6. **GPT-2 Experiments**  
We test GPT-2's capacity to learn impossible datasets as easily as their attested, "possible" counterparts by comparing validation perplexities achieved during pretraining.
The validation perplexity curve across the first 3,000 training steps is taken to represent how "hard" it is for the model to learn the language.
The lower this curve, the easier the model learned the language. 
If GPT-2 is sensitive to the possible/impossible divide, it should 

7. **Findings**  
Across most cases, GPT-2 learns the impossible language just as easily as its possible counterpart — showing no trace of human-like inductive biases.  
We also checked whether GPT-2 separates the whole set of natural languages from impossible languages, arriving again at a negative answer. See our paper for details.

7. **Conclusion**  
LLMs are undeniably remarkable engineering devices, capable of performing convincingly on many linguistic benchmarks.  
However, their lack of sensitivity to linguistic impossibility suggests caution in treating them as cognitive models of human language learning.


## Repository Structure

```
impossible_languages/
│
├── perturb.py                 # Generates impossible-language datasets
├── gpt2_retraining.py         # Fine-tunes GPT-2 on the datasets
├── add_languages.py           # Creates datasets for new languages from Wikipedia dumps
├── draw_plots_typology.py     # Visualization of retraining results
│
├── data/
│   ├── wikidumps/             # Unmodified wikidump downloads, before tokenization
│   └── multilang/             # All datasets, with subfolders for each language
│
└── gpt2_logs/                      # Output logs from training and perturbations
```

---

## Setup

### 1. Environment

Install all required Python packages:

```bash
pip install -r requirements.txt
```


##  Running Perturbations

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
- _IMPLEMENTED_PERTURBATIONS provides names of all perturbations. 
- Note that baseline versions of perturbations also need to be run: e.g. no-hop and reverse-baseline, to be compared with hops and reverses respectively.

## Creating New Datasets for New Languages

add_languages.py generates existing datasets for all languages + multilingual tokenizing with TreeTagger.

- LANGUAGES_TO_DUMP_FILES (add_languages.py) provides mapping from language name to URL of relevant wikidump. 
- Note that we do not actively maintain the repository, so the dump links might become broken at some point. In such a case, you may refer to f"https://dumps.wikimedia.org/{language_prefix}wiki" where language_prefix = {'fr', 'en'...} 
- Running add_languages.py with --download_dumps performs the download from wikimedia, to "./data/wikidumps"
- Rest of code processes sentences (filtering <unk>s, tokenizing with TreeTagger...) and creates vocab files.
- Tokenized datasets are saved under "./data/multilang"
- We also used Gulordava's (2018) original four datasets for English, Italian, Hebrew, Russian. To use them as is, run dl_gulordava.sh
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
Note that tokenizer object is "vanilla": splits by whitespace, since we had previously tokenized with TreeTagger which tokenizes all languages in uniform fashion.

---

## 📊 Plotting Curves

Use the different plotting functions in `gpt2_retraining.py` to:
- Plot validation perplexity across languages and perturbations
- Compute within- and across-language variances
- Evaluate the model’s ability to distinguish possible vs. impossible languages
- `make_combined_plot_background()`creates a tile-view cross linguistic comparison of perplexity curves, with a heatmap denoting how far away the curves are.
- `compare_metrics_side_by_side()` creates a cross-linguistic comparison of minimal perplexity values / Area Under Curve by language, to test for within-language variance compared with across-language variance.

## 🪪 License

This repository is shared under the **MIT License**.  
Parts of the code and methodology are based on:
- [facebookresearch/colorlessgreenRNNs](https://github.com/facebookresearch/colorlessgreenRNNs)
- [jkallini/mission-impossible-language-models](https://github.com/jkallini/mission-impossible-language-models)
