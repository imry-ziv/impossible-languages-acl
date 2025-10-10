# Codebase for "Biasless Language Models Learn Unnaturally".

The following code allows to reproduce our experiments, which were made after Kallini et al.'s (2024) methodology.
The add_languages.py functionality allows to add additional languages based on existing wikipedia dumps, and consider them against their perturbed, impossible versions.

Reproduction walkthrough:
1. Generate existing datasets for all languages + multilingual tokenizing with TreeTagger
   - LANGUAGES_TO_DUMP_FILES (add_languages.py) provides mapping from language name to URL of relevant wikidump.
   - Note that we do not actively maintain the repository, so the dump links might become broken at some point. In such a case, you may refer to f"https://dumps.wikimedia.org/{language_prefix}wiki" where language_prefix = {'fr', 'en'...}
   - Running add_languages.py with --download_dumps performs the download from wikimedia, to "./data/wikidumps".
   - Rest of code processes sentences (filtering <unk>s, tokenizing with TreeTagger...) and creates vocab files.
   - Tokenized datasets are saved under "./data/multilang"
   - We also used Gulordava's (2018) original four datasets for English, Italian, Hebrew, Russian. To use them as is, run dl_gulordava.sh

2. Generate perturbed datasets based on existing datasets
   - Relevant code is in perturb.py
   - _IMPLEMENTED_PERTURBATIONS provides names of all perturbations, we only use a subset in the paper.
   - Note that baseline versions of perturbations also need to be run: e.g. no-hop and reverse-baseline, to be compared with hops and reverses respectively
   - Pass language name with --language and perturbation name with --method.
   - See perturb.slurm for an example run of all perturbations + languages we had used in the paper.

3. Run perplexity experiments
   - See gpt2_retraining.py script.
   - Note that tokenizer object is "vanilla": splits by whitespace, since we had previously tokenized with TreeTagger which tokenizes all languages in uniform fashion
   - Script is provided with "language" + "method" arguments like before, and trains GPT-2 on relevant dataset.
   - See retraining_gpt2.slurm for example run for all languages and perturbations - note redirection of stdout and stderr to the log files in specified location.
   - Log files will then be read for validation perplexity values.

4. Draw perplexity curves
   - See plot.py.
   - Script first creates CSVs from perplexity log files.
   - parse_file() then runs on the CSVs.
   - make_combined_plot_background(df) creates a figure like our Figure 2.
   - compare_metrics_side_by_side(df) creates a figure like our Figure 3.
