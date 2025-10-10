import pandas as pd
import glob
import os

LANGUAGES = ["english", "italian", "russian", "hebrew", "french", "danish", "finnish", "greek", "german"]
TRANSFORMATIONS = ["shuffle-global", "shuffle-local-2", "partial-reverse", "full-reverse", "switch-indices",
                   "token-hop"]
BASELINE = ["no-perturb", "no-hop", "reverse-baseline"]
PERTURBATIONS = TRANSFORMATIONS + BASELINE
import pandas as pd


def generate_latex_tables(df, languages, perturbations, output_file="valppl_tables.tex"):
    """
    Generate LaTeX tables for each language in the same style as the example.

    Parameters:
        df: pandas.DataFrame with columns ['step', 'val_ppl', 'language', 'perturbation']
        languages: list of languages
        perturbations: list of perturbations
        output_file: path to write LaTeX code
    """

    with open(output_file, "w") as f:
        for lang in languages:
            lang_df = df[df['language'] == lang].pivot(index='step', columns='perturbation', values='val_ppl')
            lang_df = lang_df[perturbations]  # ensure correct column order

            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write("\\small\n")
            f.write(f"\\caption{{Validation Perplexity for {lang.capitalize()} per Step}}\n")
            f.write(f"\\label{{tab:valppl_{lang}}}\n")
            f.write("\\rowcolors{2}{gray!10}{white}\n")
            f.write("\\resizebox{\\textwidth}{!}{\n")
            f.write("\\begin{tabular}{|" + "|".join(["p{1.5cm}"] * (len(perturbations) + 1)) + "|}\n")
            f.write("\\toprule\n")

            # header row
            f.write(" step & " + " & ".join(perturbations) + " \\\\\n")
            f.write("\\midrule\n")

            # table rows
            for step, row in lang_df.iterrows():
                row_vals = " & ".join(f"{val:.2f}" for val in row.values)
                f.write(f"{step} & {row_vals} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("}\n")  # end resizebox
            f.write("\\end{table*}\n\n")


if __name__ == '__main__':
    # Path to your CSV files
    csv_folder = "./csvs"  # replace with your folder

    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))


    # Initialize list to collect DataFrames
    all_dfs = []

    for file in csv_files:
        # Extract filename without extension
        fname = os.path.basename(file).replace(".csv", "")

        # Parse language and perturbation from filename
        try:
            lang, perturb = fname.split("-", 1)
        except ValueError:
            print(f"Skipping {file}, cannot parse language/perturbation.")
            continue

        if lang not in LANGUAGES or perturb not in PERTURBATIONS:
            print(f"Skipping {file}, unknown language or perturbation.")
            continue

        # Read CSV
        df = pd.read_csv(file)

        # Add columns for language and perturbation
        df['language'] = lang
        df['perturbation'] = perturb

        # Keep only relevant columns
        df = df[['step', 'val_ppl', 'language', 'perturbation']]

        all_dfs.append(df)

    # Concatenate all DataFrames
    big_df = pd.concat(all_dfs, ignore_index=True)
    LANGUAGES = ["english", "italian", "russian", "hebrew", "french", "danish", "finnish", "greek", "german"]
    PERTURBATIONS = ["no-perturb", "no-hop", "reverse-baseline", "shuffle-global", "shuffle-local-2",
                     "partial-reverse", "full-reverse", "switch-indices", "token-hop"]
    generate_latex_tables(df, LANGUAGES, PERTURBATIONS)


