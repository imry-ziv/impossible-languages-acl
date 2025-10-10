import matplotlib.pyplot as plt
import pandas as pd
import re
import glob
import argparse
import os
import csv


def parse_err_file(file_name, is_single_language):
    label = extract_label(file_name, is_single_language)
    data = {'label': label, 'batch': [], 'val_ppl': []}
    pattern = re.compile(
        r'\| epoch\s+(\d+) \| (\d+)/(\d+) batches \| lr \d+\.\d+ \| ms/batch \d+\.\d+ \| loss\s+\d+\.\d+ \| ppl\s+\d+\.\d+ \| val loss\s+\d+\.\d+ \| val ppl\s+(\d+\.\d+)'
    )
    with open(file_name, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epoch_number = int(match.group(1))
                if epoch_number > 0:
                    break
                completed_batches = int(match.group(2))
                total_batches = int(match.group(3))
                validation_ppl = float(match.group(4))
                batches_processed_so_far = epoch_number * total_batches + completed_batches
                data['batch'].append(batches_processed_so_far)
                data['val_ppl'].append(validation_ppl)

    return data

def get_full_file_name(identifier:str, logs_dir="logs"):
    logs_path = os.path.abspath(logs_dir)
    # Iterate over all files in the logs directory
    for file_name in os.listdir(logs_path):
        if identifier in file_name and file_name.endswith(".err"):
            return os.path.join(logs_path, file_name)


def extract_label(file_name, is_single_language):
    # Adjusted regex to capture different possible formats
    match = re.search(r'retraining-il-([^_-]+)-([^_]+)(?:_\d+)?\.err', os.path.basename(file_name))
    if match:
        label = match.group(1).replace('-', ' ').title()
        perturbation_type = match.group(2).replace('-', ' ').title()
        if not is_single_language:
            return f"{label} {perturbation_type}"
        else:
            return perturbation_type
    return "Unknown"

def plot_comparisons(parsed_files_groups, language_pairs):
    num_plots = len(language_pairs)
    fig, axs = plt.subplots(1, num_plots, figsize=(18, 6), sharey=True, constrained_layout=True)  # Use constrained_layout for proper aspect ratio
    if num_plots == 1:
        axs = [axs]
    for idx, (parsed_files, (lang1, lang2)) in enumerate(zip(parsed_files_groups, language_pairs)):
        ax = axs[idx]

        min_length = min(len(parsed_file['batch']) for parsed_file in parsed_files)

        for parsed_file in parsed_files:
            label = parsed_file['label']
            batch = parsed_file['batch'][:min_length]  # Prune batch list to the shortest length
            val_ppl = parsed_file['val_ppl'][:min_length]  # Prune val_ppl list to the shortest length
            ax.plot(batch, val_ppl, label=label, marker='o', markersize=2)  # Smaller markers

        # Set axis labels
        if idx == 0:
            ax.set_ylabel('Validation Perplexity (lower is better)', fontsize=14, labelpad=5)

        # Set subplot title
        ax.set_title(f'{lang1.capitalize()} vs. {lang2.capitalize()}', fontsize=15)

        # Add a legend
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set common x-axis label
    fig.text(0.5, 0.04, 'Batch Number', ha='center', fontsize=14)

    plt.tight_layout(rect=[0.015, 0.05, 0.97, 1])  # Adjusted rect for equal padding on both sides
    plt.savefig("language_comparisons.png")
    plt.show()


def create_csv(identifier):
     full_file_name = get_full_file_name(identifier)
     parsed = parse_err_file(full_file_name, is_single_language=True)
     label = parsed["label"]
     with open(f"csvs/{identifier}.csv", mode="w", newline="") as file:
         writer = csv.writer(file)
         writer.writerow(["batch", "val_ppl", "label"])
         for batch, perplexity in zip(parsed["batch"], parsed["val_ppl"]):
             writer.writerow([batch, perplexity, label])


def create_csvs_for_all_perturbations():
    languages = (
        #"english",
        #"french",
        "hebrew",
        #"danish",
        #"finnish",
        #"greek",
        #"russian",
        #"italian",
    )
    perturbations = (
        "no-perturb",
        "no-hop",
        "token-hop",
        "switch-indices",
        "full-reverse",
        "partial-reverse",
    )
    for lang in languages:
        for pert in perturbations:
            identifier = f"{lang}-{pert}"
            try:
                create_csv(identifier)
            except:
                print(f"File {identifier} not found. Continuing.")
                continue



# Main function
if __name__ == '__main__':
    create_csvs_for_all_perturbations()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots_to_compare",
        type=str,
        nargs='+',
        required=True,
        help="A list of language-method identifiers (e.g., 'italian-full-reverse')"
    )
    parser.add_argument(
        "--language_pairs",
        type=str,
        nargs='+',
        required=True,
        help="Pairs of languages to compare (e.g., 'english-italian', 'italian-russian')"
    )
    parser.add_argument(
        '--all_to_csv',
        default=False,
        action='store_true',
    )
    args = parser.parse_args()

    # Parse language pairs
    language_pairs = [tuple(pair.split('-')) for pair in args.language_pairs]

    # Prepare parsed files for each pair
    parsed_files_groups = []
    for lang1, lang2 in language_pairs:
        group = []
        for identifier in args.plots_to_compare:
            if identifier.startswith(lang1) or identifier.startswith(lang2):
                full_file_name = get_full_file_name(identifier)
                parsed = parse_err_file(full_file_name, is_single_language=False)
                group.append(parsed)
        parsed_files_groups.append(group)

    if args.all_to_csv:
        string_repr_csv = "csvs/" + '_'.join(args.plots_to_compare) + ".csv"
        with open(string_repr_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["batch", "val_ppl", "label"])

            # Iterate through the nested list
            for group in parsed_files_groups:
                for entry in group:
                    for batch, val_ppl in zip(entry["batch"], entry["val_ppl"]):
                        writer.writerow([batch, val_ppl, entry["label"]])

    else:
        plot_comparisons(parsed_files_groups, language_pairs)
