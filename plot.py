import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import Normalize

LANGUAGES = ["english", "italian", "russian", "hebrew", "french", "danish", "finnish", "greek", "german"]
TRANSFORMATIONS = ["shuffle-global", "shuffle-local-2", "partial-reverse", "full-reverse", "switch-indices", "token-hop"]
BASELINE = ["no-perturb", "no-hop", "reverse-baseline"]

TRANSFORMATION_MAPPING = {
    "full-reverse": "REVERSE full",
    "partial-reverse": "REVERSE partial",
    "shuffle-global": "SHUFFLE global",
    "shuffle-local-2": "SHUFFLE local",
    "switch-indices": "SWITCH",
    "token-hop": "HOP"
}
REVERSE_TRANSFORMATION_MAPPING = {v: k for k, v in TRANSFORMATION_MAPPING.items()}

def create_all_csvs():
    all_perts = TRANSFORMATIONS + BASELINE
    for lang in LANGUAGES:
        for pert in all_perts:
            identifier = f"{lang}-{pert}"
            #try:
            create_csv(identifier)
            # except:
            #     print(f"File {identifier} not found. Continuing.")
            #     continue


def generate_csv_combos():
    names = []
    for language in LANGUAGES:
        lc_language = language.lower()
        for pb in TRANSFORMATIONS:
            filename = f"csvs/{lc_language}-{pb}.csv"
            names.append(filename)
        for bl in BASELINE:
            filename = f"csvs/{lc_language}-{bl}.csv"
            names.append(filename)
    return names


def neutral_string(s):
    res = s.lower()
    res = res.replace("-", " ")
    res = res.replace("_", " ")
    return res


def parse_feature(filename, features):
    for feature in features:
        if neutral_string(feature) in neutral_string(filename):
            return feature
    return None


def mark_condition(label):
    res = parse_feature(label, BASELINE)
    if res is None:
        return "modified"
    return "original"


def parse_file(filename):
    if filename == 'csvs/german-no-perturb.csv':
        print('a')
    try:
        # Read the current CSV file
        df = pd.read_csv(filename)
    except:
        print(f'File {filename} does not exist.')
        return pd.DataFrame()

    # Extract base filename and transformation details
    basename = filename.split("/")[-1].removesuffix(".csv")  # Remove '.csv'
    language, transformation = basename.split("-", 1)  # Split at the first '-'
    if transformation in BASELINE:
        return pd.DataFrame() # Skip
    # Mark the condition based on the transformation
    condition = "original" if transformation in BASELINE else "modified"

    # Add columns for transformation, language, and condition
    df["transformation"] = transformation
    df["language"] = language.capitalize()
    df["condition"] = condition

    # Find the corresponding original condition file for this pairing (language, transformation)
    if 'hop' in transformation:
        original_condition = "no-hop"
    elif 'reverse' in transformation:
        original_condition = "reverse-baseline"
    else:
        original_condition = "no-perturb"
    original_filename = f"csvs/{language.lower()}-{original_condition}.csv"

    try:
        # Read the original condition file (if it exists)
        original_df = pd.read_csv(original_filename)
        original_df["transformation"] = transformation
        original_df["language"] = language.capitalize()
        original_df["condition"] = "original"  # Mark this as the original condition
    except:
        print(f'Original condition file {original_filename} does not exist.')
        original_df = pd.DataFrame()  # In case the original file doesn't exist

    # If the original data exists, concatenate it with the current data
    if not original_df.empty:
        df = pd.concat([df, original_df], ignore_index=True)
    return df

def make_plot(df):
    hue_order = ["original", "modified"]
    color_palette = {"original": "red", "modified": "blue"}  # Switched colors
    g = sns.FacetGrid(
        df,
        col="language",
        row="transformation",
        hue="condition",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=1.3,
        aspect=1,
        hue_order=hue_order,  # Specify the order of legend items
    )

    g.map(sns.lineplot, "step", "val_ppl", palette=color_palette)

    for ax in g.axes.flat:
        ax.set_xticks(range(0, 3001, 1000))  # X-ticks every 100
        ax.set_yticks(range(500,2001, 500))  # Y-ticks every 100
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Whole numbers
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Whole numbers
        ax.tick_params(axis="both", labelsize=6)  # Smaller tick font size
    g.set_axis_labels("", "")
    g.figure.text(0.5, 0.04, "Training Batches", ha="center", va="center", fontsize=16)
    g.figure.text(
        0.04,
        0.5,
        "Validation Set Perplexity",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=16,
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend()
    for ax in g.axes_dict.values():
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=13)  # Larger font for column titles (languages)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)  # Smaller font for row titles (transformations)
    plt.savefig("main_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def get_min_value(df):
    res = (
        df.groupby(["transformation", "language", "condition"])["val_ppl"]
        .min()
        .reset_index()
        .rename(columns={"val_ppl": "min_value"})
    )
    return res


def get_latest_value(df):
    df_sorted = df.sort_values(by="step", ascending=False)
    res = (
        df_sorted.groupby(["transformation", "language", "condition"])
        .first()
        .reset_index()
    )
    return res[["transformation", "language", "condition", "val_ppl"]].rename(
        columns={"val_ppl": "last_value"}
    )


def get_AUC(df):
    # Ensure required columns exist
    required_columns = ["transformation", "language", "condition", "step", "val_ppl"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Step 1: Identify the common batch range across all groups
    batch_sets = df.groupby(["transformation", "language", "condition"])["step"].apply(
        set
    )
    shared_batches = set.intersection(*batch_sets)

    # Restrict the DataFrame to only the shared batch values
    df_filtered = df[df["step"].isin(shared_batches)]

    # Step 2: Sort and remove duplicates
    df_sorted = df_filtered.sort_values(by=["transformation", "step", "val_ppl"])
    df_sorted = df_sorted.drop_duplicates(
        subset=["transformation", "language", "condition", "step"]
    )

    # Step 3: Compute the AUC for each group
    auc_results = []

    for (transformation, language, condition), group in df_sorted.groupby(
        ["transformation", "language", "condition"]
    ):
        group = group.sort_values(by="step")

        # Compute the AUC using the trapezoidal rule
        auc = np.trapz(group["val_ppl"], group["step"])

        auc_results.append(
            {
                "transformation": transformation,
                "language": language,
                "condition": condition,
                "AUC": auc,
            }
        )

    # Step 4: Return the AUC as a DataFrame
    auc_df = pd.DataFrame(auc_results)

    # Scale by millions
    auc_df['AUC'] /= 1e6  # Scale AUC values to millions
    return auc_df

def get_relative_ME(df):
    # Ensure required columns exist
    required_columns = ["step", "val_ppl", "label", "transformation", "language", "condition"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    me_results = []

    # Step 1: Iterate over each unique combination of transformation and language
    for (transformation, language), group in df.groupby(["transformation", "language"]):
        # Step 2: Separate baseline and modified conditions
        baseline_group = group[group["condition"] == "original"]
        modified_group = group[group["condition"] == "modified"]

        # Ensure both baseline and modified have the same steps
        common_steps = set(baseline_group["step"]).intersection(modified_group["step"])
        baseline_group = baseline_group[baseline_group["step"].isin(common_steps)]
        modified_group = modified_group[modified_group["step"].isin(common_steps)]

        if len(baseline_group) == 0 or len(modified_group) == 0:
            continue  # Skip if no common steps

        # Sort both groups by 'step'
        # baseline_group = baseline_group.sort_values(by="step")
        # modified_group = modified_group.sort_values(by="step")

        # Compute ME between baseline and modified conditions
        me = np.mean(baseline_group["val_ppl"].values - modified_group["val_ppl"].values)
        #
        # Compute the variance of the baseline condition for relative MSE
        baseline_variance = np.var(baseline_group["val_ppl"])

        # Compute relative MSE (normalized by the variance of the baseline)
        relative_me = me / baseline_variance if baseline_variance != 0 else np.nan

        me_results.append(
            {
                "transformation": transformation,
                "language": language,
                "relative_me": me,
            }
        )

    # Step 3: Return the relative ME as a DataFrame
    me_df = pd.DataFrame(me_results)
    return me_df


def get_summaries(df):
    print(df.head())
    min_df = get_min_value(df)
    latest_df = get_latest_value(df)
    auc_df = get_AUC(df)
    return auc_df
    # combined_df = min_df.merge(
    #     latest_df, on=["transformation", "language", "condition"], how="inner"
    # ).merge(auc_df, on=["transformation", "language", "condition"], how="inner")
    # return combined_df

def create_csv(identifier):
     full_file_name = get_full_file_name(identifier)
     parsed = parse_err_file(full_file_name, is_single_language=True)
     label = parsed["label"]
     with open(f"csvs/{identifier}.csv", mode="w", newline="") as file:
         writer = csv.writer(file)
         writer.writerow(["step", "val_ppl", "label"])
         for step, perplexity in zip(parsed["step"], parsed["val_ppl"]):
             writer.writerow([step, perplexity, label])


def parse_err_file(file_name, is_single_language):
    label = extract_label(file_name, is_single_language)
    data = {'label': label, 'step': [], 'val_ppl': []}

    pattern = re.compile(
        r'.*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \| DEBUG\s+\| .*?:compute_perplexity:\d+ - Step (\d+): Perplexity = (\d+\.\d+)'
    )

    with open(file_name, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                batch_number = int(match.group(1))
                validation_ppl = float(match.group(2))
                data['step'].append(batch_number)
                data['val_ppl'].append(validation_ppl)

    return data


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

def get_full_file_name(identifier:str, logs_dir="gpt2_logs"):
    logs_path = os.path.abspath(logs_dir)
    # Iterate over all files in the logs directory
    for file_name in os.listdir(logs_path):
        if identifier in file_name and file_name.endswith(".err"):
            return os.path.join(logs_path, file_name)


def scatter_plot_min_and_auc(df, metric_name):

    min_df = get_min_value(df)
    auc_df = get_AUC(df)
    min_df = min_df.drop_duplicates(subset=['transformation', 'language', 'condition'])
    auc_df = auc_df.drop_duplicates(subset=['transformation', 'language', 'condition'])
    merged_df = min_df.merge(auc_df[['transformation', 'language', 'condition', 'AUC']],
                            on=['transformation', 'language', 'condition'],
                            how='inner')

    color_map = {'original': 'blue', 'modified': 'orange'}
    merged_df['color'] = merged_df['condition'].map(color_map)

    # Add jitter to avoid overlap on AUC axis
    merged_df['auc_jitter'] = merged_df['AUC'] + np.random.uniform(-0.005, 0.005, len(merged_df))

    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['auc_jitter'], merged_df[metric_name], c=merged_df['color'], alpha=0.1)

    plt.xlabel("AUC", fontsize=12)
    plt.ylabel("Min Value", fontsize=12)
    plt.title("Min Value vs. AUC for Original vs Modified Conditions", fontsize=14)

    # Add legend
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    plt.legend(handles=legend_patches, title="Condition")

    plt.tight_layout()
    plt.show()


def one_dimensional_scatter(df, metric_name):
    if metric_name == 'min_value':
        min_df = get_min_value(df)
        label = "Min Value"
    elif metric_name == 'AUC':
        min_df = get_AUC(df)
        label = "AUC Value"
    else:
        return
    min_df = min_df.drop_duplicates(subset=['transformation', 'language', 'condition'])

    color_map = {'original': 'blue', 'modified': 'orange'}
    min_df['color'] = min_df['condition'].map(color_map)

    # Add vertical jitter to prevent perfect overlap
    jitter = np.random.uniform(-0.05, 0.05, len(min_df))

    plt.figure(figsize=(10, 2))  # Wide and short figure
    plt.scatter(min_df[metric_name], jitter, c=min_df['color'], alpha=0.7)

    plt.xlabel(label, fontsize=12)
    plt.yticks([])  # Remove y-axis
    plt.title(f"1D Scatter Plot of {label}s", fontsize=14)

    # Add legend
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    plt.legend(handles=legend_patches, title="Condition", loc="upper right")

    plt.tight_layout()
    plt.show()


def heatmap_relative_mse(df):
    me_df = get_relative_ME(df)
    df = me_df.pivot(index="transformation", columns="language", values="relative_me").reset_index()
    df.set_index("transformation", inplace=True)

    # Define a diverging colormap centered at zero
    plt.figure(figsize=(13.6, 8.5))
    # vmin was -0.037, 0.037
    sns.heatmap(df, cmap="RdBu_r", annot=True, fmt=".3f", linewidths=0.5, vmin=df.min().min(),  vmax=df.max().max(), center=0)

    plt.yticks(rotation=90, fontsize=10)
    plt.xticks(fontsize=10, rotation=0)
    plt.title("Relative Mean Error Across Perturbations and Languages", fontsize=15)
    plt.xlabel("Language",fontsize=14)
    plt.ylabel("Perturbation Type", fontsize=14)

    plt.savefig("relative_mse_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


def compare_variances_for_metric(df, metric_name, ax=None):
    if metric_name == 'min_value':
        df = get_min_value(df)
        col_name = 'min_value'
    elif metric_name == 'AUC':
        df = get_AUC(df)
        col_name = 'AUC'

    original_metric = df[df['condition'] == 'original'].groupby('language')[col_name].mean()
    var_lang = np.var(original_metric, ddof=1)  # Sample variance

    var_within = df.groupby('language')[col_name].var().mean()
    ratio = var_lang / var_within
    #
    # print(f"Variance across languages: {var_lang:.2e}")
    # print(f"Average within-language variance: {var_within:.2e}")
    # print(f"Ratio (cross-language variance / within-language variance): {var_lang / var_within:.2f}")

    # Instead of printing
    stats_text = (
        f"Var(across-language): {var_lang:.2f}\n"
        f"Var(within-language): {var_within:.2f}\n"
        f"Ratio: {ratio:.2f}"
    )

    # Use the provided axis if available, otherwise create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 7))

    else:
        fig = ax.figure  # grab the figure that this axis belongs to

    # Ensure all transformations are mapped
    transformation_shapes = {
        'full-reverse': (5,1), 'partial-reverse': (8,1), 'reverse-baseline': (5,2),
        'token-hop': 'd', 'no-hop': 'D', 'shuffle-local-2': '>', 'shuffle-global': '<',
        'switch-indices': 'v', 'no-perturb': '^'
    }

    transformations = df['transformation'].unique()
    for t in transformations:
        if t not in transformation_shapes:
            transformation_shapes[t] = 'X'

    jitter_offset = 0.1
    y_jitter_strength = 0.0 * (df[col_name].max() - df[col_name].min())

    unique_languages = list(df['language'].unique())
    lang_positions = {lang: i for i, lang in enumerate(unique_languages)}

    for language in df['language'].unique():
        lang_data = df[df['language'] == language]
        base_x = lang_positions[language]
        plotted_points_for_language = set()

        for transformation in lang_data['transformation'].unique():
            if 'reverse' in transformation:
                corr_baseline = 'reverse-baseline'
            elif 'hop' in transformation:
                corr_baseline = 'no-hop'
            else:
                corr_baseline = 'no-perturb'

            trans_data = lang_data[lang_data['transformation'] == transformation]
            if len(trans_data) == 2:
                orig_val = trans_data[trans_data['condition'] == 'original'][col_name].values[0]
                mod_val = trans_data[trans_data['condition'] == 'modified'][col_name].values[0]
                orig_marker = transformation_shapes[corr_baseline]
                mod_marker = transformation_shapes[transformation]

                orig_x = base_x - jitter_offset
                mod_x = base_x + jitter_offset

                if (orig_x, orig_val) not in plotted_points_for_language:
                    ax.scatter(orig_x, orig_val, color='blue', marker=orig_marker, label="Original" if language == unique_languages[0] else "")
                    plotted_points_for_language.add((orig_x, orig_val))

                ax.scatter(mod_x, mod_val, color='orange', marker=mod_marker, label="Modified" if language == unique_languages[0] else "")

    ax.set_xticks(range(len(unique_languages)))
    ax.set_xticklabels(unique_languages, rotation=45, fontsize=14)

    ylabel = "Minimal Perplexity" if col_name == 'min_value' else "AUC"
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(f"{ylabel} Values by Language", fontsize=15)

    # Create legends
    shape_legend_handles = [plt.Line2D([0], [0], marker=transformation_shapes[t], color='w', markerfacecolor='black', markersize=8, label=t)
                            for t in transformations]
    color_legend_handles = [
        mpatches.Patch(color='blue', label='Original'),
        mpatches.Patch(color='orange', label='Modified')
    ]

    # Add transformation legend first
    shape_legend = ax.legend(handles=shape_legend_handles, loc="upper left", bbox_to_anchor=(0, 1), fontsize=11)
    ax.add_artist(shape_legend)  # Keep it when adding the next legend

    # Add condition legend (colors)
    ax.legend(handles=color_legend_handles, loc="upper right", bbox_to_anchor=(0.51,1), fontsize=11)

    if ax is None:
        plt.show()
    return stats_text
def compare_metrics_side_by_side(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))  # Two subplots side by side

    text_min_val = compare_variances_for_metric(df, 'min_value', ax=axes[0])
    text_auc = compare_variances_for_metric(df, 'AUC', ax=axes[1])

    axes[0].set_title("Minimal Perplexity", fontsize=15)
    axes[1].set_title("AUC", fontsize=15)
    fig.text(
        0.25, 0.075, text_min_val,
        ha='center', va='center',
        fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
    )

    fig.text(
        0.75, 0.075, text_auc,
        ha='center', va='center',
        fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
    )

    #fig.suptitle("Cross-Linguistic Comparison of Minimal Perplexity Value and AUC", fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0.01, 0.1, 1, 0.96])  # Adjust layout to fit the title
    plt.savefig("cross_linguistic_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()


def make_combined_plot(df):
    # Compute relative ME values
    me_df = get_relative_ME(df)
    heatmap_data = me_df.pivot(index="transformation", columns="language", values="relative_me")

    # Normalize color scale for heatmap
    vmin = heatmap_data.min().min()
    vmax = heatmap_data.max().max()
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)  # Adjust vmin, vmax based on data
    cmap = plt.get_cmap("PiYG")

    # Create FacetGrid for line plots
    df['transformation'] = df['transformation'].replace(TRANSFORMATION_MAPPING)
    hue_order = ["original", "modified"]
    color_palette = {"original": "red", "modified": "blue"}  # Switched colors
    # Get sorted list of languages
    sorted_languages = sorted(df["language"].unique())

    # Convert "language" to a categorical variable with explicit ordering
    df["language"] = pd.Categorical(df["language"], categories=sorted_languages, ordered=True)
    print(df["language"].unique())  # Check unique language values
    print(df["language"].dtype)  # Check data type
    # Create FacetGrid with explicit column order
    g = sns.FacetGrid(
        df,
        col="language",
        col_order=sorted_languages,  # Enforces correct order
        row="transformation",
        hue="condition",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=1.3,
        aspect=1,
        hue_order=hue_order,
    )
    # Plot the learning curves
    g.map(sns.lineplot, "step", "val_ppl", palette=color_palette, linewidth=2)

    # Overlay heatmap tiles in the top-right corner of each subplot
    for (transformation, language), ax in g.axes_dict.items():
        transformation = REVERSE_TRANSFORMATION_MAPPING.get(transformation, None)
        if transformation in heatmap_data.index and language in heatmap_data.columns:
            relative_me = heatmap_data.loc[transformation, language]
            color = cmap(norm(relative_me))  # Get corresponding heatmap color

            # Define rectangle position and size
            rect_x, rect_y = 0.7, 0.72  # Top-right corner
            rect_width, rect_height = 0.3, 0.3  # Small square

            # Add colored rectangle
            rect = patches.Rectangle(
                (rect_x, rect_y),
                rect_width,
                rect_height,
                transform=ax.transAxes,
                color=color,
                alpha=0.7,
                clip_on=False
            )
            ax.add_patch(rect)

            # Adjust font size dynamically
            text_size = 6  # Scale font size

            # Add numerical value inside the rectangle
            ax.text(
                rect_x + rect_width / 2, rect_y + rect_height / 2,  # Center text inside rectangle
                f"{relative_me:+.1f}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=max(text_size, 4),  # Ensures it never gets too small
                fontweight="bold",
                color="black"
            )

    # Adjust ticks and labels
    for ax in g.axes.flat:
        ax.set_xticks(range(0, 3001, 1000))
        ax.set_yticks(range(500, 2001, 500))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.tick_params(axis="both", labelsize=6)

    g.set_axis_labels("", "")
    g.figure.text(0.5, 0.13, "Training Batches", ha="center", va="center", fontsize=12)
    g.figure.text(0.09, 0.5, "Validation Set Perplexity", ha="center", va="center", rotation="vertical", fontsize=12)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=9)

    g.add_legend(title="Condition")
    new_labels = ["Attested", "Perturbed"]  # Custom legend labels
    for t, new_label in zip(g._legend.texts, new_labels):
        t.set_text(new_label)  # Replace "original" with "attested" and "modified" with "perturbed"
    g.figure.subplots_adjust(top=0.93,bottom=0.18)
    # Add a colorbar legend to the bottom of the figure
    fig = g.figure
    cbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.015])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Relative ME", fontsize=12)

    plt.savefig("combined_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_single_language_curves_acl(df, language, ncols=2):
    """
    Demo / aesthetic plot for a single language in ACL single-column format.

    Parameters:
        df: DataFrame with columns ['language', 'transformation', 'condition', 'step', 'val_ppl']
        language: str, the language to plot
        ncols: number of columns in the subplot grid
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Filter for the language
    df_lang = df[df['language'] == language].copy()
    df_lang['transformation'] = df_lang['transformation'].replace(TRANSFORMATION_MAPPING)

    transformations = sorted(df_lang['transformation'].unique())
    n_trans = len(transformations)
    nrows = int(np.ceil(n_trans / ncols))

    # Create grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.5 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    # Colors and linestyles
    color_palette = {"original": "#1f77b4", "modified": "#ff7f0e"}
    linestyle_palette = {"original": "solid", "modified": "dashed"}

    for ax, trans in zip(axes, transformations):
        trans_data = df_lang[df_lang['transformation'] == trans]
        for condition in ['original', 'modified']:
            cond_data = trans_data[trans_data['condition'] == condition]
            ax.plot(
                cond_data['step'], cond_data['val_ppl'],
                color=color_palette[condition],
                linestyle=linestyle_palette[condition],
                linewidth=2.5,
                alpha=0.8,
                label="Original" if condition == "original" else "Perturbed"
            )
        ax.set_title(trans.replace("-", " ").title(), fontsize=10, fontweight='semibold')
        ax.grid(True, alpha=0.2)

    # Turn off unused axes if n_trans < nrows*ncols
    for ax in axes[n_trans:]:
        ax.axis('off')

    # Global labels
    fig.text(0.5, 0.02, "Training Batches", ha='center', va='center', fontsize=11)
    fig.text(0.04, 0.5, "Validation Perplexity", va='center', rotation='vertical', fontsize=11)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], ["Original", "Perturbed"], loc='upper center', ncol=2, fontsize=11, frameon=False)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.92])
    plt.savefig(f"{language}_demo_curves.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_single_language_single_perturbation_demo(df, language, perturbation):
    """
    Plot only the 'full-reverse' transformation for a single language,
    comparing original vs perturbed curves.
    Optimized for ACL single-column demo.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Filter for language and full-reverse transformation
    df_demo = df[(df['language'] == language) & (df['transformation'] == perturbation)].copy()

    # Colors and linestyles
    color_palette = {"original": "#0072B2", "modified": "#D55E00"}  # colorblind-friendly blue/orange
    linestyle_palette = {"original": "solid", "modified": "dashed"}

    plt.figure(figsize=(3.4, 3.2))  # single-column width

    for condition in ['original', 'modified']:
        cond_data = df_demo[df_demo['condition'] == condition]
        plt.plot(
            cond_data['step'], cond_data['val_ppl'],
            color=color_palette[condition],
            linestyle=linestyle_palette[condition],
            linewidth=2.5,
            alpha=0.9,
            label="English" if condition == "original" else "Full-Reverse(English)"
        )
        # Add markers at start and end
        plt.scatter(cond_data['step'].iloc[0], cond_data['val_ppl'].iloc[0],
                    color=color_palette[condition], s=30, edgecolor='black', zorder=5)
        plt.scatter(cond_data['step'].iloc[-1], cond_data['val_ppl'].iloc[-1],
                    color=color_palette[condition], s=30, edgecolor='black', zorder=5)

    # Optional subtle shading between curves to emphasize difference
    orig_vals = df_demo[df_demo['condition'] == 'original']['val_ppl'].values
    mod_vals = df_demo[df_demo['condition'] == 'modified']['val_ppl'].values
    #plt.fill_between(df_demo['step'], orig_vals, mod_vals, color='gray', alpha=0.1)

    # Aesthetics
    #plt.title(f"{language.title()} – Full-Reverse", fontsize=11, fontweight='semibold')
    plt.xlabel("Training Batches", fontsize=10)
    plt.ylabel("Validation Perplexity", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    plt.legend(fontsize=9, frameon=False, loc='upper right')
    plt.savefig(f"{language}_{perturbation}_demo_curves.png", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def make_combined_plot_background(df):
    # Compute relative ME values
    me_df = get_relative_ME(df)
    heatmap_data = me_df.pivot(index="transformation", columns="language", values="relative_me")

    # Normalize color scale for heatmap
    vmin = heatmap_data.min().min()
    vmax = heatmap_data.max().max()
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)  # Adjust vmin, vmax based on data
    cmap = plt.get_cmap("PiYG")

    # Create FacetGrid for line plots
    df['transformation'] = df['transformation'].replace(TRANSFORMATION_MAPPING)
    hue_order = ["original", "modified"]
    color_palette = {"original": "red", "modified": "blue"}  # Switched colors
    sorted_languages = sorted(df["language"].unique())

    # Convert "language" to a categorical variable with explicit ordering
    df["language"] = pd.Categorical(df["language"], categories=sorted_languages, ordered=True)
    print(df["language"].unique())  # Check unique language values
    print(df["language"].dtype)  # Check data type
    # Create FacetGrid with explicit column order
    g = sns.FacetGrid(
        df,
        col="language",
        col_order=sorted_languages,  # Enforces correct order
        row="transformation",
        hue="condition",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=1.3,
        aspect=1,
        hue_order=hue_order,
    )

    # Plot the learning curves
    g.map(sns.lineplot, "step", "val_ppl", palette=color_palette, linewidth=2.2,  alpha=0.8)

    # Apply background color per subplot
    for (transformation, language), ax in g.axes_dict.items():
        transformation = REVERSE_TRANSFORMATION_MAPPING.get(transformation, None)
        if transformation in heatmap_data.index and language in heatmap_data.columns:
            relative_me = heatmap_data.loc[transformation, language]
            color = cmap(norm(relative_me))  # Get corresponding heatmap color

            # Set the background color
            ax.set_facecolor(color)

            # Add numerical value in top-right
            text_x, text_y = 0.95, 0.85  # Adjusted for better visibility
            ax.text(
                text_x, text_y,
                f"{relative_me:+.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                fontweight="bold",
                color="black" if abs(relative_me) < 30 else "white"  # Adjust text color for contrast
            )

    # Adjust ticks and labels
    for ax in g.axes.flat:
        ax.set_xticks(range(0, 3001, 1000))
        ax.set_yticks(range(500, 2001, 500))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.tick_params(axis="both", labelsize=6)

    g.set_axis_labels("", "")
    g.figure.text(0.5, 0.13, "Training Batches", ha="center", va="center", fontsize=15)
    g.figure.text(0.09, 0.5, "Validation Set Perplexity", ha="center", va="center", rotation="vertical", fontsize=15)
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=9)
    g.add_legend(title="Condition")

    # Change legend labels from "original" → "Attested" and "modified" → "Perturbed"
    new_labels = ["Attested", "Perturbed"]
    for t, new_label in zip(g._legend.texts, new_labels):
        t.set_text(new_label)

    g.figure.subplots_adjust(top=0.93, bottom=0.18)

    # Add a colorbar legend to the bottom of the figure
    fig = g.figure
    cbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.015])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mean Error", fontsize=12)

    # Add extreme labels
    cbar_ax.text(-0.05, 0.5, "Preference for\nattested structure",
                 fontsize=10, ha="right", va="center", transform=cbar_ax.transAxes)
    cbar_ax.text(1.05, 0.5, "Preference for\nperturbed structure",
                 fontsize=10, ha="left", va="center", transform=cbar_ax.transAxes)

    plt.savefig("combined_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def demo_compare_variances(df, example_languages=None):
    """
    Demo plot illustrating the idea of comparing variances across languages
    for original vs perturbed conditions. Simplified version.

    Parameters:
        df: DataFrame with columns ['language', 'transformation', 'condition', 'val_ppl']
        example_languages: list of languages to include; if None, use all
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    df_demo = df.copy()
    if example_languages is not None:
        df_demo = df_demo[df_demo['language'].isin(example_languages)]

    unique_languages = list(df_demo['language'].unique())
    transformations = df_demo['transformation'].unique()

    fig, ax = plt.subplots(figsize=(1.5 * len(unique_languages) * len(transformations), 6))

    # Jitter for side-by-side plotting
    jitter = 0.15
    color_palette = {"original": "#1f77b4", "modified": "#ff7f0e"}  # blue / orange

    # Map each language to an x-position
    lang_positions = {lang: i for i, lang in enumerate(unique_languages)}

    for i, lang in enumerate(unique_languages):
        lang_data = df_demo[df_demo['language'] == lang]
        for j, trans in enumerate(transformations):
            trans_data = lang_data[lang_data['transformation'] == trans]
            if len(trans_data) == 2:  # original + modified
                orig_val = trans_data[trans_data['condition'] == 'original']['val_ppl'].values[0]
                mod_val = trans_data[trans_data['condition'] == 'modified']['val_ppl'].values[0]

                x_orig = i + j * 0.2 - jitter
                x_mod = i + j * 0.2 + jitter

                # Draw points
                ax.scatter(x_orig, orig_val, color=color_palette['original'], s=100,
                           label="Original" if i == 0 and j == 0 else "")
                ax.scatter(x_mod, mod_val, color=color_palette['modified'], s=100,
                           label="Perturbed" if i == 0 and j == 0 else "")

                # Optional: draw a line connecting them
                ax.plot([x_orig, x_mod], [orig_val, mod_val], color='gray', alpha=0.5, linewidth=1.5)

    ax.set_xticks(range(len(unique_languages)))
    ax.set_xticklabels(unique_languages, rotation=45, fontsize=12)
    ax.set_ylabel("Validation Perplexity", fontsize=14)
    ax.set_title("Demo: Comparing Original vs Perturbed Variance", fontsize=16)

    # Single legend
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

def compute_df_with_metrics_for_latex(df):
    me_df = get_relative_ME(df)
    min_df = get_min_value(df)
    auc_df = get_AUC(df)
    # Print me_df separately
    me_df = me_df.rename(columns={'transformation':'perturbation', 'relative_me': 'mean_error'})
    me_df[['mean_error']] = me_df[['mean_error']].round(3)
    me_latex = me_df[['language', 'perturbation', 'mean_error']].to_latex(index=False)

    print(me_latex)

    # Now combined min and auc
    combined_df = min_df.merge(auc_df, on=['transformation', 'language', 'condition'])
    combined_df = combined_df.rename(columns={'transformation':'perturbation'})
    combined_df[['min_value', 'AUC']] = combined_df[['min_value', 'AUC']].round(3)

    combined_df = combined_df[['language', 'perturbation', 'condition', 'min_value','AUC']].to_latex(index=False)
    print(combined_df)
    # combined = me_df.merge(min_df, on=['transformation', 'language']) \
    #     .merge(auc_df, on=['transformation', 'language'])
    # combined.drop(columns=['condition_x','condition_y'])
    # combined = combined.rename(columns={'transformation':'perturbation', 'relative_me': 'mean_error'})
    #
    # # Select only the relevant columns
    # table_df = combined[['language', 'perturbation', 'mean_error', 'min_value', 'AUC']]
    #
    # # Optional: round numeric values
    #
    # # Convert to LaTeX
    # latex_table = table_df.to_latex(index=False)
    #
    # print(latex_table)


if __name__ == "__main__":
    # Compares everything present in the gpt2_logs folder.
    # Create csv files from error files.
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_csvs_anew', action='store_true', default=False)
    args = parser.parse_args()
    if args.create_csvs_anew:
        create_all_csvs()

    # Read csvs and concat them.
    df = pd.concat(parse_file(filename) for filename in generate_csv_combos())

    # For "intralinguistic perspective"
    make_combined_plot_background(df) # Creates "Figure 2" style figure.

    # For "interlingusitic perspective"
    compare_metrics_side_by_side(df) # Creates "Figure 3" style figure.

    # But see other functions as well.