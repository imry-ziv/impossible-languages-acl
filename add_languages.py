### Downloads wikidumps to data/wikidumps
### Loads from wikidumps, performs tokenization and formatting by Gulordava (2018) format.
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import sys
import subprocess
from typing import List
import os
import requests
from tqdm import tqdm
from pathlib import Path
import argparse

TREETAGGER_PATH = "../jlm-llm/data/treetagger"
TEMP_OUTPUT_PATH = "../jlm-llm/data/treetagger/temp"
LANGUAGES_TO_DUMP_FILES = {
    'french': ['https://dumps.wikimedia.org/frwiki/20250101/frwiki-20250101-pages-articles-multistream1.xml-p1p306134.bz2'],
    'german': ['https://dumps.wikimedia.org/dewiki/20250101/dewiki-20250101-pages-articles-multistream2.xml-p297013p1262093.bz2'],
    'spanish': ['https://dumps.wikimedia.org/eswiki/20250101/eswiki-20250101-pages-articles-multistream3.xml-p693324p1897740.bz2'],
    'danish': ['https://dumps.wikimedia.org/dawiki/20250101/dawiki-20250101-pages-articles-multistream.xml.bz2', 'https://dumps.wikimedia.org/dawiki/20250101/dawiki-20250101-pages-meta-current.xml.bz2'],
    'japanese': ['https://dumps.wikimedia.org/jawiki/20250101/jawiki-20250101-pages-articles-multistream4.xml-p902408p1721646.bz2', 'https://dumps.wikimedia.org/jawiki/20250101/jawiki-20250101-pages-articles-multistream1.xml-p1p114794.bz2'],
    'greek': ['https://dumps.wikimedia.org/elwiki/20250101/elwiki-20250101-pages-articles-multistream.xml.bz2'],
    'finnish': ['https://dumps.wikimedia.org/fiwiki/20250101/fiwiki-20250101-pages-articles-multistream.xml.bz2']
}

def run_wikiextractor(filepath, count=None, additional_args=None):
    father = os.path.dirname(filepath)
    output_dir = os.path.join(father, str(count))  # Ensure count is converted to a string
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Construct the command
    command = [
        sys.executable,  # Path to the current Python interpreter
        "-m", "wikiextractor.WikiExtractor",
        filepath,
        "--output", output_dir  # Specify the output directory
    ]

    # Add any additional arguments if provided
    if additional_args:
        command.extend(additional_args)

    # Execute the command
    try:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"Extraction completed. Data saved in '{output_dir}'")
    except subprocess.CalledProcessError as e:
        print(f"Error: WikiExtractor failed with error code {e.returncode}.")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: WikiExtractor module not found. Make sure it is installed.")
        sys.exit(1)

def extract_base_name(url):
    """
    Extracts the base name (e.g., 'frwiki') from a Wikipedia dump URL.

    Parameters:
        url (str): The URL of the Wikipedia dump file.

    Returns:
        str: The extracted base name.
    """
    # Split the URL to get the file name
    file_name = url.split("/")[-1]
    # Extract the part before '-latest-pages-articles'
    base_name = file_name.split("-latest-pages-articles")[0]
    return base_name
def download_dumps_for_languages(languages: List[str]):
    '''
    Check first whether is already downloaded in "./data/wikidumps".
    '''
    dumps_path = "./data/wikidumps"
    if not os.path.exists(dumps_path):
        os.makedirs(dumps_path, exist_ok=True)
    downloaded_paths = []
    for language in languages:
        for url in LANGUAGES_TO_DUMP_FILES[language]:
            output_dir = f"{dumps_path}/{language}"
            filename = extract_base_name(url)
            file_path = f"{output_dir}/{filename}"
            os.makedirs(output_dir, exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            print(f"Downloading {filename} to {file_path}...")
            with open(file_path, "wb") as file, tqdm(
                    desc=filename,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    progress_bar.update(len(chunk))

            print(f"Download complete: {file_path}")
            downloaded_paths.append(file_path)


    return downloaded_paths



def parse_dumps(language: str):
    path_to_read = f'./data/wikidumps/{language}'
    total_token_count = 0
    all_sentences = []  # All sentences, after initial processing, before inserting <unk>s.

    # Use os.walk to traverse subdirectories recursively
    for root, dirs, files in tqdm(os.walk(path_to_read), desc="Processing folders"):
        for file_name in tqdm(files, desc=f"Processing files in {root}", leave=False):
            file_path = os.path.join(root, file_name)
            if file_path.endswith('bz2'):
                continue
            output_file = Path(TEMP_OUTPUT_PATH) / (file_name + ".tagged")

            # Tokenize and filter using your existing function
            filtered_data, token_count = tokenize_and_filter_with_treetagger(language, file_path)
            total_token_count += token_count
            logger.debug(f"Tokenized file {file_path}, total tokens: {total_token_count}")
            all_sentences.extend(filtered_data)

            # Stop processing if total tokens reach 100 million
            if total_token_count >= 100_000_000:
                logger.debug(f"Stopping iterations because of sufficient dataset size.")
                return all_sentences
    logger.debug(f"Note: Iterations finished with total token count only at {total_token_count}.")
    return all_sentences

def process_sentences_and_create_vocab(all_sentences, token_limit=90_000_000, vocab_size=50000):
    # Step 1: Shuffle the sentences
    random.shuffle(all_sentences)

    # Step 2: Select a 90M token subset
    token_count = 0
    subset_of_sentences = []
    is_too_short = True
    for sentence in all_sentences:
        tokens = sentence.split()  # Tokenize the sentence
        token_count += len(tokens)
        subset_of_sentences.append(sentence)
        if token_count >= token_limit:
            is_too_short = False
            break
    if is_too_short == True:
        raise Exception("Not enough tokens!")


    # Step 3: Split into train, validation, and test sets
    train_data, temp_data = train_test_split(subset_of_sentences, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    all_tokens = [token for sentence in subset_of_sentences for token in sentence.split()]
    token_freq = Counter(all_tokens)
    vocab = {word: freq for word, freq in token_freq.most_common(vocab_size)}
    if '<unk>' not in vocab.keys():
        vocab['<unk>'] = 100 # Arbitrary
    def replace_with_unk(sentence):
        return [word if word in vocab else '<unk>' for word in sentence.split()]

    train_data = [' '.join(replace_with_unk(sentence)) for sentence in train_data]
    val_data = [' '.join(replace_with_unk(sentence)) for sentence in val_data]
    test_data = [' '.join(replace_with_unk(sentence)) for sentence in test_data]

    # Return processed data and vocabulary
    return {
        'train': train_data,
        'valid': val_data,
        'test': test_data,
        'vocab': vocab,
    }


def tokenize_and_filter_with_treetagger(language, input_file):
    total_token_count = 0
    valid_sentences = []
    lang = language[:2]
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr', TAGDIR=TREETAGGER_PATH)

    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()  # Read the entire content of the file

    for line in lines:
        if line_is_bad(line):  # Skip lines that are short, headers, etc.
            continue

        # Use TreeTagger to tag the text
        tagged_text = tagger.tag_text(line)
        unknown_count = 0
        total_words = len(tagged_text)

        # Calculate the unknown word count
        for token in tagged_text:
            parts = token.split('\t')
            if len(parts) < 2 or parts[2] == "<unknown>":  # Check for unknown words
                unknown_count += 1

        # Skip this sentence if more than 5% of words are unknown
        if total_words == 0 or unknown_count / total_words > 0.05:
            continue

        # Join the words into a full sentence
        sentence = ' '.join([token.split('\t')[0] for token in tagged_text])

        # Ensure full stops are properly spaced
        #spaced_sentence = sentence.replace('.', ' .')

        # Split the sentence around '.', keep the '.', and add <eos>
        split_sentences = [s.strip() + ' .' for s in sentence.split('.') if s.strip()]
        for split_sentence in split_sentences:
            valid_sentences.append(split_sentence + ' <eos>')

        # Update total token count
        total_token_count += total_words

    return valid_sentences, total_token_count


def line_is_bad(line:str):
    '''
    Filter out lines that are single words, only \n-s, too short, or header lines.
    '''
    is_bad = False
    if line == '\n' or len(line.split()) <= 4 or line[0] == '<':
        is_bad = True
    return is_bad

def save_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line + '\n')
def save_vocab(vocab, file_path):
    vocab_words = vocab.keys()
    with open(file_path, 'w', encoding='utf-8') as file:
        for word in vocab_words:
            file.write(word + '\n')



def write_to_txt_files(data_dict, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for name, data in data_dict.items():
        if name == 'vocab':
            save_vocab(data, f"{path}/{name}.txt")
        else:
            save_to_file(data, f"{path}/{name}.txt.original")


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--language',
        help='lowercase',
    )
    arg_parser.add_argument(
        '--download_dumps',
        action='store_true',
        default=False,

    )

    args = arg_parser.parse_args()
    if args.download_dumps:
        dumpfile_paths = download_dumps_for_languages([args.language])
        cnt = 1
        for filepath in dumpfile_paths:
            run_wikiextractor(filepath, cnt)
            cnt += 1
    all_sentences = parse_dumps(args.language)
    split_datasets_and_vocab = process_sentences_and_create_vocab(all_sentences)
    write_to_txt_files(split_datasets_and_vocab, path=f'./data/multilang/{args.language}')
