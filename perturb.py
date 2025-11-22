import multiprocessing
import os
import shutil
import string
import spacy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import lib
from loguru import logger
import argparse
import random
from spacy.cli import download
from functools import partial

# Paths.
_HOME = os.path.expanduser("~")
_NOUNS = {'NOUN', 'PROPN'}
_REV_MARKER = '<rev>'
_DETERMINER_EOR_TOKEN_OFFSET = 3
_TOKEN_HOP_TOKEN_OFFSET = 3
_SWITCH_INDEX_1 = 3
_SWITCH_INDEX_2 = 0
_LANGUAGE_MARKERS = {
    'en': 'v',
    'heb': 'פ',
    'ru': chr(1095),
    'it': 'v',
    'fr': 'v',
    'da': 'v',
    'ja': chr(0x3094), # vu, phonetic element
    'el': chr(0x03BE), # xi
    'de': 'V',
    'es': 'v',
    'fi': 'v',
    'tr': 'v',
    'ko': chr(0xD7A3), # not a word, used as pedagogical term
}

_LANGUAGE_TO_LANGUAGE_CODE = {
    'english': 'en',
    'hebrew': 'heb',
    'russian': 'ru',
    'italian': 'it',
    'french': 'fr',
    'japanese': 'ja',
    'greek': 'el',
    'spanish': 'es',
    'german': 'de',
    'finnish': 'fi',
    'danish': 'da',
    'turkish': 'tr',
    'korean': 'ko',
}


SEED = 81
DATA_PATH = "./data"

# Global vars for HF models.
_HEBREW_MODEL = None
_HEBREW_TOKENIZER = None

_ITALIAN_DETERMINERS_TO_EQUIVALENTS = {
    'il': 'lu',
    'i': 'loi',
    'lo': 'cuo',
    'gli': 'scui',
    'la': 'iara',
    'le': 'iare',
    'l\'': 'lu',
    'un': 'ru',
    'uno': 'ruo',
    'una': 'rua',
    'un\'': 'ru\'',
}

_SET_TYPES = ('train', 'test', 'valid')

_IMPLEMENTED_PERTURBATIONS = (
    'count-based-indefinite-article',
    'linear-left-number-agreement',
    'linear-dependency-length-reduplication',
    'index-based-indefinite-article',
    'full-reverse',
    'partial-reverse',
    'shuffle-global',
    'shuffle-local-5',
    'shuffle-local-10',
    'shuffle-local-2',
    'shuffle-global-non-deterministic',
    'italian-determiner-eor',
    'no-hop',
    'switch-indices',
    'token-hop',
    'reverse-baseline',
    'capitalize-by-length',
)


def load_dataset(language: str, set_type: str, method: str):
    """
    Loads and returns set_type (='train', 'test' or 'valid') dataset for specified language as list of strings.
    """
    language_path = os.path.join(DATA_PATH, 'multilang', language)
    if method == 'token-hop':
        return _load_dataset_token_hop(language, set_type)
    with open(os.path.join(language_path, f'{set_type}.txt.original'), 'r', encoding='utf-8') as file:
        dataset = file.readlines()
    return dataset

def _load_dataset_token_hop(language: str, set_type: str):
    method_path_no_hop = os.path.join(DATA_PATH, 'multilang', language, "no-hop")
    method_path_token_hop = os.path.join(DATA_PATH, 'multilang', language, "token-hop")
    with open(os.path.join(method_path_no_hop, f'{set_type}.txt.original'), 'r', encoding='utf-8') as file:
        dataset = file.readlines()
    return dataset



def load_sanity_check():
    with open('sanity_check_sentences.txt', 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    for i in range(len(sentences)):
        sentences[i] = sentences[i].rstrip()
    return sentences

def load_hebrew_hf_model():
    global _HEBREW_MODEL, _HEBREW_TOKENIZER
    if _HEBREW_MODEL is None or _HEBREW_TOKENIZER is None:
        _HEBREW_TOKENIZER = AutoTokenizer.from_pretrained('dicta-il/dictabert-morph')
        _HEBREW_MODEL = AutoModel.from_pretrained('dicta-il/dictabert-morph', trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f'Device: {device}.')
        _HEBREW_MODEL = _HEBREW_MODEL.to(device)
        _HEBREW_MODEL.eval()

def perturb_dataset(dataset: list, method: str, set_type: str, language: str):
    """
    Given dataset, returns perturbed version of it - list of sentence strings.
    Insert required method as string: ...
    """
    dataset = list(dataset)
    change_counter = 0
    nlp = None

    if 'linear' in method or 'hop' in method:
        if language != 'hebrew':
            nlp, language_code = _load_model(language)
        elif language == 'hebrew': # Edge case for Hebrew where we use huggingface model.
            load_hebrew_hf_model()
            nlp = None
            language_code = _LANGUAGE_TO_LANGUAGE_CODE[language]

    if language == 'hebrew' and method == 'no-hop': # Batchify.
        dataset = _process_in_batches_no_hop(dataset, batch_size=32)
    else:
        for i in tqdm(range(len(dataset)), desc=f'Processing {set_type}'):
            sentence = dataset[i]
            sentence_no_eos = sentence.rstrip("<eos>")
            perturbed_sentence, change = _perturb_sentence(sentence_no_eos, method, nlp, language)
            if change:
                change_counter += 1
            dataset[i] = perturbed_sentence
        change_ratio = change_counter / len(dataset) * 100
        logger.debug(f'There were {change_counter} changes, overall {change_ratio} percent.')

    method_path = os.path.join(DATA_PATH, 'multilang', language, method)
    print(f"method_path: {method_path}")
    print(f"set_type: {set_type}")
    if not os.path.exists(method_path):
        os.mkdir(method_path)
    with open(os.path.join(method_path, f'{set_type}.txt.original'), "w", encoding='utf-8') as file:
        for item in dataset:
            file.write("%s\n" % item)

    method_path = os.path.join(DATA_PATH, 'multilang', language, method)
    return method_path

def _process_in_batches_no_hop(dataset, batch_size):
    perturbed_dataset = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing Batches", unit="batch"):
        batch = dataset[i:i + batch_size]  # Create batch
        perturbed_batch = _apply_no_hop_hebrew(batch)
        perturbed_dataset.extend(perturbed_batch)
    return perturbed_dataset




def update_vocab(language: str, method:str):

    # Copy vocab and add if needed.
    logger.debug(f'Creating vocab for language {language}')
    language_path = os.path.join(DATA_PATH, 'multilang', language)
    src = os.path.join(language_path, 'vocab.txt')
    dst = os.path.join(os.path.join(language_path, method), 'vocab.txt')
    shutil.copy(src,dst)
    if method == "italian-determiner-eor":
        newlines = [new_determiner + "\n" for new_determiner in _ITALIAN_DETERMINERS_TO_EQUIVALENTS.values()]
        with open(dst, "a", encoding="utf-8") as file:
            file.writelines(newlines)
    elif 'hop' in method:
        with open(dst, "a", encoding="utf-8") as file:
            file.writelines([_LANGUAGE_MARKERS[_LANGUAGE_TO_LANGUAGE_CODE[language]] + "\n"])
    return
def _perturb_sentence(sentence: str, method: str, nlp: spacy.Language, language: str):
    if method == 'count-based-indefinite-article':
        perturbed, change = _apply_count_based_indefinite_article(sentence)

    elif method == 'linear-dependency-length-reduplication':
        perturbed, change = _apply_linear_dependency_length_reduplication(sentence, nlp)

    elif method == 'index-based-indefinite-article':
        perturbed, change = _apply_index_based_indefinite_article(sentence)

    elif method == 'full-reverse':
        perturbed = _apply_full_reverse(sentence)
        change = 1

    elif method == 'partial-reverse':
        perturbed = _apply_partial_reverse(sentence)
        change = 1

    elif method == 'reverse-baseline':
        perturbed = _apply_reverse_baseline(sentence)
        change = 1

    elif method == 'shuffle-global':
        perturbed = _apply_shuffle_global(sentence)
        change = 1

    elif method == 'shuffle-local-5':
        perturbed = _apply_local_shuffle(sentence, 5)
        change = 1
    elif method == 'shuffle-local-10':
        perturbed = _apply_local_shuffle(sentence, 10)
        change = 1

    elif method == 'shuffle-local-2':
        perturbed = _apply_local_shuffle(sentence, 2)
        change = 1

    elif method == 'shuffle-global-non-deterministic':
        perturbed = _apply_shuffle_global_non_deterministic(sentence)
        change = 1

    elif method == 'italian-determiner-eor':
        perturbed = _apply_italian_determiner_eor(sentence)
        change = 1

    elif method == 'no-hop':
        perturbed, change = _apply_no_hop(sentence, nlp)

    elif method == 'token-hop':
        perturbed, change = _apply_token_hop(sentence, language)
    elif method == 'switch-indices':
        perturbed, change = _apply_switch_indices(sentence, language)

    elif method == 'capitalize-by-length':
        perturbed, change = _apply_capitalize_by_length(sentence, language)
    else:
        perturbed = None
        change = None
    return perturbed, change

def _load_model(language:str):
    assert language in ('english','italian','russian', 'french', 'german', 'greek', 'spanish', 'danish', 'japanese', 'finnish', 'turkish', 'korean'), f"{language} not supported with POS tagging"
    if language == 'english':
        name = "en_core_web_sm"
    elif language == 'italian':
        name = "it_core_news_sm"
    elif language == 'russian':
        name = "ru_core_news_sm"
    elif language == 'french':
        name = "fr_core_news_lg"
    elif language == 'danish':
        name = "da_core_news_md"
    elif language == 'spanish':
        name = "es_core_news_sm"
    elif language == 'japanese':
        name = "ja_core_news_sm"
    elif language == 'greek':
        name = "el_core_news_md"
    elif language == 'german':
        name = "de_core_news_sm"
    elif language == 'finnish':
        name = "fi_core_news_md"
    elif language == 'korean':
        name = "ko_core_news_sm"
    elif language == 'turkish':
        name = "tr_core_news_lg"

    try:
        nlp = spacy.load(name)
    except:
        #download(name)
        nlp = spacy.load(name)
    language_code = nlp.lang
    return nlp, language_code



def _apply_count_based_indefinite_article(sentence: str):
    """
    indefinite article = 'an' iff len(adjacent_word_to_right) <= 5, else article = 'a'
    """
    articles = {'a', 'an'}
    split_sentence = sentence.split()
    changed = False
    if not any(word.lower() in articles for word in split_sentence):
        return sentence, changed
    for i in range(len(split_sentence)):
        word = split_sentence[i]
        if word in articles:
            try:
                next = split_sentence[i + 1]
                if len(next) <= 5:
                    if word == 'a':
                        changed = True
                    article = 'an'
                else:
                    if word == 'an':
                        changed = True
                    article = 'a'
                split_sentence[i] = article
            except IndexError:
                continue
    return " ".join(split_sentence), changed

def _apply_switch_indices(sentence: str, language: str):
    uppercase = False if language == 'hebrew' else True

    changed = 0
    tokens = sentence.split()
    eos = tokens.pop()
    final_punc = None
    if tokens[-1] in string.punctuation:
        final_punc = tokens.pop()
    if len(tokens) >= 4:
        temp = tokens[_SWITCH_INDEX_1].capitalize()
        tokens[_SWITCH_INDEX_1] = decapitalize(tokens[_SWITCH_INDEX_2])
        tokens[_SWITCH_INDEX_2] = temp
        changed = 1

    if final_punc:
        tokens.append(final_punc)
    tokens.append(eos)
    return " ".join(tokens), changed

def decapitalize(s):
    if not s:  # check that s is not empty string
        return s
    return s[0].lower() + s[1:]

def _apply_capitalize_by_length(sentence, language):
    if language != 'hebrew':
        sentence_split = sentence.split()
        length = len(sentence_split)
        sentence_split[0] = sentence_split[0].lower()
        index_to_capitalize = length % min(5,length)
        sentence_split[index_to_capitalize] = sentence_split[index_to_capitalize].capitalize()
        return " ".join(sentence_split), 1
    else:
        pass


def _apply_index_based_indefinite_article(sentence: str):
    """
    Value of article is 'a' iff index(a) % 2 == 0, else 'an'.
    Returns (perturbed sentence, change flag).
    """
    articles = {'a', 'an'}
    split_sentence = sentence.split()  # Tokenization is done by whitespace split, like in colorlessgreenRNNs...dictionary_corpus
    changed = False

    if not any(word.lower() in articles for word in split_sentence):
        return sentence, changed
    for i in range(len(split_sentence)):
        word = split_sentence[i]
        if word.lower() in articles:
            if i % 2 == 0:
                article = 'a'
                if word == 'an':
                    changed = True
            else:
                article = 'an'
                if word == 'a':
                    changed = True
            split_sentence[i] = article

    return " ".join(split_sentence), changed


def _apply_italian_determiner_eor(sentence: str):
    tokens = sentence.split()
    eos = tokens.pop()
    final_punc = None
    if tokens[-1] in string.punctuation:
        final_punc = tokens.pop()

    for i in range(len(tokens)):
        token = tokens[i]
        ltoken = token.lower()
        if ltoken in _ITALIAN_DETERMINERS_TO_EQUIVALENTS.keys():
            try:
                tokens.insert(i+_DETERMINER_EOR_TOKEN_OFFSET, _ITALIAN_DETERMINERS_TO_EQUIVALENTS[ltoken])
            except IndexError:
                tokens.append(_ITALIAN_DETERMINERS_TO_EQUIVALENTS[ltoken])

    if final_punc:
        tokens.append(final_punc)
    tokens.append(eos)

    return ' '.join(tokens)



    #
    # succeeding_determiners = []
    # tokens = sentence.split()
    # eos = tokens.pop()
    # final_punc = None
    #
    # if tokens[-1] in string.punctuation:
    #     final_punc = tokens.pop()
    #
    # for token in tokens:
    #     ltoken = token.lower()
    #     if ltoken in _ITALIAN_DETERMINERS_TO_EQUIVALENTS.keys():
    #         succeeding_determiners.append(_ITALIAN_DETERMINERS_TO_EQUIVALENTS[ltoken])
    #
    # tokens.extend(succeeding_determiners)
    # if final_punc:
    #     tokens.append(final_punc)
    # tokens.append(eos)

    #return ' '.join(tokens)


def _apply_intertwined_italian_determiner_eor(sentence: str):
    pass


def _apply_full_reverse(sentence: str):
    tokens = sentence.split()
    marker_index = random.randint(0, len(tokens) - 1)
    tokens.insert(marker_index, _REV_MARKER)

    if tokens and tokens[-1] == '<eos>':
        tokens.pop()

    tokens.reverse()

    if tokens[0] == '.':
        tokens.pop(0)
        tokens.append('.')
    tokens.append('<eos>')
    reversed_sentence = ' '.join(tokens)

    return reversed_sentence


def _apply_pairwise_reverse(sentence: str):
    words = sentence.split()
    eos = words.pop()
    for i in range(0, len(words) - 1, 2):
        words[i], words[i + 1] = words[i + 1], words[i]
    words.append(eos)
    words[0] = words[0].capitalize()
    return ' '.join(words)


def _apply_reverse_baseline(sentence: str):
    tokens = sentence.split()
    if tokens[-1] == "<eos>":
        tokens.pop()

    add_full_stop = False
    if tokens[-1] == '.':
        tokens.pop(-1)
        add_full_stop = True
    if len(tokens) < 2:
        return _apply_full_reverse(sentence)

    marker_index = random.randint(0, len(tokens) - 2)
    tokens.insert(marker_index, _REV_MARKER)
    # tokens[marker_index + 1:] = reversed(tokens[marker_index + 1:]) # commented out for baseline
    if add_full_stop:
        tokens.append('.')
    tokens.append('<eos>')
    sentence = ' '.join(tokens)
    return sentence


def _apply_partial_reverse(sentence: str):
    tokens = sentence.split()
    if tokens[-1] == "<eos>":
        tokens.pop()

    add_full_stop = False
    if tokens[-1] == '.':
        tokens.pop(-1)
        add_full_stop = True
    if len(tokens) < 2:
        return _apply_full_reverse(sentence)

    marker_index = random.randint(0, len(tokens) - 2)
    tokens.insert(marker_index, _REV_MARKER)
    tokens[marker_index + 1:] = reversed(tokens[marker_index + 1:])
    if add_full_stop:
        tokens.append('.')
    tokens.append('<eos>')
    sentence = ' '.join(tokens)
    return sentence


def _apply_shuffle_global(sentence: str):
    tokens = sentence.split()
    if tokens[-1] == "<eos>":
        tokens.pop()

    seed = hash(len(tokens))  # generate seed based on length in tokens
    random.seed(seed)
    # Fisher-Yates shuffle algorithm
    shuffled = tokens[:]
    random.shuffle(shuffled)

    shuffled.append('<eos>')
    return ' '.join(shuffled)

def _apply_shuffle_global_non_deterministic(sentence: str):
    tokens = sentence.split()
    if tokens[-1] == "<eos>":
        tokens.pop()

    shuffled = tokens[:]
    random.shuffle(shuffled)

    shuffled.append("<eos>")
    return " ".join(shuffled)


def _apply_local_shuffle(sentence: str, window_size: int):
    tokens = sentence.split()
    tokens[0] = tokens[0].lower()
    if tokens[-1] == "<eos>":
        tokens.pop()

    if len(sentence) < window_size:
        return _apply_shuffle_global(sentence)

    shuffled_tokens = tokens[:]
    for i in range(0, len(tokens) - window_size + 1, window_size):
        window = shuffled_tokens[i:i + window_size]

        # Generate a seed based on the length of the window
        seed = hash(window_size)
        random.seed(seed)
        random.shuffle(window)
        shuffled_tokens[i:i + window_size] = window

    shuffled_tokens.append("<eos>")
    shuffled_tokens[0] = shuffled_tokens[0].capitalize()
    return " ".join(shuffled_tokens)


def _verify_present(token):
    aux_verbs = [aux.text.lower() for aux in token.head.children if aux.dep_ == "aux"]
    if any(aux in ["will", "shall", "should", "would", "could", "might", "can", "may", "must"] for aux in aux_verbs):
        return False
    # Check for negations
    if any(child.text.lower() in ["n't"] for child in token.children):
        return False
    # Check for passive voice
    if token.dep_ == "auxpass":
        return False
    # Check for continuous/progressive aspect
    if token.tag_ == "VBG":
        return False
    # Check for imperative mood
    if token.dep_ == "ROOT" and token.tag_ == "VB" and token.head.tag_ == "VBZ":
        return False
    return True


def _apply_linear_dependency_length_reduplication(sentence: str, nlp: spacy.Language):
    """
    Verb is reduplicated (inside two token instances, e.g. "ate" -> "ate ate")
    iff
    length of dependency between verb and subject is longer than 4 tokens.
    """
    doc = nlp(sentence)
    verbs = lib.find_verbs(doc)
    sentence_with_indices = [[token.text, False] for token in doc]
    changed = False
    # Save original index in sentence
    for parent_index, verb_token in verbs:
        for child in verb_token.children:
            if child.dep_ in lib.SUBJECT_LABELS:  # Then "child" is a subject of  "verb_token"
                child_index = child.i
                if abs(child_index - parent_index) > 5:  # Arbitrarily - count four indices
                    changed = True
                    # If count is larger than four, reduplicate verb
                    sentence_with_indices[parent_index][1] = True
    perturbed_sentence = []
    for token, flag in sentence_with_indices:
        perturbed_sentence.append(token)
        if flag:  # append again if flag is on
            perturbed_sentence.append(token)

    return " ".join(perturbed_sentence), changed

def _apply_no_hop(sentence:str, nlp:spacy.Language):
    """
    Currently: find verbs (in each language), append marker next to them.
    """
    if nlp == None: # Edge case for Hebrew: perform inference with loaded HF model.
        return _apply_no_hop_hebrew(sentence), 1
    tokens = sentence.split()
    eos = tokens.pop()
    punct = None
    if tokens[-1] in string.punctuation:
        punct = tokens.pop()

    doc = nlp(" ".join(tokens))
    modified_tokens = []
    change = 0
    language_code = nlp.lang
    for token in doc:
        pos = token.pos_
        if token.text in {'unk', '<', '>'}:
            modified_tokens.append(token.text)
        elif pos == 'VERB':
            modified_tokens.append(token.text)  # Add the original token
            modified_tokens.append(_LANGUAGE_MARKERS[language_code])  # Add the special token
            change = 1
        else:
            modified_tokens.append(token.text)
    if punct:
        modified_tokens.append(punct)
    modified_tokens.append(eos)
    return _custom_unk_join(modified_tokens), change

def _apply_no_hop_hebrew(batch):
    """
    Receives list of sentences (batch).
    """
    global _HEBREW_MODEL
    global _HEBREW_TOKENIZER
    batch_predictions = _HEBREW_MODEL.predict(batch, _HEBREW_TOKENIZER)
    batch_modified = [process_sentence(pred, _LANGUAGE_MARKERS['heb']) for pred in batch_predictions]
    return batch_modified

def process_sentence(predicted_output, insert_letter):
    """Transform a predicted sentence output into the modified sentence."""
    tokens = predicted_output['tokens']
    modified_tokens = [token for token_info in tokens for token in process_token(token_info, insert_letter)]
    return _custom_unk_join(modified_tokens)

def process_token(token_info, insert_letter):
    """Modify tokens by inserting `insert_letter` after verbs."""
    if token_info['pos'] == 'VERB':
        return [token_info['token'], insert_letter]
    return [token_info['token']]
#
# def process_sentence()
#     tokens = predicted_output[0]['tokens']
#     modified_sentence = []
#     for i, token_info in enumerate(tokens):
#         if token_info['token'] in {'unk', '<', '>'}:
#             modified_sentence.append(token_info['token'])
#         elif token_info['pos'] == 'VERB':
#             modified_sentence.append(token_info['token'])
#             modified_sentence.append(_LANGUAGE_MARKERS['heb'])
#         else:
#             modified_sentence.append(token_info['token'])
#
#     return _custom_unk_join(modified_sentence), 1


def _apply_token_hop(sentence:str, language:str):
    language_code = _LANGUAGE_TO_LANGUAGE_CODE[language]
    separating_token = _LANGUAGE_MARKERS[language_code]
    split_sentence = sentence.split()
    eos = split_sentence.pop()
    final_punc = None
    if split_sentence[-1] in string.punctuation:
        final_punc = split_sentence.pop()

    indices = [i for i, x in enumerate(split_sentence) if x == separating_token]  # Find all indices of the target
    elements_to_move = []

    # Collect all target elements and mark them for removal
    for i in sorted(indices, reverse=True):  # Iterate in reverse order to avoid index shift
        elements_to_move.append(split_sentence.pop(i))

    # Move each element to its new position
    for element in reversed(elements_to_move):  # Iterate in reverse to maintain order
        current_index = len(split_sentence)
        new_position = min(indices.pop(0) + 3, current_index)
        split_sentence.insert(new_position, element)
    if final_punc:
        split_sentence.append(final_punc)
    split_sentence.append(eos)
    return " ".join(split_sentence), 1




def _custom_unk_join(tokens):
    result = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        # Handle tokens like <unk> or <eos>
        if token == '<' and i + 2 < len(tokens) and tokens[i + 2] == '>':
            temp_token = '<' + tokens[i + 1] + '>'
            if result and not result[-1].endswith(' '):  # Add a space before <unk> if needed
                result.append(' ' + temp_token)
            else:
                result.append(temp_token)
            i += 3  # Skip ahead to the next token after the closing >
        elif token in {'.', ',', ':', ';', '!', '?', '"', "'", '...'}:  # Handle punctuation
            # Add punctuation as a separate token
            if result and not result[-1].endswith(' '):
                result.append(' ' + token)
            else:
                result.append(token)
            result.append(' ')  # Add a space after punctuation
            i += 1
        elif token in {'(', ')'}:  # Handle parentheses as standalone
            if result and not result[-1].endswith(' '):
                result.append(' ' + token)
            else:
                result.append(token)
            result.append(' ')  # Add a space after the parenthesis
            i += 1
        else:  # Regular word
            if result:
                result.append(' ' + token)
            else:
                result.append(token)
            i += 1

    return ''.join(result).strip()  # Join and remove trailing space




def process_set_type(set_type, language, method):
    """
    Function to process a single set_type with the given method and language.
    """
    logger.debug(f'Creating {set_type} dataset')
    ds = load_dataset(language, set_type, method)
    dataset_path = perturb_dataset(ds, method, set_type=set_type, language=language)
    logger.debug(f'Wrote perturbed dataset for {set_type} to path {dataset_path}.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="Perturbation method")
    parser.add_argument("--language")
    parser.add_argument("--no_run", action='store_true', default=False)
    parser.add_argument("--paralellize", action='store_true', default=False)
    args = parser.parse_args()
    assert args.method in _IMPLEMENTED_PERTURBATIONS, "Invalid method."
    if not args.no_run:
        logger.debug(f'Running perturbation scheme: {args.method} for language {args.language}')
        if args.paralellize:
            num_cpus = min(len(_SET_TYPES), multiprocessing.cpu_count())
            with multiprocessing.Pool(processes=num_cpus) as pool:
                pool.map(partial(process_set_type, language=args.language, method=args.method), _SET_TYPES)
        else:
            for type in _SET_TYPES:
                process_set_type(type, args.language, args.method)
    if 'hop' in args.method or 'eor' in args.method:
        update_vocab(args.language, args.method)
