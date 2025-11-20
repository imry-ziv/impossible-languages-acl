import argparse
import hashlib
import os
import pathlib
import pickle
from tqdm import tqdm
import time
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import sys
import torch.nn as nn
from loguru import logger
from collections import defaultdict
from typing import Optional
DATA_PATH = "./data"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from modified_external_sources.lm_povstim_with_childes.utils_lm_povstim import (
     batchify,
     batchify_finetuning,
     get_batch,
     repackage_hidden,
 )
#from utils import get_free_gpu, kwargs_to_id

TRANSFORMER_DEFAULT_CONTEXT_SIZE = 1024
PARENT_PATH = pathlib.Path.cwd().parent
### Configs.
VALIDATION_SET_SIZE_TOKENS = 10000
MAX_TRAIN_STEPS = 3000
CHECKPOINT_INTERVAL = 100
OUTPUT_DIR = "./gpt2_checkpoints"
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 16

# Defining Dictionary and Tokenizer objects for HF compatibility.


class PlainDictionaryTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, **kwargs):
        """
        Hugging Face-compatible tokenizer using a plain, predefined vocabulary.
        No built-in English tokenization.
        """
        self.vocab = vocab
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = vocab

        # Define special tokens
        kwargs["unk_token"] = "<unk>"
        kwargs["pad_token"] = "<pad>"
        kwargs["bos_token"] = "<s>"
        kwargs["eos_token"] = "</s>"

        # Add special tokens to vocab if missing
        for token in kwargs.values():
            if token not in self.word2idx:
                self.word2idx[token] = len(self.idx2word)
                self.idx2word.append(token)

        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.idx2word)

    def _tokenize(self, text):
        """Splits text into tokens based on whitespace only (no subword processing)."""
        return text.split()

    def _convert_token_to_id(self, token):
        """Converts a token to its corresponding ID."""
        return self.word2idx.get(token, self.word2idx["<unk>"])

    def _convert_id_to_token(self, index):
        """Converts an ID back to its corresponding token."""
        return self.idx2word[index] if index < len(self.idx2word) else "<unk>"

    def convert_tokens_to_string(self, tokens):
        """Joins tokens into a string."""
        return " ".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids):
        """Adds special tokens (if needed) to tokenized input."""
        return [self.word2idx["<s>"]] + token_ids + [self.word2idx["</s>"]]

    @classmethod
    def from_dictionary(cls, dictionary):
        """Create a tokenizer from a Dictionary object."""
        vocab = dictionary.idx2word
        return cls(vocab=vocab)



class Dictionary(object):
    def __init__(
        self,
        datasets_dir: pathlib.Path,
        base_dataset_path: pathlib.Path,
    ):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)

        vocab_path = base_dataset_path / "vocab.txt"
        paths = [
            base_dataset_path / "train.txt.original",
        ]

        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logger.info("Vocab file not found, creating new vocab file...")
            self.create_vocab(paths)
            logger.info("Done creating new vocab file.")
            open(vocab_path, "w").write("\n".join([w for w in self.idx2word]))

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        # return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, paths):
        for path in paths:
            with open(path, "r", encoding="utf8") as f:
                for line in f:
                    words = line.split()
                    for word in words:
                        self.add_word(word)


class Corpus(object):
    def __init__(
        self,
        datasets_dir: pathlib.Path,
        base_dataset: str,
    ):
        logger.info(
            f"Initializing corpus for base dataset {base_dataset}."
        )

        base_dataset_path = datasets_dir / base_dataset
        base_train_path = base_dataset_path / "train.txt.original"
        base_valid_path = base_dataset_path / f"valid.txt.original"
        base_test_path = base_dataset_path / "test.txt.original"
        base_train_hash = _get_file_hash(base_train_path)

        logger.debug(f"Base_train_hash is {base_train_hash}")

        train_paths = [base_train_path]
        valid_paths = [base_valid_path]
        test_paths = [base_test_path]

        train_dict_hash_file = (
            datasets_dir
            / f"train_dict_{base_train_hash}.cached"
        )

        if pathlib.Path(train_dict_hash_file).exists():
            with open(train_dict_hash_file, "rb") as f:
                self.dictionary = pickle.load(f)
            logger.info(f"Loaded dictionary from {train_dict_hash_file}")
        else:
            self.dictionary = Dictionary(
                datasets_dir=datasets_dir,
                base_dataset_path=base_dataset_path,
            )
            with open(train_dict_hash_file, "wb") as f:
                pickle.dump(self.dictionary, f)
                logger.info(f"Saved dictionary to {train_dict_hash_file}")
        self.tokenizer = load_tokenizer_from_dictionary(self.dictionary)
        self.train = tokenize(self.tokenizer, train_paths, 256)
        self.valid = tokenize(self.tokenizer, valid_paths, 256)
        self.test = tokenize(self.tokenizer, test_paths, 256)
        # Iterate over tokenized IDs in self.train
        logger.info("Done tokenizing.")


def downsample_for_epochs(
    token_tensor,
    num_epochs,
    training_steps,
    chunk_size,
    batch_size,
    seed=42
):
    """
    token_tensor: 1D tensor of token IDs, shape (N,)
    num_epochs: desired number of epochs (e.g., 10)
    training_steps: total number of training steps you will run (e.g., 3000)
    chunk_size: number of tokens per example (e.g., 256)
    batch_size: examples per training step (e.g., 512)
    """

    torch.manual_seed(seed)

    # tokens consumed per training step
    tokens_per_step = batch_size * chunk_size

    # total tokens needed for training
    total_tokens_needed = training_steps * tokens_per_step

    # tokens needed per epoch (if total equals num_epochs epochs)
    tokens_per_epoch = total_tokens_needed // num_epochs

    # that is the target downsample length
    target_length = int(tokens_per_epoch)

    original_length = token_tensor.shape[0]

    if target_length > original_length:
        raise ValueError(
            f"Dataset too small. Need {target_length} tokens but only have {original_length}."
        )

    # downsample the dataset by random selection of a contiguous slice
    start = torch.randint(0, original_length - target_length, (1,))
    target_length = 150000 # Just trying out
    downsampled = token_tensor[start:start + target_length]

    return downsampled, {
        "original_length": original_length,
        "target_length": target_length,
        "tokens_per_step": tokens_per_step,
        "total_tokens_needed": total_tokens_needed,
        "tokens_per_epoch": tokens_per_epoch,
        "achieved_epochs": total_tokens_needed / target_length
    }


def _get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]



def tokenize(tokenizer, paths, batch_size):
    """Tokenizes a text file for training or testing to a sequence of indices format
    We assume that training and test data has <eos> symbols"""

    logger.info(f"Tokenizing {', '.join(map(str,paths))}...")

    for path in paths:
        assert path.exists(), path

    all_tokenized_ids = []
    for path in paths:
        all_tokenized_ids.append(_tokenize_file(tokenizer, path)) # Batch size here != batch size of model

    ids = torch.concat(all_tokenized_ids)
    return ids


def _tokenize_file(tokenizer, path, batch_size=32):
    file_hash = _get_file_hash(path)
    grnn_data_path = path.parent.parent  # .../grnn_data/wikipedia/train.txt.original
    cache_path = grnn_data_path / f"tokenized_ids_{file_hash}.pt"

    if cache_path.exists():
        return torch.load(cache_path)

    with open(path, "r", encoding="utf8") as f:
        text = f.read()

    # Split the text into chunks (e.g., paragraphs or sentences)
    # This example assumes you're splitting by newlines, adjust as needed
    chunks = text.split('\n')  # Split by newlines or another method

    # Tokenize in batches
    all_token_ids = []
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        batch_text = ' '.join(batch)  # Combine batch into one string

        # Tokenize the batch
        token_ids = tokenizer.encode(batch_text, add_special_tokens=False)
        all_token_ids.extend(token_ids)

    # Convert to a PyTorch tensor
    ids = torch.tensor(all_token_ids, dtype=torch.long)

    # Save the tokenized data to disk
    torch.save(ids, cache_path)
    return ids


# Convert your Dictionary object into a Hugging Face-compatible tokenizer
def load_tokenizer_from_dictionary(dictionary):
    return PlainDictionaryTokenizer.from_dictionary(dictionary)


def create_dictionary_for_datasets(train, test, valid):
    """
    Receives all three datasets (train, test, valid) and creates a dictionary object from them.
    """
def truncate_validation(val_data):
    """
    Truncate the validation data to VALIDATION_SET_SIZE_TOKENS.
    """

    # Ensure val_data is truncated to the correct size
    if val_data.size(0) > VALIDATION_SET_SIZE_TOKENS:
        val_data = val_data[:VALIDATION_SET_SIZE_TOKENS]  # Slice only along the first dimension
    return val_data

def load_and_tokenize_datasets(base_dataset_name, seed, cuda, gpu_id, do_downsample):
    # base_dataset_name includes dataset.
    # Set the random seed manually for reproducibility.
    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)
            logger.debug(f"Set seed to {seed} for PyTorch.")
            torch.cuda.empty_cache()
            if gpu_id is None:
                gpu_id = get_free_gpu()
            else:
                gpu_id = gpu_id
            logger.info(f"Using GPU ID {gpu_id}")
            torch.cuda.set_device(gpu_id)
    ###############################################################################
    # Load data
    ###############################################################################

    start = time.time()
    corpus = Corpus(
        datasets_dir=pathlib.Path(DATA_PATH),
        base_dataset=pathlib.Path(base_dataset_name,),
    )

    if do_downsample:
        corpus.train, data = downsample_for_epochs(corpus.train,
                                             num_epochs=10,
                                             training_steps=3000,
                                             chunk_size=256,
                                             batch_size=512)

    logger.debug("Created Corpus object.")
    logger.debug(f"Here's the data: {data}")
    logger.info("( %.2f )" % (time.time() - start))
    ntokens_by_dict = len(corpus.dictionary)
    ntokens_by_tokenizer = corpus.tokenizer.vocab_size
    assert (ntokens_by_dict == ntokens_by_tokenizer)
    logger.info(f"Vocab size is {ntokens_by_dict}")


    # train_data = batchify(corpus.train, TRAIN_BATCH_SIZE, cuda)
    # val_data = batchify(corpus.valid, EVAL_BATCH_SIZE, cuda)
    # test_data = batchify(corpus.test, EVAL_BATCH_SIZE, cuda)

    # Edit 0903: no prebatchifying, let Trainer handle it.
    train_data = corpus.train
    test_data = corpus.test
    val_data = truncate_validation(corpus.valid)
    criterion = nn.CrossEntropyLoss()
    train_dataset = GPT2Dataset(train_data)
    test_dataset = GPT2Dataset(test_data)
    val_dataset = GPT2Dataset(val_data)

    return train_dataset, test_dataset, val_dataset, corpus

def configure_gpt2_model(
        vocab_size,
        n_positions,
        n_embd,
        n_layer,
        n_head,
):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head
    )

    model = GPT2LMHeadModel(config)

    return model


def init_trainer_object(model, train_dataset, test_dataset, val_dataset, corpus, is_downsampled):
    if is_downsampled:
        log_dir = "./logs_downsampled"
        # Arguments for Nov '25 training by Kallini.
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="steps",
            eval_steps=CHECKPOINT_INTERVAL,
            save_strategy="no",  # Prevents saving the model and tokenizer
            logging_steps=100,
            learning_rate=6e-4,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=10,
            max_steps=MAX_TRAIN_STEPS,  # Stop training after this many steps
            warmup_steps=300, # Added!
            logging_dir=log_dir,
        )

    else:
        log_dir = "./logs"
        epoch_num = 1
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="steps",
            eval_steps=CHECKPOINT_INTERVAL,
            save_strategy="no",  # Prevents saving the model and tokenizer
            logging_steps=100,
            learning_rate=5e-4,
            weight_decay=0.01,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=epoch_num,
            max_steps=MAX_TRAIN_STEPS,  # Stop training after this many steps
            logging_dir=log_dir,
        )


    # ==== Train the Model ====

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=corpus.tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=corpus.tokenizer,
            mlm=False  # GPT-2 does not use masked language modeling
        ),
        compute_metrics=lambda eval_pred: compute_perplexity(eval_pred, global_step=trainer.state.global_step),
        # Custom metric function to calculate perplexity

    )

    return trainer


def compute_perplexity(eval_pred, global_step=None):
    logits, labels = eval_pred
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    try:

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        # Flatten the logits and labels for loss calculation
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss)
        if global_step is not None:
            logger.debug(f"Step {global_step}: Perplexity = {perplexity.item()}")

        return {"perplexity": perplexity.item()}
    except Exception as e:
        logger.error(f"Error in compute_perplexity: {str(e)}")
        logger.error(f"loss_fct: {loss_fct}")
        logger.error(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        raise
class GPT2Dataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data  # tensor of shape (num_samples, seq_length)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],  # Shape: (seq_length,)
            "attention_mask": torch.ones_like(self.data[idx])  # Assuming no padding
        }
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--language",
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--method",
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=73,
    )

    arg_parser.add_argument(
        "--context_size",
        type=int,
        default=30,
    )

    arg_parser.add_argument(
        "--do_downsample",
        action="store_true",  # <-- correct way for a boolean flag
        help="Whether to downsample the dataset",
    )

    args = arg_parser.parse_args()
    base_dataset_name = f"multilang/{args.language}"
    if not args.method == 'no-perturb':
        base_dataset_name = f"{base_dataset_name}/{args.method}"
    # Check for cuda.
    cuda = 1 if os.getenv("CUDA") else 0

    # Still keep Corpus object. train, test and val are batchified (10 by default) - validation is truncated by token num
    logger.debug("Tokenizing and loading datasets.")


    train_dataset, test_dataset, val_dataset, corpus_object = load_and_tokenize_datasets(
        base_dataset_name=base_dataset_name,
        seed=args.seed,
        cuda=cuda,
        gpu_id=None,
        do_downsample=args.do_downsample
    )

    model = configure_gpt2_model(
        vocab_size = corpus_object.tokenizer.vocab_size,
        n_positions=TRANSFORMER_DEFAULT_CONTEXT_SIZE,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    if torch.cuda.is_available():
        model=model.to("cuda")

    trainer = init_trainer_object(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        corpus=corpus_object,
        is_downsampled=args.do_downsample,
    )
    train_batch_size = trainer.get_train_dataloader().batch_size
    logger.debug(f"Batch size of trainer is {train_batch_size}")

    trainer.train()