#%%
import hashlib
import os
import sys
import zipfile
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import transformers
from einops import rearrange, repeat, reduce
from torch.nn import functional as F
from tqdm import tqdm
import requests
import utils
import random
import numpy as np

MAIN = __name__ == "__main__"
DATA_FOLDER = "./data"
DATASET = "2"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"wikitext_tokens_{DATASET}.pt")

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
# %%
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
#%%
def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    if not os.path.exists(path):
        with open(path, "wb") as file:
            data = requests.get(url).content
            file.write(data)
# %%
path = os.path.join(DATA_FOLDER, DATASETS[DATASET])
maybe_download(BASE_URL + DATASETS[DATASET], path)
expected_hexdigest = {"103": "0ca3512bd7a238be4a63ce7b434f8935", "2": "f407a2d53283fc4a49bcff21bc5f3770"}
with open(path, "rb") as f:
    actual_hexdigest = hashlib.md5(f.read()).hexdigest()
    assert actual_hexdigest == expected_hexdigest[DATASET]

print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

z = zipfile.ZipFile(path)

def decompress(*splits: str) -> str:
    return [
        z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8").splitlines()
        for split in splits
    ]

train_text, val_text, test_text = decompress("train", "valid", "test")
# %%
random.sample(train_text, 5)
# %%
def tokenize_1d(tokenizer, lines: list[str], max_seq: int) -> t.Tensor:
    '''Tokenize text and rearrange into chunks of the maximum length.

    Return (batch, seq) and an integer dtype.
    '''
    batch_encoding = tokenizer(
        lines, truncation=False, padding=False, add_special_tokens=False
    )
    list_of_tokens = batch_encoding.input_ids
    large_1d_tensor = t.tensor([
        token for tokens in list_of_tokens for token in tokens
    ])
    len_to_keep = (len(large_1d_tensor) // max_seq) * max_seq
    trimmed_1d_tensor = large_1d_tensor[:len_to_keep]
    reshaped_tensor = rearrange(trimmed_1d_tensor, '(b s) -> b s', s=max_seq)
    return reshaped_tensor

if MAIN:
    max_seq = 128
    print("Tokenizing training text...")
    train_data = tokenize_1d(tokenizer, train_text, max_seq)
    print("Training data shape is: ", train_data.shape)
    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Tokenizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)
# %%
def random_mask(
    input_ids: t.Tensor, mask_token_id: int, vocab_size: int, 
    select_frac=0.15, mask_frac=0.8, random_frac=0.1
) -> tuple[t.Tensor, t.Tensor]:
    '''
    Given a batch of tokens, return a copy with tokens replaced according to 
    Section 3.1 of the paper.

    input_ids: (batch, seq)

    Return: (model_input, was_selected) where:

    model_input: (batch, seq) 
        a new Tensor with the replacements made, suitable for passing to 
        the BertLanguageModel. Don't modify the original tensor!

    was_selected: (batch, seq) 
        1 if the token at this index will contribute to the MLM loss, 0 otherwise
    '''
    # Convert to absolute sizes
    batch, seq = input_ids.shape
    select_size = round(select_frac * batch * seq)
    unselected_size = batch * seq - select_size
    mask_size = round(mask_frac * select_size)
    random_size = round(random_frac * select_size)
    # Splitting into unsel_size, mask_size, rand_size, unch_size
    # Pick random split
    rand_index = rearrange(
        t.randperm(batch * seq), 
        '(b s) -> b s', 
        s=seq
    )
    sel_mask = rand_index > unselected_size
    mask_mask = (
        (unselected_size < rand_index) & 
        (rand_index <= unselected_size + mask_size)
    )
    rand_mask = (
        (unselected_size + mask_size < rand_index) & 
        (rand_index <= unselected_size + mask_size + random_size)
    )

    # Construct was_selected
    was_selected = t.zeros_like(input_ids)
    was_selected[sel_mask] = 1
    # Construct model_input
    model_input = t.where(
        mask_mask,
        mask_token_id,
        t.where(
            rand_mask,
            t.randint_like(input_ids, vocab_size),
            input_ids
        )
    )
    return model_input, was_selected


if MAIN:
    utils.test_random_mask(random_mask, input_size=10000, max_seq=max_seq)
# %%
if MAIN:
    # OSKAR CODE
    train_flattened = train_data.flatten()
    n_train = len(train_flattened)
    token_counts = t.bincount(train_flattened)
    token_counts = token_counts[token_counts != 0]
    token_percents = token_counts / n_train
    e_log_p = -(t.log(token_percents) * token_percents).sum()
    print(e_log_p)
    # 7.28
# %%
if MAIN:
    # CALLUM CODE
    # Find the word frequencies
    word_frequencies = t.bincount(train_data.flatten())
    # Drop the words with occurrence zero (because these contribute zero to cross entropy)
    word_frequencies = word_frequencies[word_frequencies > 0]
    # Get probabilities
    word_probabilities = word_frequencies / word_frequencies.sum()
    # Calculate the cross entropy
    cross_entropy = (- word_probabilities * word_probabilities.log()).sum()
    print(cross_entropy)
    # ==> 7.3446
    # OSKAR GETS 7.28
#
#  %%
from fancy_einsum import einsum

def cross_entropy_selected(
    pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor
) -> t.Tensor:
    '''
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 
        1 if the token at this index will contribute to the MLM loss, 
        0 otherwise

    Out: the mean loss per predicted token
    '''
    batch, seq, vocab_size = pred.shape
    vocab_index = repeat(t.arange(0, vocab_size), 'v -> b s v', b=batch, s=seq)
    target_broadcast = repeat(target, 'b s -> b s v', v=vocab_size)
    selected_broadcast = repeat(was_selected, 'b s -> b s v', v=vocab_size)
    pred[(vocab_index != target_broadcast) & (selected_broadcast == 0)] = -t.inf
    pred[(vocab_index == target_broadcast) & (selected_broadcast == 0)] = 0
    pred_flat = rearrange(pred, 'b s v -> (b s) v')
    target_flat = rearrange(target, 'b s -> (b s)')
    return F.cross_entropy(pred_flat, target_flat)

if MAIN:
    utils.test_cross_entropy_selected(cross_entropy_selected)

    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    masked, was_selected = random_mask(
        batch, tokenizer.mask_token_id, tokenizer.vocab_size
    )
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")
# %%
