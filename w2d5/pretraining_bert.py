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
    batch_encoding = tokenizer(lines, truncation=False)
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
