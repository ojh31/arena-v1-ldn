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
from typing import List, Tuple

MAIN = __name__ == "__main__"
DATA_FOLDER = "/home/ubuntu/arena-v1-ldn/w2d5/data"
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
# random.sample(train_text, 5)
# %%
def tokenize_1d(tokenizer, lines: List[str], max_seq: int) -> t.Tensor:
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
) -> Tuple[t.Tensor, t.Tensor]:
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
        t.randperm(batch * seq, device=input_ids.device), 
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
    was_selected[sel_mask] = t.ones_like(was_selected[sel_mask])
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

#%%
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
import importlib
importlib.reload(utils)

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
    pred_flat = rearrange(pred, 'b s v -> (b s) v')
    target_flat = rearrange(target, 'b s -> (b s)')
    selected_flat = rearrange(was_selected, 'b s -> (b s)')

    return F.cross_entropy(
        pred_flat[selected_flat == 1], 
        target_flat[selected_flat == 1]
    )

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
from dataclasses import dataclass

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05

#%%
hidden_size = 512
bert_config_tiny = TransformerConfig(
    num_layers = 8,
    num_heads = hidden_size // 64,
    vocab_size = 28996,
    hidden_size = hidden_size,
    max_seq_len = 128,
    dropout = 0.1,
    layer_norm_epsilon = 1e-12
)

config_dict = dict(
    lr=0.0002,
    epochs=40,
    batch_size=128,
    weight_decay=0.01,
    mask_token_id=tokenizer.mask_token_id,
    warmup_step_frac=0.01,
    eps=1e-06,
    max_grad_norm=None,
)
#%%
train_data, val_data, test_data = t.load("./data/wikitext_tokens_2.pt")
print("Training data size: ", train_data.shape)

train_loader = DataLoader(
    TensorDataset(train_data), shuffle=True, 
    batch_size=config_dict["batch_size"], drop_last=True
)
# %%
import matplotlib.pyplot as plt

def lr_for_step(step: int, max_step: int, max_lr: float, warmup_step_frac: float):
    '''Return the learning rate for use at this step of training.'''
    warmup_steps = max_step * warmup_step_frac
    initial_lr = max_lr * 0.1
    if step < warmup_steps:
        return initial_lr + (max_lr - initial_lr) * float(step) / warmup_steps
    return (
        max_lr - 
        (max_lr - initial_lr) * float(step - warmup_steps) / (max_step - warmup_steps)
    )


if MAIN:
    max_step = int(len(train_loader) * config_dict["epochs"])
    lrs = [
        lr_for_step(
            step, max_step, max_lr=config_dict["lr"], 
            warmup_step_frac=config_dict["warmup_step_frac"]
        )
        for step in range(max_step)
    ]
    fig, ax = plt.subplots()
    fig.suptitle(
        f'schedule for max_step={max_step}, max_lr={config_dict["lr"]}, '
        f'wu={config_dict["warmup_step_frac"]}'
    )
    ax.plot(lrs)
    ax.set_ylabel('lr')
    ax.set_xlabel('step')
# %%
import bert_architecture
importlib.reload(bert_architecture)
def make_optimizer(
    model: bert_architecture.BertLanguageModel, config_dict: dict
) -> t.optim.AdamW:
    '''
    Loop over model parameters and form two parameter groups:

    - The first group includes the weights of each Linear layer and 
        uses the weight decay in config_dict
    - The second has all other parameters and uses weight decay of 0
    '''
    weight_decay = config_dict['weight_decay']
    lr = config_dict['lr']
    params = dict(model.named_parameters())
    weight_keys = {
        p_name for p_name in params.keys()
        if 'weight' in p_name and (
            'bert_block' in p_name or 'linear' in p_name
        )
    }
    weight_params = [
        p_val for p_name, p_val in params.items() 
        if p_name in weight_keys
    ]
    bias_params = [
        p_val for p_name, p_val in params.items() 
        if p_name not in weight_keys
    ]
    optimizer = t.optim.AdamW(
        [{'params': weight_params, 'weight_decay': weight_decay}, 
        {'params': bias_params, 'weight_decay': 0}], 
        lr=lr
    )
    return optimizer

if MAIN:
    test_config = TransformerConfig(
        num_layers = 3,
        num_heads = 1,
        vocab_size = 28996,
        hidden_size = 1,
        max_seq_len = 4,
        dropout = 0.1,
        layer_norm_epsilon = 1e-12,
    )

    optimizer_test_model = bert_architecture.BertLanguageModel(test_config)
    opt = make_optimizer(
        optimizer_test_model, 
        dict(weight_decay=0.1, lr=0.0001, eps=1e-06)
    )
    expected_num_with_weight_decay = test_config.num_layers * 6 + 1
    wd_group = opt.param_groups[0]
    actual = len(wd_group["params"])
    assert (
        actual == expected_num_with_weight_decay
    ), f"Expected 6 linear weights per layer (4 attn, 2 MLP) plus the final lm_linear weight to have weight decay, got {actual}"
    all_params = set()
    for group in opt.param_groups:
        all_params.update(group["params"])
    assert all_params == set(optimizer_test_model.parameters()), "Not all parameters were passed to optimizer!"
# %%
import wandb
from tqdm.notebook import tqdm_notebook
import time
device = 'cuda'

def bert_mlm_pretrain(
    model: bert_architecture.BertLanguageModel, config_dict: dict, 
    train_loader: DataLoader
) -> None:
    '''Train using masked language modelling.'''
    wandb.init(config=config_dict)
    optimizer = make_optimizer(model, config_dict)
    examples_seen = 0
    step = 0
    start_time = time.time()
    epochs = config_dict['epochs']
    mask_token_id = config_dict['mask_token_id']
    max_lr = config_dict['lr']
    warmup_step_frac = config_dict['warmup_step_frac']
    clip_grad = 1.0
    max_step = len(train_loader) * epochs
    for _ in range(epochs):
        
        for y, in tqdm_notebook(train_loader):
            lr = lr_for_step(
                step, max_step, max_lr=max_lr, 
                warmup_step_frac=warmup_step_frac
            )
            for g in optimizer.param_groups:
                g['lr'] = lr
            y = y.to(device)
            x, selected = random_mask(y, mask_token_id, tokenizer.vocab_size)
            x = x.to(device)
            selected = selected.to(device)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = cross_entropy_selected(y_hat, y, selected)
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            examples_seen += len(y)
            step += 1
            wandb.log({
                "train_loss": loss, "elapsed": time.time() - start_time
            }, step=examples_seen)
    
    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)


if MAIN:
    model = bert_architecture.BertLanguageModel(bert_config_tiny).to(device).train()
    num_params = sum((p.nelement() for p in model.parameters()))
    print("Number of model parameters: ", num_params)
    bert_mlm_pretrain(model, config_dict, train_loader)
# %%
