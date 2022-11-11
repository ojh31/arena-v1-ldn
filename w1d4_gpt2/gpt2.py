#%%
import torch as t
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gpt_architecture
from typing import Optional,Union
import re
import torch.nn as nn
from fancy_einsum import einsum
from typing import Union, Optional
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import time
import wandb
# import sampling
import requests
import glob
import yaml
import pandas as pd
from IPython.display import display
import transformers
import importlib
import numpy as np

#%%
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

#%%
importlib.reload(gpt_architecture)
config = gpt_architecture.TransformerConfig(
    num_layers=12,
    num_heads=8,
    dropout=0.1,
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    max_seq_len=1024,
)
my_gpt = gpt_architecture.GPT(config).train()
gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").train()

#%%
def name_mapper(name):
    return (
        name
        .replace('transformer.h', 'bert_blocks')
        .replace('.ln_', '.layer_norm')
        .replace('mlp.c_fc', 'mlp.linear1')
        .replace('mlp.c_proj', 'mlp.linear2')
        .replace('attn.c_proj', 'attention.W_O')
        .replace('attn.c_attn', 'attention.W_QKV')
        .replace('transformer.wte', 'token_embedding')
        .replace('transformer.wpe', 'positional_embedding')
        .replace('transformer.ln_f', 'ln')
        .replace('transformer.layer_normf', 'layer_norm')
    )

#%%
def print_param_count(*models, display_df=True, use_state_dict=True):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe

    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        print(f"Model {i}, total params = {sum([param.numel() for name, param in model.named_parameters()])}")
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        df = pd.DataFrame([
            {f"name_{i}": name_mapper(name), f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name_mapper(name)}
            for name, param in iterator
        ])
        display(df)
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df
#%%
print_param_count(my_gpt, gpt, use_state_dict=False)

#%%
def copy_weights_from_gpt(my_gpt: GPT, gpt) -> GPT:
    '''
    Copy over the weights from gpt to your implementation of gpt.

    gpt should be imported using: 
        gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    Returns your gpt model, with weights loaded in.

    You might find the function `copy_weights` from w0d3 helpful as a template.
    '''
    hf_params = dict(gpt.named_parameters())
    my_params = dict(my_gpt.named_parameters())
    state_dict = {
        name_mapper(hf_key): hf_value
        for hf_key, hf_value in hf_params.items()
    }
    for k, v in state_dict.items():
        assert v.ndim <= 2
        if v.ndim == 1:
            assert v.shape == my_params[k].shape
        elif v.shape[::-1] == my_params[k].shape:
            state_dict[k] = t.transpose(v, 0, 1) 
        else:
            assert v.shape == my_params[k].shape

    my_gpt.load_state_dict(state_dict, strict=True)
    return my_gpt
# %%
loaded_gpt = copy_weights_from_gpt(my_gpt, gpt)

#%%
def test_load_pretrained_weights(model, tokenizer):

    model.eval()
    device = next(model.parameters()).device
    
    def encode(text: str) -> t.Tensor:
        """Return a Tensor of shape (batch=1, seq)."""
        return tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    prompt = "Former President of the United States of America, George"
    input_ids = encode(prompt)
    with t.inference_mode():
        output = model(input_ids)
        logits = output[0, -1] if isinstance(output, t.Tensor) else output.logits[0, -1]
    topk = t.topk(logits, k=10).indices
    next_tokens = tokenizer.batch_decode(topk.reshape(-1, 1))
    print("Prompt: ", prompt)
    print("Your model's top 10 predictions: ", next_tokens)
    assert " Washington" in next_tokens
    assert " Bush" in next_tokens


#%%
# test_load_pretrained_weights(my_gpt, tokenizer)
# %%
test_load_pretrained_weights(loaded_gpt, tokenizer)
# %%
test_load_pretrained_weights(gpt, tokenizer)
# %%
