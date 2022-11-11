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
import transformers
import torch as t

def beam_search(
    model, input_ids: t.Tensor, num_return_sequences: int, num_beams: int, 
    max_new_tokens: int, tokenizer, verbose=False
) -> list[tuple[float, t.Tensor]]:
    '''
    input_ids: (seq, ) - the prompt
    max_new_tokens: 
        stop after this many new tokens are generated, even if no EOS is generated. In this case, 
        the best incomplete sequences should also be returned.
    verbose: 
        if True, print the current (unfinished) completions after each iteration for debugging purposes

    Return:
        list of length num_return_sequences. 
        Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, 
        sorted by descending logprob.
    '''
    assert num_return_sequences <= num_beams
    seq_len = len(input_ids)
    tbc_seqs = [input_ids]
    tbc_log_probs = [0]
    finished_seqs = []
    finished_log_probs = []
    new_tokens = 0
    while len(finished_seqs) < num_return_sequences and new_tokens < max_new_tokens:
        new_tokens += 1
        new_tbc_seqs = []
        new_tbc_probs = []
        assert len(tbc_seqs) <= num_beams, f'len(tbc_seqs)={len(tbc_seqs)}, num_beams={num_beams}'
        for cur_seq, cur_log_prob in zip(tbc_seqs, tbc_log_probs):
            inp = cur_seq[-seq_len:]
            with t.inference_mode():
                output = model.eval()(inp)
            all_logits = output if isinstance(output, t.Tensor) else output.logits
            logits = all_logits[-1, :]
            log_probs = nn.functional.log_softmax(logits, dim=0)
            for token, token_log_prob in enumerate(log_probs.numpy()):
                new_log_prob = cur_log_prob + token_log_prob
                new_seq = t.cat((cur_seq, t.tensor([token])), dim=0)
                if token == getattr(tokenizer, "eos_token_id", None):
                    finished_seqs.append(new_seq)
                    finished_log_probs.append(new_log_prob)
                else:
                    new_tbc_seqs.append(new_seq)
                    new_tbc_probs.append(new_log_prob)
        new_tbc_argsort = np.argsort(new_tbc_probs)[-num_beams:]
        tbc_seqs = [new_tbc_seqs[i] for i in new_tbc_argsort]
        tbc_log_probs = [new_tbc_probs[i] for i in new_tbc_argsort]
        if verbose:
            print(f'Logging TBC sequences after {new_tokens} new tokens with num_beams={num_beams}')
            for seq in tbc_seqs:
                seq_str = tokenizer.decode(seq)
                print(seq_str)
    finished_seqs, finished_log_probs = finished_seqs + tbc_seqs, finished_log_probs + tbc_log_probs
    finished_argsort = np.argsort(finished_log_probs)[::-1]
    finished_seqs = [finished_seqs[i] for i in finished_argsort]
    finished_log_probs = [finished_log_probs[i] for i in finished_argsort]
    return [(logprob, tokens) for logprob, tokens in zip(finished_log_probs, finished_seqs)]
       

#%%
your_prompt = "I don't want to rule the universe. I just think"
input_ids = tokenizer(your_prompt, return_tensors="pt", return_attention_mask=False)["input_ids"][0]

num_return_sequences = 3
num_beams = 6
max_new_tokens = 10

final_logitsums_and_completions = beam_search(
    gpt, input_ids, num_return_sequences, num_beams, max_new_tokens, tokenizer, verbose=True
)
# %%
