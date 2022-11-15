#%%
import transformers
import bert_architecture
import torch.nn as nn
import torch as t
from typing import Optional, List
import numpy as np
import importlib
import pandas as pd
import os
import importlib
from bert_params import reformat_params, print_param_count_from_dicts, copy_weights_from_bert
#%%
# This makes a certain kind of error message more legible
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#%%
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
bert = transformers.AutoModelForCausalLM.from_pretrained("bert-base-cased").train()

#%%
state_dict_names = set(bert.state_dict().keys())
param_names = set(dict(bert.named_parameters()).keys())

print(len(state_dict_names))  # 205
print(len(param_names))       # 202

print(state_dict_names - param_names)

#%% [markdown]
#### Creating BERT and copying weights
#%%
importlib.reload(bert_architecture)
bert_config = bert_architecture.TransformerConfig(
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1,
    layer_norm_epsilon=1e-12,
    max_seq_len=512,
)
my_bert = bert_architecture.BertLanguageModel(bert_config)
my_params = dict(my_bert.named_parameters())
#%%
bert_params = reformat_params(
    dict(bert.named_parameters()), my_params
)
# bert_params = dict(bert.named_parameters())
#%%
print_param_count_from_dicts(my_params, bert_params)

# %%
my_bert = copy_weights_from_bert(my_bert, bert)
# %%
def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    tokens = tokenizer.encode(text, return_tensors='pt')
    with t.inference_mode():
        output = model.eval()(tokens)
    all_logits = output if isinstance(output, t.Tensor) else output.logits
    strings_per_mask = []
    for i, token in enumerate(tokens.tolist()[0]):
        if token != tokenizer.mask_token_id:
            continue
        logits = all_logits[0, i]
        topk_tokens = t.topk(logits, k=k).indices
        topk_words = tokenizer.batch_decode(topk_tokens.reshape(-1, 1))
        strings_per_mask.append(topk_words)
    return strings_per_mask


def test_bert_prediction(predict, model, tokenizer):
    '''Your Bert should know some names of American presidents.'''
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]

#%%
test_bert_prediction(predict, my_bert, tokenizer)

# %%
