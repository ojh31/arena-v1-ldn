#%%
import transformers
import bert_architecture
import torch.nn as nn
import torch as t
from typing import Optional, List
import numpy as np
from IPython.display import display
import pandas as pd

#%%
def rename_param(p: str):
    return (
        p
        .replace('bert.embeddings.word_embeddings', 'common.token_embedding')
        .replace('bert.embeddings.position_embeddings', 'common.positional_embedding')
        .replace('bert.embeddings.token_type_embeddings', 'common.segment_embedding')
        .replace('bert.encoder.layer', 'common.bert_blocks')
        .replace('intermediate.dense', 'mlp.linear1')
        .replace('attention.output.dense', 'attention.W_O')
        .replace('bert.embeddings.LayerNorm', 'common.layer_norm')
        .replace('self.query', 'W_Q')
        .replace('self.key', 'W_K')
        .replace('self.value', 'W_V')
        .replace('attention.output.LayerNorm', 'layer_norm1')
        .replace('output.LayerNorm', 'layer_norm2')
        .replace('cls.predictions.bias', 'unembed_bias')
        .replace('cls.predictions.transform.LayerNorm', 'layer')
        .replace('cls.predictions.transform.dense', 'linear')
        .replace('output.dense', 'mlp.linear2')
    )

#%%
def print_param_count_from_dicts(
    *model_params, display_df=True
):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe
    """
    df_list = []
    gmap_list = []
    for i, param_dict in enumerate(model_params, start=1):
        iterator = sorted(param_dict.items())
        print(f"Model {i}, total params = {sum([param.numel() for name, param in iterator])}")
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        display(df)
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(model_params) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(model_params) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(model_params) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(model_params) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df

def reformat_params(
    hf_params: dict[str, nn.Parameter],
    my_params: dict[str, nn.Parameter],
):
    hf_params = {rename_param(k): v for k, v in hf_params.items()}
    new_state_dict = {
        k: v 
        for k, v in hf_params.items()
        if 'W_Q' not in k and 'W_V' not in k and 'W_K' not in k
    }
    # print(f'Left {len(new_state_dict)} params unchanged up to renaming')
    concatenated = {
        k: t.concat([
            hf_params[k.replace('QKV', 'Q')],
            hf_params[k.replace('QKV', 'K')],
            hf_params[k.replace('QKV', 'V')],
        ])
        for k in my_params.keys()
        if 'W_QKV' in k
    }
    # print(f'Added {len(concatenated)} params by concatenation')
    new_state_dict.update(concatenated)
    return new_state_dict

#%%
def copy_weights_from_bert(
    my_bert: bert_architecture.BertLanguageModel, 
    hf_bert: transformers.models.bert.modeling_bert.BertForMaskedLM
) -> bert_architecture.BertLanguageModel:
    '''
    Copy over the weights from bert to your implementation of bert.

    bert should be imported using: 
        bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

    Returns your bert model, with weights loaded in.
    '''
    my_params = dict(my_bert.named_parameters())
    hf_params = dict(hf_bert.named_parameters())
    bert_params = reformat_params(hf_params, my_params)
    assert set(bert_params.keys()) == set(my_params)
    my_bert.load_state_dict(bert_params)
    return my_bert

# %%
def copy_weights_from_bert_common(
    my_bert: bert_architecture.BertClassifier, 
    hf_bert: transformers.models.bert.modeling_bert.BertForMaskedLM,
    hidden_size: int,
) -> bert_architecture.BertClassifier:
    '''
    Copy over the weights from bert to your implementation of bert.

    bert should be imported using: 
        bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

    Returns your bert model, with weights loaded in.
    '''
    my_params = dict(my_bert.named_parameters())
    hf_params = dict(hf_bert.named_parameters())
    bert_params = reformat_params(hf_params, my_params)
    
    bert_params = {k: v for k, v in bert_params.items() if 'common' in k}
        
    classifier_params = {
        'stars.weight': t.randn((1, hidden_size)),
        'stars.bias': t.randn((1)),
        'sentiment.weight': t.randn((2, hidden_size)),
        'sentiment.bias': t.randn((2)),
    }
    bert_params.update(classifier_params)
    assert set(bert_params.keys()) == set(my_params)
    my_bert.load_state_dict(bert_params)
    return my_bert

