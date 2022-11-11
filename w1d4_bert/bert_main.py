#%%
import transformers
import bert_architecture
import torch.nn as nn
import torch as t
from typing import Optional, List
from einops import rearrange, reduce, repeat
from fancy_einsum import einsum
import numpy as np
import importlib
from IPython.display import display
import pandas as pd

#%%
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
bert = transformers.AutoModelForCausalLM.from_pretrained("bert-base-cased").train()

#%%
state_dict_names = set(bert.state_dict().keys())
param_names = set(dict(bert.named_parameters()).keys())

print(len(state_dict_names))  # 205
print(len(param_names))       # 202

print(state_dict_names - param_names)

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

#%%
my_params = dict(my_bert.named_parameters())
def reformat_params(params: dict[str, nn.Parameter]):
    params = {rename_param(k): v for k, v in params.items()}
    new_state_dict = {
        k: v 
        for k, v in params.items()
        if 'W_Q' not in k and 'W_V' not in k and 'W_K' not in k
    }
    # print(f'Left {len(new_state_dict)} params unchanged up to renaming')
    concatenated = {
        k: t.concat([
            params[k.replace('QKV', 'Q')],
            params[k.replace('QKV', 'K')],
            params[k.replace('QKV', 'V')],
        ])
        for k in my_params.keys()
        if 'W_QKV' in k
    }
    # print(f'Added {len(concatenated)} params by concatenation')
    new_state_dict.update(concatenated)
    return new_state_dict
#%%
bert_params = reformat_params(dict(bert.named_parameters()))
# bert_params = dict(bert.named_parameters())
#%%
print_param_count_from_dicts(my_params, bert_params)
# %%
def copy_weights_from_bert(
    my_bert: bert_architecture.BertLanguageModel, 
    bert: transformers.models.bert.modeling_bert.BertForMaskedLM
) -> bert_architecture.BertLanguageModel:
    '''
    Copy over the weights from bert to your implementation of bert.

    bert should be imported using: 
        bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

    Returns your bert model, with weights loaded in.
    '''
    bert_params = reformat_params(dict(bert.named_parameters()))
    assert set(bert_params.keys()) == set(my_params)
    my_bert.load_state_dict(bert_params)
    return my_bert

# %%
loaded_bert = copy_weights_from_bert(my_bert, bert)
# %%
def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    tokens = tokenizer.encode(text, return_tensors='pt')
    print(tokens)
    with t.inference_mode():
        output = model.eval()(tokens)
    all_logits = output if isinstance(output, t.Tensor) else output.logits
    strings_per_mask = []
    for i, token in enumerate(tokens):
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
