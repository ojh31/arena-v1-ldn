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
import os
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

#%% [markdown]
#### Fine-tuning BERT: IMDB
# %%
import os
import re
import tarfile
from dataclasses import dataclass
import requests
import torch as t
import transformers
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
import plotly.express as px
import pandas as pd
from typing import Callable, Optional, List
import time

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/bert-imdb/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")

device = t.device('cpu')
# %%
def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    if not os.path.exists(IMDB_PATH):
        with open(IMDB_PATH, "wb") as file:
            data = requests.get(url).content
            file.write(data)

os.makedirs(DATA_FOLDER, exist_ok=True)
expected_hexdigest = "d41d8cd98f00b204e9800998ecf8427e"
maybe_download(IMDB_URL, IMDB_PATH)

@dataclass(frozen=True)
class Review:
    split: str
    is_positive: bool
    stars: int
    text: str

def load_reviews(path: str) -> list[Review]:
    reviews = []
    tar = tarfile.open(path, "r:gz")
    for member in tqdm(tar.getmembers()):
        m = re.match(r"aclImdb/(train|test)/(pos|neg)/\d+_(\d+)\.txt", member.name)
        if m is not None:
            split, posneg, stars = m.groups()
            buf = tar.extractfile(member)
            assert buf is not None
            text = buf.read().decode("utf-8")
            reviews.append(Review(split, posneg == "pos", int(stars), text))
    return reviews
        
reviews = load_reviews(IMDB_PATH)
assert sum((r.split == "train" for r in reviews)) == 25000
assert sum((r.split == "test" for r in reviews)) == 25000
# %%
reviews[0]
# # %%
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# review_df = pd.DataFrame(reviews)
# review_df['length'] = review_df.text.str.len()
# review_df.head()
# # %%
# fig, ax = plt.subplots()
# sns.histplot(x=review_df.stars, ax=ax, bins=10)
# # %%
# assert review_df.stars.ne(5).any()
# assert review_df.stars.ne(6).any()
# # %%
# review_df.is_positive.value_counts()
# # %%
# review_df.split.value_counts()
# # %%
# fig, ax = plt.subplots()
# sns.histplot(x=review_df.length, ax=ax, bins=200)
# ax.set_xlim([0, 2000])
# # %%
# review_df.length.describe()
# # %%
# fig, ax = plt.subplots()
# sns.histplot(x=review_df.stars, hue=review_df.is_positive, ax=ax, bins=10)
# #%%
# fig, ax = plt.subplots()
# sns.histplot(x=review_df.length, hue=review_df.is_positive, ax=ax, bins=200)
# ax.set_xlim([0, 2000])
# # %%
# assert review_df.loc[review_df.is_positive].stars.min() == 7
# # %%
# assert review_df.loc[~review_df.is_positive].stars.max() == 4
# # %%
# import ftfy
# # %%
# review_df['badness'] = [
#     ftfy.badness.badness(text) for text in review_df.text
# ]
# #%%
# review_df.badness.gt(0).sum(), len(review_df)
# # %%
# fig, ax = plt.subplots()
# sns.histplot(x=review_df.loc[review_df.badness.gt(0)].badness, ax=ax)
# # %%
# review_df.badness.sum()
# # %%
# review_df.loc[review_df.badness.gt(0)].text.iloc[0]
# # %%
# review_df.loc[review_df.badness.gt(0)].badness.iloc[0]
# # %%
# ftfy.fix_text(review_df.loc[review_df.badness.gt(0)].text.iloc[0])
# # Fixes the \x85 ellipses
# # %%
# from lingua import LanguageDetectorBuilder
# detector = LanguageDetectorBuilder.from_all_languages().build()
# # %%
# languages = []
# for text in tqdm(review_df.text.sample(n=100)):
#     languages.append(detector.detect_language_of(text))
# # %%
# pd.Series(languages).value_counts()
# %%
from torch.utils.data import TensorDataset

def to_dataset(tokenizer, reviews: list[Review]) -> TensorDataset:
    '''Tokenize the reviews (which should all belong to the same split) and 
    bundle into a TensorDataset.

    The tensors in the TensorDataset should be (in this exact order):

    input_ids: shape (batch, sequence length), dtype int64
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    '''
    encoding = tokenizer.batch_encode_plus(
        [review.text for review in reviews],
        max_length=tokenizer.model_max_length,
        return_tensors='pt',
        truncation=True,
        padding=True,
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    sentiment_labels = t.tensor([review.is_positive for review in reviews])
    star_labels = t.tensor([review.stars for review in reviews])
    return TensorDataset(input_ids, attention_mask, sentiment_labels, star_labels)

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
#%%
train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"])
test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"])
t.save((train_data, test_data), SAVED_TOKENS_PATH)
# %%
def copy_weights_from_bert_common(
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
    bert_params = {k: v for k, v in bert_params.items() if 'common' in k}
    classifier_params = {
        'stars.weight': t.randn((1, bert_config.hidden_size)),
        'stars.bias': t.randn((1)),
        'sentiment.weight': t.randn((2, bert_config.hidden_size)),
        'sentiment.bias': t.randn((2)),
    }
    bert_params.update(classifier_params)
    my_bert.load_state_dict(bert_params)
    return my_bert

#%%
importlib.reload(bert_architecture)
classifier = bert_architecture.BertClassifier(bert_config)
# %%
loaded_classifier = copy_weights_from_bert_common(classifier, bert)
#%%
# import wandb
from tqdm.notebook import tqdm_notebook
from torch.utils.data import DataLoader
# %%
def train():
 
    # wandb.init(
    #     project='imdb',
    #     config = {
    #         'batch_size': 16,
    #         'hidden_size': bert_config.hidden_size,
    #         'lr': 2e-5,
    #         'epochs': 2,
    #         'max_seq_len': bert_config.max_seq_len,
    #         'dropout': 0.1,
    #         'num_layers': bert_config.num_layers,
    #         'num_heads': bert_config.num_heads,
    #         'loss_weight': .02,
    #         'clip_grad': 1.0,
    #         'weight_decay': .01,
    #     }
    # ) 

    batch_size = 16 # wandb.config.batch_size
    epochs = 2 # wandb.config.epochs
    lr = 2e-5 # wandb.config.lr
    clip_grad = 1.0 # wandb.config.clip_grad
    loss_weight = .02 # wandb.config.loss_weight
    weight_decay = .01 # wangb.config.weight_decay

    print('Loading model...')

    model = bert_architecture.BertClassifier(bert_config)
    model = copy_weights_from_bert_common(model, bert)
    model = model.to(device).train()

    print('Creating optimiser...')
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sentiment_loss_fn = nn.CrossEntropyLoss()
    stars_loss_fn = nn.functional.l1_loss

    examples_seen = 0
    start_time = time.time()

    print('Loading data...')

    train_data, test_data = t.load(SAVED_TOKENS_PATH)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )
    
    def loss_fn(y_hat, y):
        sentiment_hat, star_hat = y_hat
        sentiment_labels, star_labels = y
        loss = (
            sentiment_loss_fn(sentiment_hat, sentiment_labels) + 
            stars_loss_fn(star_labels, star_hat) * loss_weight
        )
        return loss

    
    # wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    print('Starting epochs...')

    for epoch in range(epochs):
        progress_bar = tqdm_notebook(train_dataloader)

        print('model.train(')
        model.train()
        for input_ids, attention_mask, sentiment_labels, star_labels in progress_bar:
            print('Unpacking trainloader')
            print(input_ids, attention_mask, sentiment_labels, star_labels)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            sentiment_labels = sentiment_labels.to(device=device, dtype=t.int)
            star_labels = star_labels.to(device)
            optimizer.zero_grad()
            print('Feeding forward...')
            sentiment_hat, star_hat = model(input_ids, attention_mask)
            print('COmputing loss...')
            loss = loss_fn((sentiment_hat, star_hat), (sentiment_labels, star_labels))
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")
            examples_seen += len(input_ids)
            # wandb.log(
            #     {"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen
            # )

        print('model.eval(')
        with t.inference_mode():
            model.eval()
            sentiment_correct = 0
            total = 0
            for input_ids, attention_mask, sentiment_labels, star_labels in test_dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                star_labels = star_labels.to(device)
                sentiment_labels = sentiment_labels.to(device=device, dtype=t.int)
                sentiment_hat, star_hat = model(input_ids, attention_mask)
                sentiment_predictions = sentiment_hat.argmax(-1)
                sentiment_correct += (sentiment_predictions == sentiment_labels).sum().item() 
                total += star_labels.size(0)

            # wandb.log({"sentiment_accuracy": sentiment_correct/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {sentiment_correct}/{total}")

    # filename = f"{wandb.run.dir}/model_state_dict.pt"
    # print(f"Saving model to: {filename}")
    # t.save(model.state_dict(), filename)
    # wandb.save(filename)
    return model
# %%
model = train()
# %%
