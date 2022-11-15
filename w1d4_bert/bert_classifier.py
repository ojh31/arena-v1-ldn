#%%
import transformers
import bert_architecture
import torch.nn as nn
import torch as t
import numpy as np
import importlib
import pandas as pd
import os
import re
import tarfile
from dataclasses import dataclass
import requests
from torch.utils.data import TensorDataset
import plotly.express as px
import pandas as pd
import time
from bert_params import copy_weights_from_bert_common, reformat_params
import wandb
from tqdm.notebook import tqdm_notebook
from torch.utils.data import DataLoader
#%%
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # This makes a certain kind of error message more legible
device = t.device('cuda')

#%%
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
bert = transformers.AutoModelForCausalLM.from_pretrained("bert-base-cased").train()
#%%
bert_config = bert_architecture.TransformerConfig(
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1,
    layer_norm_epsilon=1e-12,
    max_seq_len=512,
)
# %%
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/bert-imdb/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")
# %%
def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    if not os.path.exists(path):
        with open(path, "wb") as file:
            data = requests.get(url).content
            file.write(data)

#%%
os.makedirs(DATA_FOLDER, exist_ok=True)
expected_hexdigest = "d41d8cd98f00b204e9800998ecf8427e"
maybe_download(IMDB_URL, IMDB_PATH)
#%%
@dataclass(frozen=True)
class Review:
    split: str
    is_positive: bool
    stars: int
    text: str
#%%
def load_reviews(path: str) -> list[Review]:
    reviews = []
    tar = tarfile.open(path, "r:gz")
    for member in tqdm_notebook(tar.getmembers()):
        m = re.match(r"aclImdb/(train|test)/(pos|neg)/\d+_(\d+)\.txt", member.name)
        if m is not None:
            split, posneg, stars = m.groups()
            buf = tar.extractfile(member)
            assert buf is not None
            text = buf.read().decode("utf-8")
            reviews.append(Review(split, posneg == "pos", int(stars), text))
    return reviews
        
#%%
reviews = load_reviews(IMDB_PATH)
assert sum((r.split == "train" for r in reviews)) == 25000
assert sum((r.split == "test" for r in reviews)) == 25000
# %%
# reviews[0]
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

#%%
train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"])
test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"])
t.save((train_data, test_data), SAVED_TOKENS_PATH)
# %%
#%%
importlib.reload(bert_architecture)
classifier = bert_architecture.BertClassifier(bert_config)
classifier = copy_weights_from_bert_common(classifier, bert, hidden_size=bert_config.hidden_size)
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

    batch_size = 4 # wandb.config.batch_size
    epochs = 2 # wandb.config.epochs
    lr = 2e-5 # wandb.config.lr
    clip_grad = 1.0 # wandb.config.clip_grad
    loss_weight = .02 # wandb.config.loss_weight
    weight_decay = .01 # wandb.config.weight_decay

    print('Loading model...')

    model = bert_architecture.BertClassifier(bert_config)
    model = copy_weights_from_bert_common(model, bert, hidden_size=bert_config.hidden_size)
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
            sentiment_loss_fn(sentiment_hat, sentiment_labels).float().squeeze() + 
            stars_loss_fn(star_labels.float(), star_hat.squeeze()) * loss_weight
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
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            sentiment_labels = sentiment_labels.to(device=device, dtype=t.long)
            star_labels = star_labels.to(device)
            optimizer.zero_grad()
            # print(input_ids, attention_mask, sentiment_labels, star_labels)
            print(
                input_ids.shape, attention_mask.shape, 
                sentiment_labels.shape, star_labels.shape
            )
            print(
                input_ids.dtype, attention_mask.dtype, 
                sentiment_labels.dtype, star_labels.dtype
            )
            print('Feeding forward...')
            y_hat = model(input_ids, attention_mask)
            sentiment_hat = y_hat['sentiment']
            star_hat = y_hat['stars']
            print('Computing loss...')
            print(
                sentiment_hat.shape,
                star_hat.shape,
                sentiment_labels.shape,
                star_labels.shape,
            )
            print(
                sentiment_hat.dtype,
                star_hat.dtype,
                sentiment_labels.dtype,
                star_labels.dtype,
            )
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
                sentiment_labels = sentiment_labels.to(device=device, dtype=t.long)
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
