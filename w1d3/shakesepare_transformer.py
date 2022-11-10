# %%

import torch as t
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transformer_replication
from typing import Optional,Union
import re
import torch.nn as nn
from fancy_einsum import einsum
from typing import Union, Optional
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import time
import wandb
import sampling
import requests
import glob
import yaml
# %%
device = t.device('cuda')
#%%
class WordsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        sample = (text, label)
        return sample

# %%
def tokenize(text):
    return re.split(r"\b", text)

def _remove_duplicates(text, string=" "):
    if string + string in text:
        text = text.replace(string + string, string)
        return _remove_duplicates(text, string)
    return text

def remove_duplicates(text):
    text = _remove_duplicates(text, ' ')
    text = _remove_duplicates(text, '\n')
    return text

# %%
class WordData():
    def __init__(self, text, start, end):
        self.complete_text = remove_duplicates(text)
        if start is not None and end is not None:
            self.complete_text = self.get_excerpt(start, end)
        self.complete_tokens = tokenize(self.complete_text)
        self.vocab = sorted(set(self.complete_tokens))
        self.token_to_id = dict(zip(self.vocab, list(range(len(self.vocab)))))
        self.id_to_token = dict(zip(list(range(len(self.vocab))), self.vocab))
        self.model_max_length = None

    @staticmethod
    def from_link(link, start=None, end=None):
        return WordData(requests.get(link).content.decode('utf-8'), start, end)
    
    @staticmethod
    def from_file(filename, start=None, end=None):
        with open(filename, encoding='utf-8') as f:
            text = f.read()
        return WordData(text, start, end)

    def get_excerpt(self, start="THE SONNETS", end="THE END", text=None):
        if text is None:
            text = self.complete_text
        assert start in text, f'get_excerpt: cannot find {start} in text'
        l_stripped = text.split(start, maxsplit=1)[1]
        assert end in l_stripped, f'get_excerpt: cannot find {end} in text'
        r_stripped = l_stripped.split(end, maxsplit=1)[0]
        return r_stripped

    def generate_autoregressive_dataset(self, sequence_length, text=None):
        self.model_max_length = sequence_length
        if text is None:
            text = self.complete_text
        token_ids = self.encode(text, return_tensors="pt")
        inputs = [token_ids[i:i + sequence_length] for i in range(len(token_ids) - sequence_length)]
        labels = [token_ids[i + 1:i + 1 + sequence_length] for i in range(len(token_ids) - sequence_length)]
        return WordsDataset(inputs, labels)

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        tokens = tokenize(initial_text)
        token_ids = [self.token_to_id[t] for t in tokens]
        if return_tensors == "pt":
            return t.tensor(token_ids, device=device)
        return token_ids

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        tokens = [self.id_to_token[int(i)] for i in list_of_ids]
        return "".join(tokens)

#%%
# shakespeare = WordData.from_file('100-0.txt', start="1\n", end='ALL’S WELL THAT ENDS WELL')
# print('Vocab size: ', len(shakespeare.vocab))
#%%
mlk = WordData.from_file('mlk.txt')
print('Vocab size: ', len(mlk.vocab))

#%%
def train_mlk():
 
    wandb.init(
        project='mlk',
        config = {
            'batch_size': 64,
            'hidden_size': 512,
            'lr': 0.001,
            'epochs': 1,
            'max_seq_len': 30,
            'dropout': 0.1,
            'num_layers': 6,
            'num_heads': 8,
        }
    ) 

    transformer_config = transformer_replication.TransformerConfig(
        num_layers=wandb.config.num_layers,
        num_heads=wandb.config.num_heads,
        vocab_size=len(mlk.vocab),
        hidden_size=wandb.config.hidden_size,
        max_seq_len=wandb.config.max_seq_len,
        dropout=wandb.config.dropout
    )

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = transformer_replication.DecoderOnlyTransformer(transformer_config).to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    examples_seen = 0
    start_time = time.time()

    traintext = mlk.get_excerpt("Mr. Chairman", "James Weldon")
    testtext = mlk.get_excerpt("Johnson:", "glory of the coming of the Lord!!")

    # traintext = mlk.get_excerpt("From fairest", "140")
    # testtext = mlk.get_excerpt("140", "THE END")

    trainset = mlk.generate_autoregressive_dataset(transformer_config.max_seq_len, traintext)
    testset = mlk.generate_autoregressive_dataset(transformer_config.max_seq_len, testtext)

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    print(f'# characters: train={len(traintext)}, test={len(testtext)}')
    print(f'# sequences: train={len(trainset)}, test={len(testset)}')
    
    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):
        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(rearrange(y_hat, "batch seq vocab_size -> (batch seq) vocab_size"), rearrange(y, "batch seq -> (batch seq)"))
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")
            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        with t.inference_mode():
            accuracy = 0
            total = 0
            for (x, y) in testloader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                y_flat = rearrange(y, "batch seq -> (batch seq)")
                y_pred_flat = rearrange(y_hat, "batch seq vocab_size -> (batch seq) vocab_size")
                y_predictions = y_pred_flat.argmax(-1)
                accuracy += (y_predictions == y_flat).sum().item()
                total += y_flat.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

#%% [markdown]
#### Train model
#%%
# sweep_config = {
#     'method': 'bayes',
#     'name': 'mlk_sweep',
#     'metric': {'name': 'train_loss', 'goal': 'minimize'},
#     'parameters': 
#     {
#         'batch_size': {'values': [64]},
#         'hidden_size': {'values': [512]},
#         'max_seq_len': {'values': [60]},
#         'lr': {'values': [.001]},
#         'dropout': {'values': [.1]},
#         'epochs': {'values': [2]},
#         'num_layers': {'values': [6]},
#         'num_heads': {'values': [8]},
#      }
# }
# sweep_id = wandb.sweep(sweep=sweep_config, project='w1d3_mlk')
# wandb.agent(sweep_id=sweep_id, function=train_mlk, count=1)
#%%
train_mlk()

#%% [markdown]
### Load model
#%%
run_id = '30cv3o0n'
root = '/home/oskar/projects/arena-v1-ldn/w1d3/wandb'
model_path = glob.glob(
    f'{root}/run-*-{run_id}/files/model_state_dict.pt'
)[0]
yaml_path = glob.glob(
    f'{root}/run-*-{run_id}/files/config.yaml'
)[0]
with open(yaml_path, 'r') as f:
    yaml_cfg = yaml.safe_load(f)
#%%
base_config = transformer_replication.TransformerConfig(
    num_layers=yaml_cfg['num_layers']['value'],
    num_heads=yaml_cfg['num_heads']['value'],
    vocab_size=len(mlk.vocab),
    hidden_size=yaml_cfg['hidden_size']['value'],
    max_seq_len=yaml_cfg['max_seq_len']['value'],
    dropout=yaml_cfg['dropout']['value'],
)
mlk.model_max_length = yaml_cfg['max_seq_len']['value']
model = transformer_replication.DecoderOnlyTransformer(base_config)
state_dict = t.load(
    model_path
)
model.load_state_dict(state_dict)

#%%
text_output = sampling.sample_tokens(model, mlk, " I pray that one day we might ", max_tokens_generated=300, temperature=1.0, top_k=10)
print(text_output)
# %%
# TODO: 
# why do we need to surround with spaces?
# why is bigger seq_len better?
# why do some models seem to get stuck in local minima?
# how can we include freq_penalty when punctuation are tokens?
# why is accuracy < 50%?