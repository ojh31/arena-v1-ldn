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
with open('100-0.txt', encoding='utf-8') as f:
    shakespeare_text = f.read()
# %%
def tokenize(text):
    return re.split(r"\b", text)

def remove_spaces(text):
    return re.sub('\s+', ' ', text)

# %%
class Data():
    def __init__(self, text):
        self.complete_text = remove_spaces(text)
        self.complete_tokens = tokenize(self.complete_text)
        self.vocab = sorted(set(self.complete_tokens))
        self.token_to_id = dict(zip(self.vocab, list(range(len(self.vocab)))))
        self.id_to_token = dict(zip(list(range(len(self.vocab))), self.vocab))
        self.model_max_length = None

    def get_excerpt(self, start="THE SONNETS", end="THE END", text=None):
        if text is None:
            text = self.complete_text
        return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]

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

shakespeare = Data(shakespeare_text)


#%%

def train() -> transformer_replication.DecoderOnlyTransformer:
    # wandb_config_dict = {
    #    'batch_size': 64,
    #    'hidden_size': 64,
    #    'lr': 0.001,
    #    'epochs': 5,
    #    'max_seq_len': 20,
    #    'dropout': 0.1,
    # }

    wandb.init() # project='w1d3_shakespeare'

    transformer_config = transformer_replication.TransformerConfig(
        num_layers=wandb.config.num_layers, #N=6
        num_heads=wandb.config.num_heads, #h=8
        vocab_size=len(shakespeare.vocab),
        hidden_size=wandb.config.hidden_size, #d_model = 64 x 8 = 512
        max_seq_len=wandb.config.max_seq_len,
        dropout=wandb.config.dropout #p=0.1
    )

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = transformer_replication.DecoderOnlyTransformer(transformer_config).to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    examples_seen = 0
    start_time = time.time()

    traintext = shakespeare.get_excerpt("From fairest", "140")
    testtext = shakespeare.get_excerpt("140", "THE END")

    trainset = shakespeare.generate_autoregressive_dataset(transformer_config.max_seq_len, traintext)
    testset = shakespeare.generate_autoregressive_dataset(transformer_config.max_seq_len, testtext)

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

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

#%%
sweep_config = {
    'method': 'bayes',
    'name': 'shakespeare_sweep',
    'metric': {'name': 'train_loss', 'goal': 'minimize'},
    'parameters': 
    {
        'batch_size': {'values': [64]},
        'hidden_size': {'values': [64, 128, 256]},
        'max_seq_len': {'values': [20, 30, 40]},
        'lr': {'values': [.001]},
        'dropout': {'values': [.1]},
        'epochs': {'values': [2]},
        'num_layers': {'values': 6},
        'num_heads': {'values': 8},
     }
}
sweep_id = wandb.sweep(sweep=sweep_config, project='w1d3_shakespeare')
wandb.agent(sweep_id=sweep_id, function=train, count=10)
#%%
# model = train()
# %%

# print(wandb.run.dir)
base_config = transformer_replication.TransformerConfig(
    num_layers=6,
    num_heads=8,
    vocab_size=len(shakespeare.vocab),
    hidden_size=128,
    max_seq_len=40,
    dropout=0.1,
)
model = transformer_replication.DecoderOnlyTransformer(base_config)
state_dict = t.load("/home/oskar/projects/arena-v1-ldn/w1d3/wandb/dazzling_sweep_7.pt")
model.load_state_dict(state_dict)

#%%
text_output = sampling.sample_tokens(model, shakespeare, " I sang a wonderful song ", max_tokens_generated=100, temperature=1.0, top_k=10)
print(text_output)
# %%
# TODO: 
# why do we need to surround with spaces?