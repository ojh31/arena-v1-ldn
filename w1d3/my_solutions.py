#%%
import torch as t
import torch.nn.functional as F
import transformers

gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
# %%
def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)

#%%
def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.int64, device=device)
        new_input_ids_truncated = new_input_ids[
            -min(tokenizer.model_max_length, new_input_ids.shape[0]):
        ].unsqueeze(0)
        output = model(new_input_ids_truncated)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)

#%%
def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    most_likely = logits.argmax().item()
    assert isinstance(most_likely, int)
    return most_likely

prompt = "Jingle bells, jingle bells, jingle all the way"
print("Greedy decoding with prompt: ", prompt)
output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output}")
expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Greedy decoding a second time (should be deterministic): ")
output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output}")
expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Tests passed!")
# %%
def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    dist = t.distributions.categorical.Categorical(logits=logits)
    return dist.sample().item()

N = 20000
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, probs, atol=0.01, rtol=0)
print("Tests passed!")
# %%
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    return logits / temperature

logits = t.tensor([1, 2]).log()
cold_logits = apply_temperature(logits, 0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)
hot_logits = apply_temperature(logits, 1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)
print("Tests passed!")
# %%
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''

    counts = input_ids.bincount()
    counts_broadcasted = t.zeros_like(logits)
    counts_broadcasted[:len(counts)] = counts
    return logits - freq_penalty * counts_broadcasted

bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt").squeeze()
logits = t.ones(tokenizer.vocab_size)
penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
print("Tests passed!")
# %%
N_RUNS = 1
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(freq_penalty=100.0)),
    ("Negative freq penalty", dict(freq_penalty=-1.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]
for (name, kwargs) in cases:
    for i in range(N_RUNS):
        output = sample_tokens(gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
        print(f"Sample {i} with: {name} ({kwargs}):")
        print(f"Your model said: {repr(output)}\n")
# %%
import numpy as np
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    topk_obj = logits.topk(top_k)
    masked_logits = t.full_like(logits, -np.inf)
    masked_logits[topk_obj.indices] = logits[topk_obj.indices]
    return sample_basic(masked_logits)

k = 3
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[:-k] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0)
print("Tests passed!")
# %%
your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
print(f"Your model said: {repr(output)}")
# %%
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    sorted_idx = logits.argsort(descending=True)
    sorted_logits = logits[sorted_idx].double()
    sorted_probs = t.exp(sorted_logits)
    sorted_probs /= sorted_probs.sum()
    sorted_cumsums = sorted_probs.cumsum(dim=0)
    first_idx = t.where(sorted_cumsums >= top_p)[0][0].item()
    tokens_to_keep = max(first_idx + 1, min_tokens_to_keep)
    keep_idx = list(range(tokens_to_keep))
    keep_reindexed = sorted_idx[keep_idx]
    masked_logits = t.full_like(logits, -t.inf)
    masked_logits[keep_reindexed] = logits[keep_reindexed]
    # print(logits, top_p, min_tokens_to_keep)
    # print(sorted_idx, sorted_logits)
    # print(sorted_probs, sorted_cumsums)
    # print(first_idx, tokens_to_keep, keep_idx, keep_reindexed)
    # print(masked_logits)
    return sample_basic(masked_logits)

#%%
N = 2000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p of 0.5 or lower should only return token 2: ", counts)
assert counts[0] == 0 and counts[1] == 0

#%%
N = 2000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
assert counts[0] == 0

#%%
N = 40_000
top_p = 0.71
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[0:2] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0.0)

print("All tests passed!")
# %%
your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
print(f"Your model said: {repr(output)}")
# %%
#### Shakespeare
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import urllib.request
import re
from typing import Union, Optional
#%%
class WordsTokenizer():
    model_max_length: int

    def __init__(self, text: str):
        self.text = text
        vocab = set()
        model_max_length = 0
        for input in self.split(text):
            vocab.update(set(input))
            model_max_length = max(model_max_length, len(input))
        self.model_max_length = model_max_length
        self.vocab = list(sorted(list(vocab)))
        self.token_to_str = {i: t for i, t in enumerate(self.vocab)}
        self.str_to_token = {t: i for i, t in enumerate(self.vocab)}
        
    @staticmethod
    def split(text: str):
        return re.split(r"\b", text)

    @staticmethod
    def join(l: list[str]):
        return r"\b".join(l)

    def encode(self, initial_text: str = None, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        if initial_text is None:
            initial_text = self.text
        text_split = self.split(initial_text)
        token_list = [self.str_to_token[t] for t in text_split]
        if return_tensors == 'pt':
            return t.tensor(token_list)
        else:
            return token_list

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        list_of_ids = [t.item() for t in list_of_ids]
        text_list = [self.token_to_str[t] for t in list_of_ids]
        text_join = self.join(text_list)
        return text_join

#%%
class WordsDataset(Dataset):
    def __init__(self, text):
        self.text = text
        self.tokenizer = WordsTokenizer(text)
        tokens = self.tokenizer.encode()
        self.inputs = [tokens[i: i + seq_len] for i in range(len(tokens) - seq_len)]
        self.labels = [tokens[i + 1: i + 1 + seq_len] for i in range(len(tokens) - seq_len)]

    @staticmethod
    def from_html(seq_len, first, last, url="https://www.gutenberg.org/files/100/100-0.txt"):
        with urllib.request.urlopen(url) as f:
            full_text = f.read()
        sonnets = full_text.decode('utf-8').split('THE SONNETS')[-1].split(
            'ALL’S WELL THAT ENDS WELL'
        )[0]
        subset = sonnets.split(str(first), maxsplit=1)[1]
        subset = subset.split(str(last), maxsplit=1)[0]
        return WordsDataset(subset)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.inputs[idx]
        sample = (text, label)
        return sample

#%%
seq_len = 20
batch_size = 16
trainset = WordsDataset.from_html(seq_len, first=1, last=120)
testset = WordsDataset.from_html(seq_len, first=121, last=154)

trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)


#### From yesterday
#%%
from einops import rearrange, repeat
import torch.nn as nn
from tqdm import tqdm_notebook
from fancy_einsum import einsum


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(t.randn((self.num_embeddings, self.embedding_dim)))

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''
        #return einsum('num_embeddings embedding_dim, i num_embeddings -> i embedding_dim', self.weight, nn.functional.one_hot(x, num_classes=self.num_embeddings).float())
        return self.weight[x]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        # Defining our positional encoding array, with `max_seq_len` rows
        # This is an advantage of using sinusoidal encoding: we can easily expand to sequences of greater length without adding more learned params
        angles = t.outer(t.arange(max_seq_len), 1 / 10000 ** (2 * t.arange(embedding_dim//2) / embedding_dim))
        pe = t.zeros((max_seq_len, embedding_dim))
        pe[:, ::2] = t.sin(angles)
        pe[:, 1::2] = t.cos(angles)
        # Register array as a buffer, rather than parameter (we don't want it to be updated by gradient descent)
        self.register_buffer('pe', pe)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq_len, embedding_dim)
        """
        batch, seq_len, embedding_dim = x.shape
        # We slice the positional encoding, so it's the same shape as x
        # This is equivalent to just using an nn.Embedding, but having the input be t.arange(seq_len)
        return x + self.pe[:seq_len, :] # type: ignore


def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)

    returns: shape (batch, seq, nheads*headsize)
    '''
    new_Q = rearrange(Q, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)
    new_K = rearrange(K, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)
    new_V = rearrange(V, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)

    attention_scores = einsum('batches nheads seq_Q head_size, batches nheads seq_K head_size -> batches nheads seq_Q seq_K', new_Q, new_K)
    batches, _, seq_Q, head_size = new_Q.shape
    batches, _, seq_K, head_size = new_K.shape
    q_index = repeat(t.arange(0, seq_Q), 'seq_Q -> batches nheads seq_Q seq_K', batches=batches, seq_K=seq_K, nheads=num_heads)
    k_index = repeat(t.arange(0, seq_K), 'seq_K -> batches nheads seq_Q seq_K', batches=batches, seq_Q=seq_Q, nheads=num_heads)
    mask = k_index <= q_index
    device_inf = t.tensor(-np.inf).to(Q.device)
    device_mask = mask.to(Q.device)
    masked_attention_scores = t.where(device_mask, attention_scores, device_inf)
    attention_probabilities = nn.functional.softmax(masked_attention_scores / np.sqrt(head_size), dim=-1)
    attention_values = einsum('batches nheads seq_Q seq_K, batches nheads seq_K head_size -> batches seq_Q nheads head_size', attention_probabilities, new_V)
    return rearrange(attention_values, 'batches seq_Q nheads head_size -> batches seq_Q (nheads head_size)')


class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0
        self.W_QKV = nn.Linear(hidden_size, 3 * hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        QKV = self.W_QKV(x)
        Q = QKV[..., :self.hidden_size]
        K = QKV[..., self.hidden_size:-self.hidden_size]
        V = QKV[..., -self.hidden_size:]
        attention_values = multihead_masked_attention(Q, K, V, self.num_heads)
        return self.W_O(attention_values)
#%%
#### Putting together the transformer
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



# %%
class BertMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = BertMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:
        y = self.attention(x)
        y = self.layer_norm1(y)
        x = x + y
        z = self.mlp(x)
        z = self.layer_norm2(z)
        x = x + z
        return x

class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = PositionalEncoding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.bert_blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.token_embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        for block in self.bert_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = einsum('num_embeddings embedding_dim,batch seq_len embedding_dim ->batch seq_len num_embeddings', self.token_embedding.weight, x)
        return x


#### Run
import wandb
import os
import time
device = t.device('cuda')
os.environ['WANDB_NOTEBOOK_NAME'] = 'my_solutions.py'
def train():

    wandb_config_dict = {
        'batch_size': 256,
        'hidden_size': 64,
        'lr': 0.00125
    }
    
    wandb.init(project='w1d1_transformer', config=wandb_config_dict)

    config = TransformerConfig(
        num_layers=2, #N=6
        num_heads=4, #h=8
        vocab_size=10,
        hidden_size=wandb.config.hidden_size, #d_model = 64 x 8 = 512
        max_seq_len=6,
        dropout=0.0 #p=0.1
    )

    epochs = 1
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = DecoderOnlyTransformer(config).to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    examples_seen = 0
    start_time = time.time()

    trainset = WordsDataset.from_html(seq_len=seq_len)
    testset = WordsDataset.from_html(seq_len=seq_len)

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
    return model


#%%
model = train()
# %%
