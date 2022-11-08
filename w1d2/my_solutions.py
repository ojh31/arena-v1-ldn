#%% 
import transformers
import torch as t
import torch.nn as nn
from typing import Union, List
from fancy_einsum import einsum
import torch as t
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Union, Optional, Callable, Tuple
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb
#### Old stuff
# %%
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

# %%
#TODO replace positional encoding
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


# %%
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(normalized_shape))
            self.bias = nn.Parameter(t.zeros(normalized_shape))

    def forward(self, x: t.Tensor) -> t.Tensor:
        normalized_shape_dims = 1 if isinstance(self.normalized_shape, int) else len(self.normalized_shape)
        x_mean = x.mean(dim=list(range(x.dim()))[-normalized_shape_dims:], keepdim=True) # complement of the normalised shape
        x_var = x.var(dim=list(range(x.dim()))[-normalized_shape_dims:], keepdim=True, unbiased=False) # complement of the normalised shape
        x_scaled = (x - x_mean) / t.sqrt(x_var + self.eps)
        if self.elementwise_affine:
            return x_scaled * self.weight + self.bias
        return x_scaled

    def extra_repr(self) -> str:
        pass


#### Attention

# %%
import torch as t
import torch.nn as nn
from typing import Union, List
from fancy_einsum import einsum
from einops import repeat, rearrange, reduce
import numpy as np
#%%
def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (batches x seq_Q x head_size)
    K: shape (batches x seq_K x head_size)
    V: shape (batches x seq_K x head_size)

    Return: shape (batches x seq_Q x head_size)
    '''
    
    attention_scores = einsum('batches seq_Q head_size, batches seq_K head_size -> batches seq_Q seq_K', Q, K)
    #Ignore masking
    attention_probabilities = nn.functional.softmax(attention_scores / np.sqrt(Q.shape[-1]), dim=2)
    attention_values = einsum('batches seq_Q seq_K, batches seq_K head_size -> batches seq_Q head_size', attention_probabilities, V)
    return attention_values

def test_single_head_attention_shape(single_head_attention):
    Q = t.randn(1, 3, 2)
    K = t.randn(1, 5, 2)
    V = t.randn(1, 5, 2)
    attention_values = single_head_attention(Q, K, V)
    assert Q.shape == attention_values.shape
    print(f"All tests in `test_single_head_attention_shape` passed.")

def test_single_head_attention(single_head_attention):
    Q = t.tensor([[[7, 4, 1], [6, 3, 0], [5, 2, 1]]])
    K = t.tensor([[[1, 3, 5], [2, 4, 6]]])
    V = t.tensor([[[1, 0, 1], [0, 1, 0]]])
    attention_values = single_head_attention(Q.float(), K.float(), V.float())
    t.testing.assert_close(attention_values, t.tensor([[[9.7880e-04, 9.9902e-01, 9.7880e-04], [5.5073e-03, 9.9449e-01, 5.5073e-03], [9.7682e-03, 9.9023e-01, 9.7682e-03]]]), rtol=0.01, atol=0.001)
    print(f"All tests in `test_single_head_attention` passed.")
    
test_single_head_attention_shape(single_head_attention)
test_single_head_attention(single_head_attention)
# %%
def single_head_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of masked self-attention.

    See "The Decoder Side" section of the Illustrated Transformer for an explanation of masking.

    Q: shape (batches x seq_Q x head_size)
    K: shape (batches x seq_K x head_size)
    V: shape (batches x seq_K x head_size)

    Return: shape (batches x seq_Q x head_size)
    '''
    attention_scores = einsum('batches seq_Q head_size, batches seq_K head_size -> batches seq_Q seq_K', Q, K)
    batches, seq_Q, head_size = Q.shape
    batches, seq_K, head_size = K.shape

    q_index = repeat(t.arange(0, seq_Q), 'q -> b q k', b=batches, k=seq_K)
    k_index = repeat(t.arange(0, seq_K), 'k -> b q k', b=batches, q=seq_Q)
    mask = k_index <= q_index
    attention_scores = t.where(mask, attention_scores, -t.inf)
    attention_probabilities = nn.functional.softmax(attention_scores / np.sqrt(Q.shape[-1]), dim=2)
    attention_values = einsum('batches seq_Q seq_K, batches seq_K head_size -> batches seq_Q head_size', attention_probabilities, V)
    return attention_values

def test_single_head_masked_attention(single_head_masked_attention):
    Q = t.tensor([[[7, 4, 1], [6, 3, 0], [5, 2, 1]]])
    K = t.tensor([[[1, 3, 5], [2, 4, 6]]])
    V = t.tensor([[[1, 0, 1], [0, 1, 0]]])
    attention_values = single_head_masked_attention(Q.float(), K.float(), V.float())
    t.testing.assert_close(attention_values, t.tensor([[[1, 0, 1], [5.5073e-03, 9.9449e-01, 5.5073e-03], [9.7682e-03, 9.9023e-01, 9.7682e-03]]]), rtol=0.01, atol=0.001)
    print(f"All tests in `test_single_head_attention` passed.")

test_single_head_attention_shape(single_head_masked_attention)
test_single_head_masked_attention(single_head_masked_attention)
# %%
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
    masked_attention_scores = t.where(mask, attention_scores, -t.inf)
    attention_probabilities = nn.functional.softmax(masked_attention_scores / np.sqrt(head_size), dim=-1)
    attention_values = einsum('batches nheads seq_Q seq_K, batches nheads seq_K head_size -> batches seq_Q nheads head_size', attention_probabilities, new_V)
    return rearrange(attention_values, 'batches seq_Q nheads head_size -> batches seq_Q (nheads head_size)')

def test_multihead_masked_attention(multihead_masked_attention):
    Q = t.tensor([[[7, 4, 1], [6, 3, 0], [5, 2, 1]]])
    K = t.tensor([[[1, 3, 5], [2, 4, 6]]])
    V = t.tensor([[[1, 0, 1], [0, 1, 0]]])
    attention_values = multihead_masked_attention(Q.float(), K.float(), V.float(), num_heads=1)
    t.testing.assert_close(attention_values, t.tensor([[[1, 0, 1], [5.5073e-03, 9.9449e-01, 5.5073e-03], [9.7682e-03, 9.9023e-01, 9.7682e-03]]]), rtol=0.01, atol=0.001)
    print(f"All tests in `test_multihead_masked_attention` passed.")  

test_multihead_masked_attention(multihead_masked_attention)
# %%
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
# %%
def test_MultiheadMaskedAttention_shape(MultiheadMaskedAttention):
    mma = MultiheadMaskedAttention(1, 1)
    x = t.randn(2, 7, 1)
    output = mma.forward(x)
    assert x.shape == output.shape
    print(f"All tests in `test_MultiheadMaskedAttention_shape` passed.")

test_MultiheadMaskedAttention_shape(MultiheadMaskedAttention)
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

# %%
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    @staticmethod
    def from_config(config, samples):
        texts = [t.randint(high=config.vocab_size, size=(config.max_seq_len,)) for _ in range(samples)]
        labels = [t.flip(text, (0,)) for text in texts]
        return CustomTextDataset(texts, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        sample = (text, label)
        return sample

# %%
import wandb
import os
device = t.device('cpu')
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
    train_samples = 500000
    test_samples = 1000

    model = DecoderOnlyTransformer(config).to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    examples_seen = 0
    start_time = time.time()

    trainset = CustomTextDataset.from_config(config, train_samples)
    testset = CustomTextDataset.from_config(config, test_samples)

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

#%%
# sweep_config = {
#     'method': 'bayes',
#     'name': 'w1d1_transformer',
#     'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
#     'parameters': 
#     {
#         'batch_size': {'values': [32]},
#         'hidden_size': {'values': [64]},
#         'lr': {'max': 0.2, 'min': 0.0001, 'distribution': 'log_uniform_values'}
#      }
# }

# sweep_id = wandb.sweep(sweep=sweep_config, project='w1d1_transformer')

# wandb.agent(sweep_id=sweep_id, function=train, count=1)

# %%
print(model(t.tensor([[1, 2, 3, 4, 5, 6]])).argmax(-1))
print(model(t.tensor([[1, 2, 3, 4, 6, 6]])).argmax(-1))
print(model(t.tensor([[1, 2, 3, 4, 8, 6]])).argmax(-1))
print(model(t.tensor([[4, 4, 8, 4, 9, 6]])).argmax(-1))
# %%
# 981cb292a2f387e5609b50b05f9acf9f4a2e4fb8