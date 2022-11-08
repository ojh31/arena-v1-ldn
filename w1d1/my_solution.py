#%%
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
#%%
tokenizer("Hello world")['input_ids']

#%%
tokenizer.tokenize('Hello world, I am Oskar')

#%%
tokenizer.encode("Hello world, I am Oskar")
#%%
tokenizer.decode([15496, 995, 11, 314])
#%%
import torch as t
import torch.nn as nn
#%%
# vocab size 20 and embedding dim 3
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = t.LongTensor([[1,2,4,5],[4,3,2,9]])
embedding(input)
#%%
embedding(input).shape

#%%
input.shape

#%%
import torch
a = torch.nn.Embedding(10, 50)
b = torch.LongTensor([2,8])
results = a(b)

def get_embedding_index(x):
    results = torch.where(torch.sum((a.weight==x), axis=1))
    if len(results[0])==len(x):
        return None
    else:
        return results[0][0]

indices = torch.Tensor(list(map(get_embedding_index, results)))
indices
# %%
import utils
class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = t.normal(
            mean=t.Tensor([0]).repeat(num_embeddings, embedding_dim),
            std=t.Tensor([1]).repeat(num_embeddings, embedding_dim)
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''
        For each integer in the input, return that row of the embedding.
        x.shape: batch, seq len
        '''
        return self.weight[x]

    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}'

assert repr(Embedding(10, 20)) == repr(t.nn.Embedding(10, 20))
utils.test_embedding(Embedding)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pos_enc(pos, d, D, base=10_000):
    """
    pos: token position to be encoded
    d: component in embedding dimension
    D: size of embedding dimension
    base: used for sinusoidal frequency
    """
    return np.where(
        d % 2 == 0,
        np.sin(
            pos / 
            (base ** (d / D))
        ),
        np.cos(
            pos / 
            (base ** ((d - 1) / D))
        )
    )

x, y =  np.mgrid[0:32, 0:128]
z = pos_enc(x, y, D=128)

sns.heatmap(z, cmap='coolwarm_r')
# %%
import itertools
dots = np.zeros((32, 32))
for p1, p2 in itertools.product(range(32), repeat=2):
    
    p1_vec = pos_enc(p1, np.arange(128), D=128)
    p2_vec = pos_enc(p2, np.arange(128), D=128)
    dot = np.dot(p1_vec, p2_vec) / (np.linalg.norm(p1_vec) * np.linalg.norm(p2_vec))
    dots[p1, p2] = dot

sns.heatmap(dots, cmap='coolwarm_r')

#%%
from einops import rearrange, repeat, reduce
class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len # same as base above
        self.embedding_dim = embedding_dim # same as D above

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq_len, embedding_dim)
        """
        batch, seq_len, embedding_dim = x.shape
        embedding_idx = repeat(t.arange(embedding_dim), 'e -> b s e', s=seq_len, b=batch)
        pos_idx = repeat(t.arange(seq_len), 's -> b s e', e=embedding_idx, b=batch)
        return t.where(
            embedding_idx % 2 == 0,
            t.sin(pos_idx / t.pow(seq_len, embedding_idx / embedding_dim)),
            t.cos(pos_idx / t.pow(seq_len, (embedding_idx - 1) / embedding_dim))
        )

#%%
# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)* 2 + 1
layer_norm = nn.LayerNorm(embedding_dim)
# Activate module
nlp_out = layer_norm(embedding)
layer_norm.weight.shape, layer_norm.bias.shape, nlp_out[0, 0, :].mean(), nlp_out[0, 0, :].std()

#%%
# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W) * 2 + 1
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])
img_out = layer_norm(input)
layer_norm.weight.shape, layer_norm.bias.shape, img_out[0, ...].mean(), img_out[0, ...].std()
# %%
from typing import Union, List
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = t.ones(normalized_shape)
        self.bias = t.zeros(normalized_shape)
        if elementwise_affine:
            self.weight = nn.Parameter(self.weight, requires_grad=True)
            self.bias = nn.Parameter(self.bias, requires_grad=True)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def forward(self, x: t.Tensor) -> t.Tensor:
        if isinstance(self.normalized_shape, int):
            dims = [-1]
        else:
            n_dims = len(self.normalized_shape) 
            dims = [-i - 1 for i in range(n_dims)]
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_centred = x - mean
        std_err = t.sqrt(var + self.eps)
        x_scaled = x_centred / std_err
        return self.weight * x_scaled + self.bias

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affinew}'

utils.test_layernorm_mean_1d(LayerNorm)
utils.test_layernorm_mean_2d(LayerNorm)
utils.test_layernorm_std(LayerNorm)
utils.test_layernorm_exact(LayerNorm)
utils.test_layernorm_backward(LayerNorm)
# %%
class Dropout(nn.Module):

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training:
            probs = torch.rand(x.shape)
            return x * (probs > self.p) / (1 - self.p)
        return x

    def extra_repr(self) -> str:
        return f'p={self.p}'

utils.test_dropout_eval(Dropout)
utils.test_dropout_training(Dropout)
# %%
from scipy.stats import norm


class GELU(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x * norm.cdf(x)


class GELU2(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        tanh_arg = np.sqrt(2 / np.pi) * (x + .044715 * t.pow(x, 3))
        return 0.5 * x * (1 + t.tanh(tanh_arg))


class GELU3(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x * (1 / (1+ t.exp(-1.702 * x)))

class Swish(nn.Module):

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x / (1 + t.exp(-self.beta * x))


utils.plot_gelu(GELU)
# utils.plot_gelu(GELU2)
#
# %%
x = t.linspace(-5, 5, steps=100)
fig, ax = plt.subplots()
sns.lineplot(x=x, y=GELU()(x), label='exact', color='tab:blue')
sns.lineplot(x=x, y=GELU2()(x), label='approx', color='tab:orange')
sns.lineplot(x=x, y=GELU3()(x), label='approx2', color='tab:green')
sns.lineplot(x=x, y=Swish(1)(x), label='swish1', color='tab:red')
# %%
