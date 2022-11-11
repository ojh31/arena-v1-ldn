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
from fancy_einsum import einsum
from typing import Union, Optional, Callable, Tuple
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb
# %%
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
        return self.weight[x]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


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
import attention_replication

class BertMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=True)
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
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.attention = attention_replication.MultiheadMaskedAttention(
            config.hidden_size, config.num_heads, config.dropout
        )
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = BertMLP(config)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        y = self.layer_norm1(x)
        y = self.attention(y)
        x = x + y
        z = self.layer_norm2(x)
        z = self.mlp(z)
        x = x + z
        return x


class GPT(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.bert_blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        pos = t.arange(x.shape[1], device=x.device)
        x = self.token_embedding(x) + self.positional_embedding(pos)
        x = self.dropout(x)
        for block in self.bert_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = einsum(
            'num_embeddings embedding_dim,batch seq_len embedding_dim ->batch seq_len num_embeddings', 
            self.token_embedding.weight, 
            x
        )
        return x

# %%
