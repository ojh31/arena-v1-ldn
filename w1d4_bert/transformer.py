#%%
import attention
from dataclasses import dataclass
import torch.nn as nn
import torch as t
from fancy_einsum import einsum

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
        self.attention = attention.MultiheadMaskedAttention(config.hidden_size, config.num_heads)
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
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
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
        x = einsum('num_embeddings embedding_dim,batch seq_len embedding_dim ->batch seq_len num_embeddings', self.token_embedding.weight, x)
        return x