#%%
import torch as t
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
from einops import rearrange
from fancy_einsum import einsum
import numpy as np
# %%
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

def multihead_masked_attention(
    Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int, dropout: float, 
    additive_attention_mask: Optional[t.Tensor]
):
    '''
    Implements multihead masked attention on the matrices Q, K and V.

    x: shape (batch, seq)
    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)

    returns: shape (batch, seq, nheads*headsize)
    '''
    new_Q = rearrange(
        Q, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads
    )
    new_K = rearrange(
        K, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads
    )
    new_V = rearrange(
        V, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads
    )
    batches, _, seq_Q, head_size = new_Q.shape
    einsum_eq = (
        'batches nheads seq_Q head_size, '
        'batches nheads seq_K head_size -> '
        'batches nheads seq_Q seq_K'
    )
    attention_scores = einsum(einsum_eq, new_Q, new_K)
    attention_scores /= np.sqrt(head_size)
    if additive_attention_mask is not None:
        attention_scores += additive_attention_mask
    attention_probabilities = nn.functional.softmax(
        attention_scores, dim=-1
    )
    dropped_probabilities = nn.functional.dropout(attention_probabilities, dropout)
    attention_values = einsum(
        'batches nheads seq_Q seq_K, batches nheads seq_K head_size -> '
        'batches seq_Q nheads head_size', 
        dropped_probabilities, 
        new_V
    )
    attention_rearranged = rearrange(
        attention_values, 
        'batches seq_Q nheads head_size -> batches seq_Q (nheads head_size)'
    )
    return attention_rearranged

class MultiheadAttention(nn.Module):

    def __init__(self, config: TransformerConfig, mult_hidden=3):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        assert self.hidden_size % self.num_heads == 0
        self.W_QKV = nn.Linear(config.hidden_size, mult_hidden * config.hidden_size, bias=True)
        self.W_O = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        QKV = self.W_QKV(x)
        Q = QKV[..., :self.hidden_size]
        K = QKV[..., self.hidden_size:-self.hidden_size]
        V = QKV[..., -self.hidden_size:]
        attention_values = multihead_masked_attention(
            Q, K, V, self.num_heads, self.dropout, additive_attention_mask, 
        )
        attention_times_o = self.W_O(attention_values)
        attention_dropped = nn.functional.dropout(attention_times_o, self.dropout)
        return attention_dropped


class BracketTokens:
    CLS = 0
    SEP = 1
    PAD = 2
    LEFT = 3
    RIGHT = 4


class SpecialTokens:
    CLS = '[CLS]'
    SEP = '[SEP]'
    PAD = '[PAD]'
    LEFT = '{'
    RIGHT = '}'

class BracketTokenizer:

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def tokenize(self, s: str) -> List[str]:
        tokens = ['[CLS]'] + s.split() + ['[SEP]']
        assert len(tokens) <= self.seq_len
        if len(tokens) < self.seq_len:
            tokens += ['[PAD]'] * (self.seq_len - len(tokens))
        return tokens

    def encode(self, s: str) -> int:
        if s == SpecialTokens.CLS:
            return BracketTokens.CLS
        elif s == SpecialTokens.SEP:
            return BracketTokens.SEP
        elif s == SpecialTokens.PAD:
            return BracketTokens.PAD
        elif s == SpecialTokens.LEFT:
            return BracketTokens.LEFT
        elif s == SpecialTokens.RIGHT:
            return BracketTokens.RIGHT
        else:
            raise NotImplementedError(f'BracketTokenizer.encode str={s}')

    def decode(self, i: int) -> str:
        if i == BracketTokens.CLS:
            return SpecialTokens.CLS
        elif i == BracketTokens.SEP:
            return SpecialTokens.SEP
        elif i == BracketTokens.PAD:
            return SpecialTokens.PAD
        elif i == BracketTokens.LEFT:
            return SpecialTokens.LEFT
        elif BracketTokens.RIGHT:
            return SpecialTokens.RIGHT
        else:
            raise NotImplementedError(f'BracketTokenizer.decode token={i}')

    def batch_decode(self, tokens: List[int]) -> str:
        return ''.join([self.decode(token) for token in tokens])

    def batch_encode(self, s: str) -> List[int]:
        tokens = self.tokenize(s)
        codes = [self.encode(token) for token in tokens]
        return codes