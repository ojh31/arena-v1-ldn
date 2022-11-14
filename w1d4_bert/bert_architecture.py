#%%
from dataclasses import dataclass
import torch.nn as nn
import torch as t
from fancy_einsum import einsum
from einops import repeat, rearrange, reduce
from typing import Optional, Tuple
import numpy as np
import utils


def make_additive_attention_mask(
    one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000
) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 
        0.0 for the floating point precision used.

    Out: 
        shape (batch, nheads=1, seqQ=1, seqK)
        Contains 0 if attention is allowed, big_negative_number if not.
    '''
    assert isinstance(one_zero_attention_mask, t.Tensor), (
        f'one_zero_attention_mask={one_zero_attention_mask}'
    )
    is_float = isinstance(big_negative_number, float) 
    is_int = isinstance(big_negative_number, int)
    assert is_float or is_int, (
        f'big_negative_number ={big_negative_number}'
    )
    filled = t.where(
        one_zero_attention_mask == 0, 
        big_negative_number,
        0,
    )
    reshaped = repeat(filled, 'b s -> b n_heads seq_q s', n_heads=1, seq_q=1)
    return reshaped

#%%
utils.test_make_additive_attention_mask(make_additive_attention_mask)

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

    attention_scores = einsum(
        'batches nheads seq_Q head_size, batches nheads seq_K head_size -> '
        'batches nheads seq_Q seq_K', 
        new_Q, 
        new_K,
    )
    batches, _, seq_Q, head_size = new_Q.shape
    if additive_attention_mask is not None:
        attention_scores += additive_attention_mask
    attention_probabilities = nn.functional.softmax(
        attention_scores / np.sqrt(head_size), dim=-1
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


#%%
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
        if additive_attention_mask is not None:
            print(f'x.shape={x.shape}, mask.shape={additive_attention_mask.shape}')
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

# %%
class BertMLP(nn.Module):
    # Unchanged from Shakespeare model
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

#%%
class BertBlock(nn.Module):
    # Layer norms moved after add
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiheadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = BertMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)

    def forward(
        self, x: t.Tensor, additive_attention_mask: t.Tensor
    ) -> t.Tensor:
        print(f'Calling attention')
        att = self.attention(x, additive_attention_mask)
        print('Calling layer_norm1')
        att_sum = self.layer_norm1(x + att)
        print('Calling mlp...')
        mlp = self.mlp(att_sum)
        print('Calling layer_norm2...')
        mlp_sum = self.layer_norm2(att_sum + mlp)
        return mlp_sum


class BertCommon(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.positional_embedding = nn.Embedding(
            config.max_seq_len, config.hidden_size
        )
        self.segment_embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)
        self.bert_blocks = nn.Sequential(
            *[BertBlock(config) for _ in range(config.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)

    def forward(
        self,
        x: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        '''
        input_ids: (batch, seq) - the token ids
        one_zero_attention_mask: 
        (batch, seq) - only used in training, passed to `make_additive_attention_mask` and 
        used in the attention blocks.
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        assert isinstance(x, t.Tensor)
        pos = t.arange(x.shape[1], device=x.device)
        if one_zero_attention_mask is None:
            additive_attention_mask = None
        else:
            print('Calling make_additive_attention_mask')
            additive_attention_mask = make_additive_attention_mask(
                one_zero_attention_mask
            )
        print('Calling token_embedding')
        embedding_sum =  self.token_embedding(x)
        print('Calling positional_embedding...')
        embedding_sum += self.positional_embedding(pos)
        if token_type_ids is not None:
            print('Calling segment_embedding')
            embedding_sum += self.segment_embedding(token_type_ids)
        print('Finished embedding sum')
        x = self.layer_norm(embedding_sum)
        x = self.dropout(x)
        print('Starting bert blocks...')
        for block in self.bert_blocks:
            x = block(x, additive_attention_mask)
        return x


class BertLanguageModel(nn.Module):

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.common = BertCommon(config=config)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.layer = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.unembed_bias = nn.Parameter(t.randn(config.vocab_size))

    def tied_unembed(self, x: t.Tensor):
        return einsum(
            'num_embeddings embedding_dim, batch seq_len embedding_dim -> '
            'batch seq_len num_embeddings', 
            self.common.token_embedding.weight, 
            x,
        )

    def forward(self, x: t.Tensor, one_zero_attention_mask: Optional[t.Tensor] = None,):
        '''
        x: (batch, seq)
        '''
        assert isinstance(x, t.Tensor), f'Bad input to BertLanguageModel.forward: {type(x)} of length {len(x)}'
        x = self.common(x, one_zero_attention_mask=one_zero_attention_mask)
        x = self.linear(x)
        x = self.gelu(x)
        x = self.layer(x)
        x = self.tied_unembed(x)
        x += self.unembed_bias
        return x

    
class BertClassifier(nn.Module):

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.common = BertCommon(config=config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.stars = nn.Linear(config.hidden_size, 1)
        self.sentiment = nn.Linear(config.hidden_size, 2)

    def forward(self, x: t.Tensor, one_zero_attention_mask: Optional[t.Tensor] = None,) -> Tuple[t.Tensor, t.Tensor]:
        '''
        x: (batch, seq)

        return: Tuple(sentiment, stars)
        '''
        assert isinstance(x, t.Tensor), (
            f'Bad input to BertClassifier.forward: {type(x)} of length {len(x)}'
        )
        print('Calling BertCommon...')
        x = self.common(x, one_zero_attention_mask=one_zero_attention_mask)
        print('Taking first position...')
        x = x[:, 0] # First position only
        x = self.dropout(x)
        stars = self.stars(x) * 5 + 5
        sentiment = self.sentiment(x)
        return sentiment, stars
