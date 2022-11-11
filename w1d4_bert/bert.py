#%%
from transformer import TransformerConfig
import torch.nn as nn
import torch as t
from typing import Optional
from einops import rearrange, reduce, repeat
import utils

#%%
class MultiheadAttention(nn.Module):

    def __init__(self, config: TransformerConfig):
        pass

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        pass 


class BERTBlock(nn.Module):

    def __init__(self, config):
        pass

    def forward(
        self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None
    ) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        pass


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
    filled = t.where(
        one_zero_attention_mask == 0, 
        big_negative_number,
        0,
    )
    reshaped = repeat(filled, 'b s -> b n_heads seq_q s', n_heads=1, seq_q=1)
    return reshaped

#%%
def test_make_additive_attention_mask(make_additive_attention_mask):
    from solutions_build_bert import make_additive_attention_mask as make_additive_attention_mask_soln
    arr = t.randint(low=0, high=2, size=(3, 4))
    expected = make_additive_attention_mask_soln(arr)
    actual = make_additive_attention_mask(arr)
    t.testing.assert_close(expected, actual)
#%%
utils.test_make_additive_attention_mask(make_additive_attention_mask)
# %%
