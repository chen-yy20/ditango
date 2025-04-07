import torch
import torch.nn as nn
from einops import rearrange

from ditango.core.parallel_state import get_isp_group
from ditango.core.attention import proAttention
from ditango.logger import init_logger

try:
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except ImportError:
    xFuserLongContextAttention = None
    

logger = init_logger(__name__)

class Attention(nn.Module):
    def __init__(self, layer_id = -1):
        super().__init__()
        self.oAttn = proAttention(layer_id, get_isp_group())
    
    def attn_processor(self, attn_type):
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            return self.parallel_attn_func
        elif attn_type == 'ditango':
            return self.o_attn_func
        else:
            raise Exception('Not supported attention type...')

    def torch_attn_func(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs
    ):

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
            
        if attn_mask is not None and attn_mask.ndim == 3:   ## no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        x = rearrange(x, 'b h s d -> b s h d')
        return x        

    def parallel_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):
        # TODO: Change to oAttention
        assert xFuserLongContextAttention is not None; 'to use sequence parallel attention, xFuserLongContextAttention should be imported...'
        hybrid_seq_parallel_attn = xFuserLongContextAttention()
        x = hybrid_seq_parallel_attn(
            None, q,k,v, causal=causal
        )
        return x

    def o_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):
        x = self.oAttn.attn(
            q,
            k,
            v
        )
        x = x.to(q.dtype)
        
        return x
