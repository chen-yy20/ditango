from typing import Optional
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb


from ditango.core.parallel_state import get_isp_group
from ditango.logger import init_logger

from ditango.core.attention import proAttention

import nvtx
from contextlib import contextmanager

logger = init_logger(__name__)

@contextmanager
def nvtx_range(name, domain="CUSTOM"):
    nvtx.push_range(name, domain=domain)
    try:
        yield
    finally:
        nvtx.pop_range(domain=domain)
  
  
class CVX_oCacheAttnProcessor:
  def __init__(self, layer_id):
    self.oAttn = proAttention(layer_id, get_isp_group())

  def __call__(self,
      attn: Attention,
      hidden_states: torch.Tensor, # 已经按照seq切分完成
      encoder_hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      image_rotary_emb: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    text_seqlen = encoder_hidden_states.size(1) # 57
    latent_seqlen = hidden_states.size(1) # 4050
    batch_size = hidden_states.shape[0]

    assert attention_mask is None
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    

    inner_dim = key.shape[-1]
    head_num = attn.heads
    head_dim = inner_dim // head_num

    # head_num = attn.head_num
    query = query.view(batch_size, -1, head_num, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, head_num, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, head_num, head_dim).transpose(1, 2)
    
    if attn.norm_q is not None:
      query = attn.norm_q(query)
    if attn.norm_k is not None:
      key = attn.norm_k(key)

    if image_rotary_emb is not None:
      query[:, :, text_seqlen:] = apply_rotary_emb(query[:, :, text_seqlen:], image_rotary_emb)
      if not attn.is_cross_attention:
          key[:, :, text_seqlen:] = apply_rotary_emb(key[:, :, text_seqlen:], image_rotary_emb)

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    
    # [b, head_num, seq, head_dim]
    out = self.oAttn.attn(
      query,
      key,
      value
    )

    out = out.to(hidden_states.dtype)
    hidden_states = out.reshape(batch_size, -1, head_num * head_dim)
    hidden_states = attn.to_out[0](hidden_states) # linear proj
    hidden_states = attn.to_out[1](hidden_states) # drop out
    
    encoder_hidden_states, hidden_states = hidden_states.split([text_seqlen, latent_seqlen], dim=1)
    return hidden_states, encoder_hidden_states
  

