import torch
from ditango.core.attention import proAttention
from ditango.core.parallel_state import get_isp_group


MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "oAttn": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    )
}

class Hunyuan_DiTangoProcessor:
  def __init__(self, layer_id):
        self.varlen_Attn = proAttention(layer_id, get_isp_group()).varlen_attn 
      
  def ringfusion_attn(
        self,
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        batch_size=1,
        
  ):
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["flash"]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)
    # print(f"{q.shape=} {k.shape=} {v.shape=} {max_seqlen_q=} {max_seqlen_kv=} {cu_seqlens_q=} {cu_seqlens_kv=}")
    x = self.varlen_Attn(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    )
    x = x.view(
        batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
    )  # reshape x to [b, s, a, d]
    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out