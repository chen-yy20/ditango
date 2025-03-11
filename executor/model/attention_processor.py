from typing import Optional
import torch
import torch.nn.functional as F
import warnings
from math import gcd
from einops import rearrange, repeat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, flash_attn_func
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb
import torch.distributed as dist


from ditango.core.parallel_state import get_usp_group, get_isp_group, get_osp_group
from ditango.core.redundancy_map import get_redundancy_map
from ditango.core.feature_cache import get_cache, exist_cache
from ditango.logger import init_logger
from ditango.executor.utils import split_tensor_uneven

from ditango.core.attention import oAttention

import math

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

def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
      if slice_ is not None:
        raise RuntimeError("first update_out_and_lse should not pass slice_ args")
      out = block_out.to(torch.float32)
      lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
      slice_out, slice_lse = out[slice_], lse[slice_]
      slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
      out[slice_], lse[slice_] = slice_out, slice_lse
    else:
      out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse
# ========================================== CogVideoX-5B ==========================================

def patch_attn_processor(module, attn_processor):
  for name, module in module.named_modules():
    if hasattr(module, "set_attn_processor"):
      module.set_attn_processor(attn_processor)

# ulysses attention processor
class Ulysses_CogVideoXAttnProcessor2_0:
  r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
  """

  def __init__(self):
    logger.info("Using Ulysses_CogVideoXAttnProcessor2_0")
    self.world_size = 1
    self.rank = 0
    if get_usp_group().world_size > 1:
      self.world_size = get_usp_group().world_size
      self.rank = get_usp_group().rank_in_group
    if not hasattr(F, "scaled_dot_product_attention"):
      raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

  def __call__(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    text_seq_length = encoder_hidden_states.size(1)

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = attn.to_q(encoder_hidden_states)
    encoder_key = attn.to_k(encoder_hidden_states)
    encoder_value = attn.to_v(encoder_hidden_states)

    if get_usp_group().world_size > 1:
      assert attn.heads % get_usp_group().world_size == 0, f"Number of heads {attn.heads} must be divisible by sequence parallel size {get_usp_group().world_size}"
      attn_heads = attn.heads // get_usp_group().world_size
      query, key, value = map(
        lambda x: get_usp_group().all_to_all(x, scatter_dim=2, gather_dim=1),
        [query, key, value],
      ) # 为每个头收集完整的qkv
      encoder_query, encoder_key, encoder_value = map(
        lambda x: split_tensor_uneven(x, self.world_size, dim=2)[self.rank],
        [encoder_query, encoder_key, encoder_value],
      ) # 每个rank只保留自己对应头部分的encoder hidden states
      
    else:
      attn_heads = attn.heads

    query = torch.cat([encoder_query, query], dim=1)
    key = torch.cat([encoder_key, key], dim=1)
    value = torch.cat([encoder_value, value], dim=1)

    batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)

    if attention_mask is not None:
      attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
      attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    inner_dim = key.shape[-1] # 头内部的dim
    head_dim = inner_dim // attn_heads

    query = query.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
      query = attn.norm_q(query)
    if attn.norm_k is not None:
      key = attn.norm_k(key)

    # Apply RoPE if needed
    if image_rotary_emb is not None:
      query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
      if not attn.is_cross_attention:
        key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
    
    # original attn
    hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

    #TODO:flash attn

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn_heads * head_dim)

    encoder_hidden_states, hidden_states = hidden_states.split([text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)
    if get_usp_group().world_size > 1:
      hidden_states = get_usp_group().all_to_all(hidden_states, scatter_dim=1, gather_dim=2) # 恢复原有的切分方式
      encoder_hidden_states = get_usp_group().all_gather(encoder_hidden_states.contiguous(), dim=2) # 收集所有rank的encoder hidden states

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    encoder_hidden_states = attn.to_out[0](encoder_hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    encoder_hidden_states = attn.to_out[1](encoder_hidden_states)
    return hidden_states, encoder_hidden_states



class CVX_UlyssesAttnProcessor:
    def __init__(self, layer_id=-1):
        logger.info(f"Using CVX_UlyssesAttnProcessor, {layer_id=}")
        self.world_size = 1
        self.rank = 0
        if get_usp_group().world_size > 1:
            self.world_size = get_usp_group().world_size
            self.rank = get_usp_group().rank_in_group
        self.layer_id = layer_id

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
          
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        encoder_query = attn.to_q(encoder_hidden_states)
        encoder_key = attn.to_k(encoder_hidden_states)
        encoder_value = attn.to_v(encoder_hidden_states)

        if get_isp_group().world_size > 1:
            assert attn.heads % get_isp_group().world_size == 0
            attn_heads = attn.heads // get_isp_group().world_size
            query, key, value = map(
                lambda x: get_isp_group().uneven_all_to_all(x, scatter_dim=2, gather_dim=1, uneven_dim=1, seq_id=0),
                [query, key, value],
            )
            encoder_query, encoder_key, encoder_value = map(
                lambda x: get_isp_group().uneven_all_to_all(x, scatter_dim=2, gather_dim=1, uneven_dim=1, seq_id=1),
                [encoder_query, encoder_key, encoder_value],
            )
        else:
            attn_heads = attn.heads
        text_seq_length = encoder_query.size(1)
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn_heads

        query = query.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # Prepare for flash attention
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        # Calculate cu_seqlens for variable length support
        total_seq_len = query.size(1)
        cu_seqlens = torch.arange(0, (batch_size + 1) * total_seq_len, total_seq_len, 
                                device=hidden_states.device, dtype=torch.int32)

        # Flash attention forward pass
        hidden_states, _, _, _= _flash_attn_varlen_forward(
            query.view(-1, attn_heads, head_dim),
            key.view(-1, attn_heads, head_dim),
            value.view(-1, attn_heads, head_dim),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=total_seq_len,
            max_seqlen_k=total_seq_len,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=False,
            alibi_slopes=None,
            return_softmax=False,
            block_table=None,
        )

        hidden_states = hidden_states.view(batch_size, -1, attn_heads * head_dim)
        # logger.debug(f"2:{hidden_states.shape=} ")
        # Split back encoder and hidden states
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)

        # logger.debug(f"3:{hidden_states.shape=} {encoder_hidden_states.shape=}")
        if get_isp_group().world_size > 1:
            hidden_states = get_isp_group().uneven_all_to_all(hidden_states, scatter_dim=1, gather_dim=2, uneven_dim=1, seq_id=0)
            encoder_hidden_states = get_isp_group().uneven_all_to_all(encoder_hidden_states, scatter_dim=1, gather_dim=2, uneven_dim=1, seq_id=1)
        # logger.debug(f"4:{hidden_states.shape=} {encoder_hidden_states.shape=}")
        # Linear projections and dropout
        hidden_states = attn.to_out[0](hidden_states)
        encoder_hidden_states = attn.to_out[0](encoder_hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_out[1](encoder_hidden_states)

        return hidden_states, encoder_hidden_states

class CVX_RingAttnProcessor:
  def __init__(self, layer_id):
    self.group = get_usp_group()
    self.full_sp_size = self.group.world_size
    self.block_id = self.group.rank_in_group
    self.rank = self.group.rank_in_group
    self.layer_id = layer_id
    logger.info(f"Using CVX RingAttnProcessor, {self.block_id=}, {self.layer_id=}")

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

    isp_stride = self.full_sp_size
    if exist_cache():
      cache = get_cache()
      timestep = get_cache().timestep
      isp_stride = int(get_redundancy_map()[timestep, self.layer_id])
      cache.flush_cache(self.layer_id)
      cache.new_set_should_cache(self.layer_id, self.block_id)
      merge_block_size = cache.merge_block_size[self.layer_id]
      merge_block_num = cache.merge_block_num[self.layer_id]

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

    padded_block_seqlen = text_seqlen + latent_seqlen
    padded_block_shape = torch.Size([batch_size, padded_block_seqlen, head_num, head_dim])

    dtype = hidden_states.dtype

    next_k, next_v = None, None
    out, lse = None, None
    
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
    
    # print(f"{query.shape=} {hidden_states.shape=}", flush=True)
    # query.shape=torch.Size([1, 17776, 48, 64]) hidden_states.shape=torch.Size([1, 17776, 3072])
    
    # # =================== update prev =================
    if exist_cache():
      end_block_id = (self.block_id - self.rank + isp_stride) % self.full_sp_size
      merge_end_block_id = end_block_id // merge_block_size
      assert (self.full_sp_size - isp_stride) % merge_block_size == 0, "Something went wrong..."
      block_left = (self.full_sp_size - isp_stride) // merge_block_size
      for i in range(block_left):
        merge_block_id = (merge_end_block_id + i) % merge_block_num
        # logger.debug(f"{isp_stride=} {self.rank=} | {merge_block_num=}, tried to store {merge_block_id=}")
        # if selfish and cache_block_id == self.rank:
        #   continue
        cache_out, cache_lse = cache.new_get_kv(self.layer_id, merge_block_id)
        out, lse = update_out_and_lse(out, lse, cache_out, cache_lse)
      cache.new_purge_cache(self.layer_id)
      
    # ================== get first kv =======================
    if isp_stride != self.full_sp_size and self.rank != self.block_id:
      recv_rank = self.block_id
      skip = (self.rank - self.block_id) % self.full_sp_size
      
      send_key = key.detach().clone()
      send_value = value.detach().clone() 
      send_rank = (self.rank + skip) % self.full_sp_size
      # logger.debug(f"{cache.timestep} | {self.block_id} | {recv_rank} -> rank:{self.rank} -> {send_rank}")
      self.group.p2p_isend(send_key, dst=send_rank)
      self.group.p2p_isend(send_value, dst=send_rank)
      first_key = self.group.p2p_irecv(size=padded_block_shape, dtype=dtype, src=recv_rank)
      first_value = self.group.p2p_irecv(size=padded_block_shape, dtype=dtype, src=recv_rank)
      self.group.p2p_commit()
      
      # should overlap with something
      
      
      self.group.p2p_wait()
      key = first_key
      value = first_value
    # ================== ISP Loop =====================
    
    next_rank = (self.rank - 1) % self.full_sp_size
    prev_rank = (self.rank + 1) % self.full_sp_size
    fresh_out, fresh_lse = None, None
    merge_cnt = 0
    for step in range(isp_stride):
      if step + 1 != isp_stride:
        send_key = key.detach().clone()
        send_value = value.detach().clone()
        self.group.p2p_isend(send_key, dst=next_rank)
        self.group.p2p_isend(send_value, dst=next_rank)
        next_k = self.group.p2p_irecv(size=padded_block_shape, dtype=dtype, src=prev_rank)
        next_v = self.group.p2p_irecv(size=padded_block_shape, dtype=dtype, src=prev_rank)
        self.group.p2p_commit()

      block_out, block_lse, _  = flash_attn_func(
              query,
              key,
              value,
              dropout_p=0.0,
              softmax_scale=1.0 / math.sqrt(head_dim),
              causal=False,
              return_attn_probs=True,
          )
      fresh_out, fresh_lse = update_out_and_lse(fresh_out, fresh_lse, block_out, block_lse)
      merge_cnt += 1
      # =============== update Cache ================
      if exist_cache() and merge_cnt == merge_block_size:
          merge_block_id = ((self.block_id - self.rank) % self.full_sp_size) // merge_block_size
          fresh_lse = fresh_lse.squeeze(-1).transpose(-1,-2) # reshape lse for block save
          cache.new_store(self.layer_id, merge_block_id, fresh_out, fresh_lse)
          # logger.debug(f"{out.shape if out is not None else None} {lse.shape if lse is not None else None} {fresh_out.shape=} {fresh_lse.shape=}" )
          out, lse = update_out_and_lse(out, lse, fresh_out, fresh_lse)
          # logger.debug(f"{out.shape=} {lse.shape=} {fresh_out.shape=} {fresh_lse.shape=}" )
          # exit()
          merge_cnt = 0
          fresh_out = None
          fresh_lse = None
        # for analyse
        # cache.store_and_save_diff(self.layer_id, self.block_id, block_out, block_lse)

      if step + 1 != isp_stride:
        # logger.debug(f"{rank=} waiting for step {step}")
        self.group.p2p_wait()
        key = next_k
        value = next_v
      # update block id
      self.block_id = (self.block_id + 1) % self.full_sp_size

    out = out.to(hidden_states.dtype)
    hidden_states = out.reshape(batch_size, -1, head_num * head_dim)
    hidden_states = attn.to_out[0](hidden_states) # linear proj
    hidden_states = attn.to_out[1](hidden_states) # drop out
    
    encoder_hidden_states, hidden_states = hidden_states.split([text_seqlen, latent_seqlen], dim=1)
    return hidden_states, encoder_hidden_states
  
  
class CVX_oCacheAttnProcessor:
  def __init__(self, layer_id):
    # oAttn / uAttn
    self.oAttn = oAttention(layer_id, get_isp_group())

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
  

# ========================================== Mochi-Preview ==========================================
class Mochi_oCacheAttnProcessor:
  def __init__(self, layer_id):
    self.oAttn = oAttention(layer_id, get_isp_group())

  def __call__(self,
      attn: Attention,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      image_rotary_emb: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    
    encoder_query = attn.add_q_proj(encoder_hidden_states)
    encoder_key = attn.add_k_proj(encoder_hidden_states)
    encoder_value = attn.add_v_proj(encoder_hidden_states)

    encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
    encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
    encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

    if attn.norm_added_q is not None:
        encoder_query = attn.norm_added_q(encoder_query)
    if attn.norm_added_k is not None:
        encoder_key = attn.norm_added_k(encoder_key)
    
    if image_rotary_emb is not None:
      
        def apply_rotary_emb(x, freqs_cos, freqs_sin):
            x_even = x[..., 0::2].float()
            x_odd = x[..., 1::2].float()

            cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
            sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

            return torch.stack([cos, sin], dim=-1).flatten(-2)

        query = apply_rotary_emb(query, *image_rotary_emb)
        key = apply_rotary_emb(key, *image_rotary_emb)
        
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    encoder_query, encoder_key, encoder_value = (
            encoder_query.transpose(1, 2),
            encoder_key.transpose(1, 2),
            encoder_value.transpose(1, 2),
        )
    sequence_length = query.size(2)
    encoder_sequence_length = encoder_query.size(2)
    
    query = torch.cat([query, encoder_query], dim=2)
    key = torch.cat([key, encoder_key], dim=2)
    value = torch.cat([value, encoder_value], dim=2)

    # ================= DoIT ========================
    # torch.SDPA： [b, head_num, seq, head_dim]
    # flash Attn: [b, seq, head num, head_dim]
    query = query.transpose(1,2).contiguous()
    key = key.transpose(1,2).contiguous()
    value = value.transpose(1,2).contiguous()
    
    hidden_states = self.oAttn.attn(
      query,
      key,
      value
    )
    # output should be [2, 24, 22516, 128]
    hidden_states = hidden_states.flatten(2, 3)
    # ================= DoIT ==========================
    hidden_states = hidden_states.to(query.dtype)

    hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )
    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if hasattr(attn, "to_add_out"):
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
    # print(f"{hidden_states.shape=} {encoder_hidden_states.shape=}", flush=True)
    return hidden_states, encoder_hidden_states
  
class Mochi_RingAttnProcessor:
  def __init__(self):
    logger.info("Using Mochi RingAttnProcessor")

  def _update_out_and_lse(
    self,
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse

  def update_out_and_lse(
    self,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
      if slice_ is not None:
        raise RuntimeError("first update_out_and_lse should not pass slice_ args")
      out = block_out.to(torch.float32)
      lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
      slice_out, slice_lse = out[slice_], lse[slice_]
      slice_out, slice_lse = self._update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
      out[slice_], lse[slice_] = slice_out, slice_lse
    else:
      out, lse = self._update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse
  
  def __call__(self,
      attn: Attention,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      image_rotary_emb: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    text_seq_length = encoder_hidden_states.size(1)
    batch_size = hidden_states.shape[0]

    assert attention_mask is None
    group = get_usp_group()
    # group.barrier()
    rank = group.rank_in_group
    world_size = group.world_size
    # local rank
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    block_seqlen = hidden_states.size(1)
    
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    
    encoder_query = attn.add_q_proj(encoder_hidden_states)
    encoder_key = attn.add_k_proj(encoder_hidden_states)
    encoder_value = attn.add_v_proj(encoder_hidden_states)

    encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
    encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
    encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

    if attn.norm_added_q is not None:
        encoder_query = attn.norm_added_q(encoder_query)
    if attn.norm_added_k is not None:
        encoder_key = attn.norm_added_k(encoder_key)

    
    if image_rotary_emb is not None:
        def apply_rotary_emb(x, freqs_cos, freqs_sin):
            x_even = x[..., 0::2].float()
            x_odd = x[..., 1::2].float()

            cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
            sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

            return torch.stack([cos, sin], dim=-1).flatten(-2)

        query = apply_rotary_emb(query, *image_rotary_emb)
        key = apply_rotary_emb(key, *image_rotary_emb)

    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)

    head_num = query.size(-2)
    head_dim = query.size(-1)
    # logger.info(f"{query.shape=} {key.shape=}")
    # exit()

    # 开始ring attn计算，输入shape应该为：？query.shape=torch.Size([1, 24, 22516, 128]) encoder_query.shape=torch.Size([1, 24, 256, 128])
    # STANDARD INPUT ⬇️
    block_shape = torch.Size([batch_size, block_seqlen, head_num, head_dim])
    padded_block_seqlen = text_seq_length + block_seqlen
    padded_block_shape = torch.Size([batch_size, padded_block_seqlen, head_num, head_dim])

    cu_seqlens = torch.arange(0, (batch_size + 1) * block_seqlen, block_seqlen, device=hidden_states.device, dtype=torch.int32)
    padded_cu_seqlens = torch.arange(0, (batch_size + 1) * padded_block_seqlen, padded_block_seqlen, device=hidden_states.device, dtype=torch.int32)

    softmax_scale = 1.0 / head_dim**0.5
    dtype = hidden_states.dtype

    next_k, next_v = None, None
    out, lse = None, None
    
    for step in range(world_size):
      # with nvtx.annotate(f"zrx_step_{step}"):
      if step + 1 != world_size:
        group.p2p_isend(key, dst=next_rank)
        group.p2p_isend(value, dst=next_rank)
        next_k = group.p2p_irecv(size=padded_block_shape if rank == step + 1 else block_shape, dtype=dtype, src=prev_rank)
        next_v = group.p2p_irecv(size=padded_block_shape if rank == step + 1 else block_shape, dtype=dtype, src=prev_rank)
        group.p2p_commit()

      block_out, block_lse, _, _= _flash_attn_varlen_forward(
        query.view(-1, attn.heads, head_dim),
        key.view(-1, attn.heads, head_dim),
        value.view(-1, attn.heads, head_dim),
        cu_seqlens_q=padded_cu_seqlens if rank == 0 else cu_seqlens,
        cu_seqlens_k=padded_cu_seqlens if rank == step else cu_seqlens,
        max_seqlen_q=padded_block_seqlen if rank == 0 else block_seqlen,
        max_seqlen_k=padded_block_seqlen if rank == step else block_seqlen,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
        alibi_slopes=None,
        return_softmax=False,
        block_table=None,
      )
      out, lse = update_out_and_lse(out, lse, block_out, block_lse)
      # with nvtx.annotate(f"zrx_step_{step}_wait"):
      if step + 1 != world_size:
        # logger.debug(f"{rank=} waiting for step {step}")
        group.p2p_wait()
        key = next_k
        value = next_v
    out = out.to(hidden_states.dtype)
    hidden_states = out.reshape(batch_size, -1, head_num * head_dim)
    # print(f"{hidden_states.shape=} {hidden_states[0][0][:3]=}", flush=True)
    # exit()

    hidden_states, encoder_hidden_states  = hidden_states.split([hidden_states.size(1) - text_seq_length, text_seq_length], dim=1)

    hidden_states = attn.to_out[0](hidden_states) # linear proj
    hidden_states = attn.to_out[1](hidden_states) # drop out
    
    if hasattr(attn, "to_add_out"):
        
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
    hidden_states = hidden_states.contiguous()
    
    # print(f"{hidden_states[0][0][:3]=} {encoder_hidden_states[0][0][:3]=}", flush=True)
    # exit()
    # print(f"{hidden_states.shape=} {encoder_hidden_states.shape=}", flush=True)
    # exit()
    return hidden_states, encoder_hidden_states

class Mochi_FakeUlyssesAttnProcessor:
    def __init__(self):
        logger.info("Using Mochi_FakeUlyssesAttnProcessor")
        self.world_size = 1
        self.rank = 0
        if get_usp_group().world_size > 1:
            self.world_size = get_usp_group().world_size
            self.rank = get_usp_group().rank_in_group

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if get_usp_group().world_size > 1:
            assert attn.heads % get_usp_group().world_size == 0
            attn_heads = attn.heads // get_usp_group().world_size
            query, key, value = map(
                lambda x: get_usp_group().uneven_all_to_all(x, scatter_dim=2, gather_dim=1, uneven_dim=1),
                [query, key, value],
            )
            encoder_query, encoder_key, encoder_value = map(
                lambda x: split_tensor_uneven(x, self.world_size, dim=2)[self.rank],
                [encoder_query, encoder_key, encoder_value],
            )
            if image_rotary_emb is not None:
              cos, sin = image_rotary_emb
              cos_splits = split_tensor_uneven(cos, self.world_size, dim=1)
              sin_splits = split_tensor_uneven(sin, self.world_size, dim=1)
              image_rotary_emb = (cos_splits[self.rank], sin_splits[self.rank])
                

        else:
            attn_heads = attn.heads
        
        if image_rotary_emb is not None:
            if get_usp_group().world_size == 8:
                query = query[:, :-4, :, :]
                key = key[:, :-4, :, :]
                value = value[:, :-4, :, :]
            # print(f"{query.shape=} {image_rotary_emb[0].shape=} {image_rotary_emb[1].shape=}", flush=True)
            # exit()
            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()

                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)
            query = apply_rotary_emb(query, *image_rotary_emb)
            key = apply_rotary_emb(key, *image_rotary_emb)
        # [B, S, H]
        
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        
        # logger.debug(f"{query.shape=} {key.shape=} {value.shape=}")
        # exit()
        
        head_dim = key.size(-1)
        # Calculate cu_seqlens for variable length support
        total_seq_len = query.size(1)
        cu_seqlens = torch.arange(0, (batch_size + 1) * total_seq_len, total_seq_len, 
                                device=hidden_states.device, dtype=torch.int32)
        # Flash attention forward pass
        hidden_states, _, _  = flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            return_attn_probs=True,
        )
        # hidden_states, _, _, _ = _flash_attn_varlen_forward(
        #     query.view(-1, attn_heads, head_dim),
        #     key.view(-1, attn_heads, head_dim),
        #     value.view(-1, attn_heads, head_dim),
        #     cu_seqlens_q=cu_seqlens,
        #     cu_seqlens_k=cu_seqlens,
        #     max_seqlen_q=total_seq_len,
        #     max_seqlen_k=total_seq_len,
        #     dropout_p=0.0,
        #     softmax_scale=1.0 / math.sqrt(head_dim),
        #     causal=False,
        #     alibi_slopes=None,
        #     return_softmax=False,
        #     block_table=None,
        # )
        hidden_states = hidden_states.view(batch_size, -1, attn_heads * head_dim)
        # print(f"{hidden_states.shape=}, {hidden_states[0][0][:3]=}", flush=True)
        # exit()

        # Split back encoder and hidden states
        hidden_states, encoder_hidden_states = hidden_states.split(
            [hidden_states.size(1) - text_seq_length, text_seq_length], dim=1)
        if get_usp_group().world_size > 1:
            hidden_states = get_usp_group().uneven_all_to_all(hidden_states, scatter_dim=1, gather_dim=2, uneven_dim=1)
            encoder_hidden_states = get_usp_group().all_gather(encoder_hidden_states.contiguous(), dim=2)
        # Linear projections and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        # encoder_hidden_states = attn.to_out[1](encoder_hidden_states)

        return hidden_states, encoder_hidden_states


# ========================================== OpenSora-Plan 3D ==========================================
from ditango.executor.model.modules.opensora_modules import RoPE3D, PositionGetter3D  

class OpenSoraAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, interpolation_scale_thw=(1, 1, 1), 
                 sparse1d=False, sparse_n=2, sparse_group=False, is_cross_attn=True):
        self.sparse1d = sparse1d
        self.is_cross_attn = is_cross_attn
        self.interpolation_scale_thw = interpolation_scale_thw
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.is_cross_attn = is_cross_attn
        self.interpolation_scale_thw = interpolation_scale_thw
        temp = sparse_n if self.sparse1d else 1 
        self.real_sparse_n = temp // gcd(temp, get_usp_group().world_size) 
        
        self._init_rope(interpolation_scale_thw)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()
    
    def _sparse_1d(self, x, frame, height, width):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        l = x.shape[0]
        # assert l == frame*height*width
        pad_len = 0
        if l % (self.real_sparse_n * self.real_sparse_n) != 0:
            pad_len = self.real_sparse_n * self.real_sparse_n - l % (self.real_sparse_n * self.real_sparse_n)
        if pad_len != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        if not self.sparse_group:
            x = rearrange(x, '(g k) b d -> g (k b) d', k=self.real_sparse_n)
        else:
            x = rearrange(x, '(n m k) b d -> (n k) (m b) d', m=self.real_sparse_n, k=self.real_sparse_n)
        return x, pad_len
    
    def _reverse_sparse_1d(self, x, frame, height, width, pad_len):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        # assert x.shape[0] == (frame*height*width+pad_len) // self.sparse_n
        if not self.sparse_group:
            x = rearrange(x, 'g (k b) d -> (g k) b d', k=self.real_sparse_n)
        else:
            x = rearrange(x, '(n k) (m b) d -> (n m k) b d', m=self.real_sparse_n, k=self.real_sparse_n)
        x = x[:frame*height*width, :, :]
        return x
    
    def _sparse_1d_kv(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = repeat(x, 's b d -> s (k b) d', k=self.real_sparse_n)
        return x
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        frame: int = 8, 
        height: int = 16, 
        width: int = 16, 
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states

        sequence_length, batch_size, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        FA_head_num = attn.heads
        total_frame = frame

        query = query.view(-1, batch_size, FA_head_num, head_dim)
        key = key.view(-1, batch_size, FA_head_num, head_dim)

        if not self.is_cross_attn:
            # require the shape of (ntokens x batch_size x nheads x dim)
            pos_thw = self.position_getter(batch_size, t=total_frame, h=height, w=width, device=query.device)

            query = self.rope(query, pos_thw)
            key = self.rope(key, pos_thw)
            

        query = query.view(-1, batch_size, FA_head_num * head_dim)
        key = key.view(-1, batch_size, FA_head_num * head_dim)
        value = value.view(-1, batch_size, FA_head_num * head_dim)
        if self.sparse1d:
            query, pad_len = self._sparse_1d(query, total_frame, height, width)
            if self.is_cross_attn:
                key = self._sparse_1d_kv(key)
                value = self._sparse_1d_kv(value)
            else:
                key, pad_len = self._sparse_1d(key, total_frame, height, width)
                value, pad_len = self._sparse_1d(value, total_frame, height, width)
        # if npu_config is not None:
        #     hidden_states = npu_config.run_attention(query, key, value, attention_mask, "SBH", head_dim, FA_head_num)
        # else:
        query = rearrange(query, 's b (h d) -> b h s d', h=FA_head_num)
        key = rearrange(key, 's b (h d) -> b h s d', h=FA_head_num)
        value = rearrange(value, 's b (h d) -> b h s d', h=FA_head_num)
        # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
        # 0, 0 ->(bool) False, False ->(any) False ->(not) True
        # if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
        #     attention_mask = None
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True):
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = rearrange(hidden_states, 'b h s d -> s b (h d)', h=FA_head_num)

        if self.sparse1d:
            hidden_states = self._reverse_sparse_1d(hidden_states, total_frame, height, width, pad_len)

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # if attn.residual_connection:
        #     print('attn.residual_connection')
            # hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class oCache_OpenSoraAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, interpolation_scale_thw=(1, 1, 1), 
                 sparse1d=False, sparse_n=2, sparse_group=False, is_cross_attn=True, layer_id=-1):
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.is_cross_attn = is_cross_attn
        self.interpolation_scale_thw = interpolation_scale_thw
        self.layer_id = layer_id
        temp = sparse_n if self.sparse1d else 1 
        # self.real_sparse_n = temp // gcd(temp, get_usp_group().world_size) 
        self.real_sparse_n = sparse_n
        self.oAttn = oAttention(layer_id, get_isp_group(temp))
        if dist.get_rank() == 0:
            logger.debug(f"layer{layer_id} | init oAttn, sparse= {temp}")
        
        self._init_rope(interpolation_scale_thw)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()
    
    def _sparse_1d(self, x, frame, height, width):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        l = x.shape[0]
        # assert l == frame*height*width
        pad_len = 0
        if l % (self.real_sparse_n * self.real_sparse_n) != 0:
            pad_len = self.real_sparse_n * self.real_sparse_n - l % (self.real_sparse_n * self.real_sparse_n)
        if pad_len != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        if not self.sparse_group:
            x = rearrange(x, '(g k) b d -> g (k b) d', k=self.real_sparse_n)
        else:
            x = rearrange(x, '(n m k) b d -> (n k) (m b) d', m=self.real_sparse_n, k=self.real_sparse_n)
        return x, pad_len
    
    def _reverse_sparse_1d(self, x, frame, height, width, pad_len):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        # assert x.shape[0] == (frame*height*width+pad_len) // self.sparse_n
        if not self.sparse_group:
            x = rearrange(x, 'g (k b) d -> (g k) b d', k=self.real_sparse_n)
        else:
            x = rearrange(x, '(n k) (m b) d -> (n m k) b d', m=self.real_sparse_n, k=self.real_sparse_n)
        x = x[:frame*height*width, :, :]
        return x
    
    def _sparse_1d_kv(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = repeat(x, 's b d -> s (k b) d', k=self.real_sparse_n)
        return x
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        frame: int = 8, 
        height: int = 16, 
        width: int = 16, 
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states

        sequence_length, batch_size, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        FA_head_num = attn.heads
        total_frame = frame

        query = query.view(-1, batch_size, FA_head_num, head_dim)
        key = key.view(-1, batch_size, FA_head_num, head_dim)

        if not self.is_cross_attn:
            # require the shape of (ntokens x batch_size x nheads x dim)
            pos_thw = self.position_getter(batch_size, t=total_frame, h=height, w=width, device=query.device)
            query = self.rope(query, pos_thw)
            key = self.rope(key, pos_thw)
            

        query = query.view(-1, batch_size, FA_head_num * head_dim)
        key = key.view(-1, batch_size, FA_head_num * head_dim)
        value = value.view(-1, batch_size, FA_head_num * head_dim)
        # logger.debug(f'Before Sparse: q {query.shape=}, k {key.shape=}, v {value.shape=}')
        if self.sparse1d:
            # logger.debug(f"layer: {self.layer_id} | do sparse, input:{query.shape=}")
            query, pad_len = self._sparse_1d(query, total_frame, height, width)
            if self.is_cross_attn:
                key = self._sparse_1d_kv(key)
                value = self._sparse_1d_kv(value)
            else:
                key, pad_len = self._sparse_1d(key, total_frame, height, width)
                value, pad_len = self._sparse_1d(value, total_frame, height, width)
            # ================== DoIT - osp swtich ======================
            if get_osp_group(self.sparse_n).world_size > 1:
                query = get_osp_group(self.sparse_n).all_to_all(query, scatter_dim=1, gather_dim=0)
                key = get_osp_group(self.sparse_n).all_to_all(key, scatter_dim=1, gather_dim=0)
                value = get_osp_group(self.sparse_n).all_to_all(value, scatter_dim=1, gather_dim=0)
            # =========================== DoIT ==============================
                
        # if npu_config is not None:
        #     hidden_states = npu_config.run_attention(query, key, value, attention_mask, "SBH", head_dim, FA_head_num)
        # else:
        # query = rearrange(query, 's b (h d) -> b h s d', h=FA_head_num)
        # key = rearrange(key, 's b (h d) -> b h s d', h=FA_head_num)
        # value = rearrange(value, 's b (h d) -> b h s d', h=FA_head_num)
        
        # ======================= DoIT =========================
        query = rearrange(query, 's b (h d) -> b s h d', h=FA_head_num).contiguous()
        key = rearrange(key, 's b (h d) -> b s h d', h=FA_head_num).contiguous()
        value = rearrange(value, 's b (h d) -> b s h d', h=FA_head_num).contiguous()
  
        if attention_mask is not None:
          warnings.warn("Seems that attention mask is not none. Current system does not support attention computation with custom attention mask")
        hidden_states = self.oAttn.attn(
          query,
          key,
          value
        )
        hidden_states = hidden_states.to(query.dtype).contiguous()
        hidden_states = rearrange(hidden_states, 'b s h d -> s b (h d)', h=FA_head_num)
        
        # ======================= DoIT =========================
        if self.sparse1d:
            # ================== DoIT - osp swtich ======================
            if get_osp_group(self.sparse_n).world_size > 1:
                  hidden_states = get_osp_group(self.sparse_n).all_to_all(hidden_states, scatter_dim=0, gather_dim=1)
            # ========================== DoIT ===========================
            hidden_states = self._reverse_sparse_1d(hidden_states, total_frame, height, width, pad_len)

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ========================================== HunyuanVideo ==========================================
# ditango Adapt 6
MEMORY_LAYOUT = {
  
    "flash": (
        # lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
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
}

from ditango.timer import get_timer
class HunYuan_RingProcessor:
  def __init__(self, layer_id):
      self.oAttn = oAttention(layer_id, get_isp_group())
      
  def ring_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seqlen_q: int,
        mode: str = "flash",
        drop_rate: int = 0,
        attn_mask: torch.Tensor = None,
        batch_size: int = 1,
        
  ):
      assert mode == "flash", "Use flash attn please. You are slow enough."
      pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
      query = pre_attn_layout(q)
      key = pre_attn_layout(k)
      value = pre_attn_layout(v)

      out = self.oAttn.attn(
        query,
        key,
        value
      )
              
      out = out.view(
        batch_size, seqlen_q, out.shape[-2], out.shape[-1]
      )
      out = post_attn_layout(out)
      b, s, a, d = out.shape
      out = out.reshape(b, s, -1)
      return out
  @get_timer("Ulysses Attention")
  def ulysses_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seqlen_q: int,
        mode: str = "flash",
        drop_rate: int = 0,
        attn_mask: torch.Tensor = None,
        batch_size: int = 1,
  ):
    # TODO
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    query = pre_attn_layout(q)
    key = pre_attn_layout(k)
    value = pre_attn_layout(v)
    heads = query.shape[-2]
    # logger.debug(f"{q.shape=}")
    if get_usp_group().world_size > 1:
      assert heads % get_usp_group().world_size == 0
      attn_heads = heads // get_usp_group().world_size
      query, key, value = map(
          lambda x: get_usp_group().all_to_all(x, scatter_dim=2, gather_dim=1),
          [query, key, value],
      )
      
    # logger.debug(f"{query.shape=}")
    # exit()
      
    out, _, _  = flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            return_attn_probs=True,
        )
    # a2a
    
    # logger.debug(f'{out.shape}')
    # exit()
    if get_usp_group().world_size > 1:
        out = get_usp_group().all_to_all(out, scatter_dim=1, gather_dim=2)
    
    
    out = post_attn_layout(out)
    b, s, a, d = out.shape
    out = out.reshape(b, s, -1)
    return out
    
    