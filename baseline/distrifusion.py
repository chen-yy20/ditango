from typing import Optional
import torch
import threading
from typing import Optional, Tuple, Union, Dict
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, flash_attn_func
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb
import torch.distributed as dist

from .cache import get_fusion_cache
from ditango.core.config import get_config
from ditango.core.group_coordinate import GroupCoordinator
from ditango.utils import split_tensor_uneven, get_timestep
from ditango.logger import init_logger
from ditango.timer import get_timer
import math

logger = init_logger(__name__)

class CVX_DistriFusion_AttnProcessor:
    def __init__(self, layer_id, isp_group: GroupCoordinator):
        assert get_config().use_distrifusion, "This is Distrifusion Attention Processor, set '--use-distrifusion' to enable it."
        self.layer_id = layer_id
        self.group = isp_group
        self.cache = get_fusion_cache()

    @get_timer("DF Attn")
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
        timestep = get_timestep()

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
    
        # logger.debug(f"1: {query.shape=} {key.shape=} {value.shape=}")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
  
        # logger.info(f"t{timestep} l{self.layer_id} | isp_group_size={isp_world_size} rank={isp_rank}")

        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # logger.debug(f"3: {query.shape=} {key.shape=} {value.shape=}")

        # Prepare for flash attention
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        # k、v gather
        # ========== ISP gather ===========
        if not self.cache.is_cached(self.layer_id):
            fresh_key = self.group.all_gather(key, dim=1)
            fresh_value = self.group.all_gather(value, dim=1)
            # logger.debug(f"t{timestep} l{self.layer_id}| After ISP gather: key.shape={fresh_key.shape}, value.shape={fresh_value.shape}")
        else:
            fresh_key = key
            fresh_value = value
            # logger.debug(f"t{timestep} l{self.layer_id}| No ISP gather: key.shape={fresh_key.shape}, value.shape={fresh_value.shape}")
        # ========== Cache fetch ===========
        cache_dict = self.cache.get_kv(self.layer_id)
        if cache_dict is not None:
            cache_key = cache_dict['k']
            cache_value = cache_dict['v']
            # logger.debug(f"t{timestep} l{self.layer_id}| After Cache fetch: cache_key.shape={cache_key.shape}, cache_value.shape={cache_value.shape}")
        # ========== Cache Store ===========
        self.cache.store(self.layer_id, fresh_key, fresh_value)

        # ========== cat ===========
        if cache_dict is not None:
            cache_key_list= split_tensor_uneven(cache_key, self.group.world_size, dim=1)
            cache_value_list= split_tensor_uneven(cache_value, self.group.world_size, dim=1)
            cache_key_list[self.group.rank_in_group] = fresh_key
            cache_value_list[self.group.rank_in_group] = fresh_value
            key = torch.cat(cache_key_list, dim=1)
            value = torch.cat(cache_value_list, dim=1)
            # logger.info(f"t{timestep} l{self.layer_id}| After cat: key.shape={key.shape}, value.shape={value.shape}")
        else:
            key = fresh_key
            value = fresh_value
        # logger.info(f"t{timestep} l{self.layer_id}| After cat: key.shape={key.shape}, value.shape={value.shape}")

        total_seq_len_q = query.size(1)  # s/sp
        total_seq_len_kv = key.size(1)   # s = total_seq_len_q * sp

        # 为query设置cu_seqlens
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * total_seq_len_q, total_seq_len_q, 
                                  device=hidden_states.device, dtype=torch.int32)

        # 为key/value设置cu_seqlens
        cu_seqlens_kv = torch.arange(0, (batch_size + 1) * total_seq_len_kv, total_seq_len_kv, 
                                    device=hidden_states.device, dtype=torch.int32)

        hidden_states, _, _, _ = _flash_attn_varlen_forward(
            query.view(-1, attn.heads, head_dim),
            key.view(-1, attn.heads, head_dim),
            value.view(-1, attn.heads, head_dim),
            cu_seqlens_q=cu_seqlens_q,         
            cu_seqlens_k=cu_seqlens_kv,        
            max_seqlen_q=total_seq_len_q,        
            max_seqlen_k=total_seq_len_kv,       
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=False,
            alibi_slopes=None,
            return_softmax=False,
            block_table=None,
        )
        hidden_states = hidden_states.view(batch_size, -1, attn.heads * head_dim)

        # Linear projections and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Split back encoder and hidden states
        encoder_hidden_states, hidden_states = hidden_states.split([text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)

        # Gather Cache
        self.cache.async_gather(self.layer_id, dim=1, group=self.group)
        # encoder_hidden_states = encoder_hidden_states.contiguous()
        # hidden_states = hidden_states.contiguous()

        return hidden_states, encoder_hidden_states
    