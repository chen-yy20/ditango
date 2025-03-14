from typing import Optional
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, flash_attn_func
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb
import torch.distributed as dist


from ditango.core.parallel_state import get_usp_group, get_isp_group, get_osp_group
from ditango.core.redundancy_map import get_redundancy_map
from ditango.core.feature_cache import get_cache, exist_cache
from ditango.logger import init_logger
from ditango.timer import get_timer

import math

logger = init_logger(__name__)


class CVX_UlyssesAttnProcessor:
    def __init__(self, layer_id=-1):
        logger.info(f"Using CVX_UlyssesAttnProcessor, {layer_id=}")
        self.world_size = 1
        self.rank = 0
        if get_usp_group().world_size > 1:
            self.world_size = get_usp_group().world_size
            self.rank = get_usp_group().rank_in_group
        self.layer_id = layer_id
    @get_timer("ulyssesAttn")
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