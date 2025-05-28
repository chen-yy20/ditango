import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_func

from ditango.core.parallel_state import get_isp_group
from ditango.core.attention import proAttention
from ditango.logger import init_logger
from ditango.core.config import get_config
from ditango.baseline.cache import get_fusion_cache
from ditango.utils import get_timestep, split_tensor_uneven
from ditango.timer import get_timer



try:
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except ImportError:
    xFuserLongContextAttention = None
    

logger = init_logger(__name__)

class Attention(nn.Module):
    def __init__(self, layer_id = -1):
        super().__init__()
        self.config = get_config()
        self.layer_id = layer_id
        if not self.config.use_distrifusion:
            self.oAttn = proAttention(layer_id, get_isp_group())
    
    def attn_processor(self, attn_type):
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            return self.parallel_attn_func
        elif self.config.use_distrifusion:
            return self.distrifusion_attn_func
        else:
            return self.o_attn_func
        # else:
        #     raise Exception('Not supported attention type...')

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

    def distrifusion_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):
        timestep = get_timestep()
        cache = get_fusion_cache()
        
        with get_timer("df_comm"):
            kv_dict = cache.get_kv(self.layer_id)
            k = k.contiguous()
            v = v.contiguous()
            if kv_dict is None or timestep < 3: #full
                new_k = get_isp_group().all_gather(k, dim=1)
                new_v = get_isp_group().all_gather(v, dim=1)
                cache.store(self.layer_id, k, v)
            else:
                cached_k = kv_dict['k'].contiguous()
                cached_v = kv_dict['v'].contiguous()
                # logger.debug(f"L{self.layer_id} | cached_k shape: {cached_k.shape}, cached_v shape: {cached_v.shape}")
                cache.store(self.layer_id, k, v)
                stale_k = get_isp_group().all_gather(cached_k, dim=1)
                stale_v = get_isp_group().all_gather(cached_v, dim=1)
                stale_k_list = split_tensor_uneven(stale_k, get_isp_group().world_size, dim=1)
                stale_v_list = split_tensor_uneven(stale_v, get_isp_group().world_size, dim=1)
                stale_k_list[get_isp_group().rank_in_group] = k
                stale_v_list[get_isp_group().rank_in_group] = v
                new_k = torch.cat(stale_k_list, dim=1).contiguous()
                new_v = torch.cat(stale_v_list, dim=1).contiguous()
            
        # logger.debug(f"{q.shape=}, new_k shape: {new_k.shape}, new_v shape: {new_v.shape}")
        x = flash_attn_func(
            q,
            new_k,
            new_v,
        )
        x = x.to(q.dtype)
        
        return x
    