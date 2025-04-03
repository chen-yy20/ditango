import torch
from typing import Optional, Tuple, List
import torch.distributed
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from .group_coordinate import GroupCoordinator
from .feature_cache import get_cache, exist_oCache
from ...logger import init_logger
from ...timer import get_timer
from ..diff_sensor import get_diff_sensor

import nvtx
from contextlib import contextmanager

enable_nsys = True

@contextmanager
def nvtx_range(name, domain="CUSTOM"):
    if enable_nsys:
        nvtx.push_range(name, domain=domain)
    try:
        yield
    finally:
        if enable_nsys:
            nvtx.pop_range(domain=domain)


logger = init_logger(__name__)
gpus_per_node = 4


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
    
    
    
class oAttention:
    def __init__(self, layer_id: int, isp_group: GroupCoordinator):
        self.group = isp_group
        self.isp_size = self.group.world_size
        self.global_rank = torch.distributed.get_rank()
        self.isp_rank = self.group.rank_in_group
        self.layer_id = layer_id
        self.target_block_id = self.isp_rank // self.isp_size
        self.cache = None
        if exist_oCache(): 
            self.cache = get_cache()
        
        self.use_overlap_ring = True
        
    def async_ring_p2p_commit(self, tensors: Tuple[torch.Tensor, ...], src_rank: int, dst_rank: int):
        """Set up ring communication for sending and receiving tensors asynchronously.
    
        Args:
            tensors: Tuple of tensors to be sent
            dst_rank: Destination rank to send tensors to
            src_rank: Source rank to receive tensors from
            
        Returns:
            Tuple[torch.Tensor, ...]: Tuple of tensors to be received after wait
        """
        recv_tensors = []
        
        for tensor in tensors:
            send_tensor = tensor.detach().clone()
            recv_size = send_tensor.shape
            recv_dtype = send_tensor.dtype
            self.group.p2p_isend(send_tensor, dst=dst_rank)
            next_tensor = self.group.p2p_irecv(size=recv_size, dtype=recv_dtype, src=src_rank)
            recv_tensors.append(next_tensor)
            
        self.group.p2p_commit()
        return tuple(recv_tensors)
    
    def async_ring_p2p_wait_and_update(self, recv_tensors: Tuple[torch.Tensor, ...]):
        """Wait for asynchronous communication to complete and return received tensors.
    
        Args:
            recv_tensors: Tuple of tensors returned from async_ring_p2p_commit
            
        Returns:
            Tuple[torch.Tensor, ...]: Tuple of received tensors after communication completes
        """
        self.group.p2p_wait()
        return recv_tensors
    @get_timer("oAttn")
    def attn(
        self,
        query: torch.Tensor, 
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        # Note: input size should be (b, s, a, d)
        next_k, next_v = None, None
        out, lse = None, None
        
        if self.cache is None:
            curr_isp_stride = self.isp_size
            next_isp_stride = self.isp_size
            q_block_id = self.isp_rank // self.isp_size
            total_block_num = 1
        
        # =================== update prev =================
        if self.cache is not None:
            self.cache.set_isp_size(isp_size=self.isp_size)
            curr_isp_stride = self.cache.get_curr_isp_stride(self.layer_id)
            next_isp_stride = self.cache.get_next_isp_stride(self.layer_id)
            
            assert next_isp_stride == self.isp_size or curr_isp_stride == self.isp_size or next_isp_stride == curr_isp_stride
            
            if next_isp_stride == self.isp_size:
                total_block_num = self.isp_size // curr_isp_stride # 刷新
                q_block_id = self.isp_rank // curr_isp_stride 
            elif next_isp_stride < self.isp_size:
                total_block_num = self.isp_size // next_isp_stride
                q_block_id = self.isp_rank // next_isp_stride

            if self.isp_size == curr_isp_stride: # 满，从头开始算
                self.target_block_id = self.isp_rank // next_isp_stride 
            
            else: # 不满，先更新存好的内容
                for i in range(total_block_num - 1):
                    cached_block_id = (self.target_block_id + 1 + i) % total_block_num
                    cache_out, cache_lse = self.cache.get_block(self.layer_id, cached_block_id)
                    if cache_out is not None:
                        out, lse = update_out_and_lse(out, lse, cache_out, cache_lse)
                        
            # 下一轮有几个block，哪些要算，哪些要取
            self.cache.purge_cache(self.layer_id, self.target_block_id, total_block_num)
        # ================== get first kv ======================= 
        # logger.debug(f"{self.cache.timestep}-{self.layer_id} | {self.isp_rank=} {self.target_block_id=} {q_block_id=}")
        if self.cache is not None and self.target_block_id != q_block_id:
            # 准备数据和确认目标进程
            first_key, first_value = None, None   
            block_id_stride = (self.target_block_id - q_block_id) % total_block_num
            send_block_id = (q_block_id - block_id_stride) % total_block_num
            isp_block_rank = self.isp_rank % curr_isp_stride
           
            recv_rank = self.target_block_id * curr_isp_stride + isp_block_rank
            send_rank = curr_isp_stride * send_block_id + isp_block_rank
            
            first_key, first_value = self.async_ring_p2p_commit(
                (key, value),
                src_rank=recv_rank,
                dst_rank=send_rank
            )
            
            # ============ overlap part begin ==============
            block_out, block_lse, _  = flash_attn_func(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    return_attn_probs=True,
                )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            # ============ overlap part end ==============
            key, value = self.async_ring_p2p_wait_and_update((first_key, first_value))
            
        # ================= ISP Loop ====================
        large_ring = 1
        full_isp_stride = curr_isp_stride
        if curr_isp_stride > gpus_per_node and self.use_overlap_ring:
            assert curr_isp_stride % gpus_per_node == 0, f"Does not support this ISP stride {curr_isp_stride} for overlap ring strategy"
            large_ring = curr_isp_stride // gpus_per_node
            curr_isp_stride = gpus_per_node
            
        q_offset = self.isp_rank // curr_isp_stride * curr_isp_stride # stride小环起始chunk的id
        large_q_offset = self.isp_rank // full_isp_stride * full_isp_stride # stride大环起始chunk的id
        next_rank = (self.isp_rank - 1) % curr_isp_stride + q_offset # 在环内做ring
        prev_rank = (self.isp_rank + 1) % curr_isp_stride + q_offset
        fresh_out, fresh_lse = None, None
        merge_block_cnt = 0
        for inter_node_step in range(large_ring):
            for intra_node_step in range(curr_isp_stride):
                if intra_node_step + 1 != curr_isp_stride: 
                    next_k, next_v = self.async_ring_p2p_commit(
                        (key, value),
                        src_rank=prev_rank,
                        dst_rank=next_rank,
                    )
                    if self.cache.timestep > 4 and self.global_rank==14:
                        logger.debug(f"{self.cache.timestep}-{self.layer_id} |IR-{self.isp_rank} | SMALL_STEP {intra_node_step} |  {prev_rank} -> rank:{self.isp_rank} -> {next_rank}")
                elif inter_node_step + 1 != large_ring:
                    inter_next_rank =  (self.isp_rank - gpus_per_node) % full_isp_stride + large_q_offset
                    inter_prev_rank = (self.isp_rank + gpus_per_node) % full_isp_stride + large_q_offset
                    next_k, next_v = self.async_ring_p2p_commit(
                        (key, value),
                        src_rank=inter_prev_rank,
                        dst_rank=inter_next_rank,
                    )
                block_out, block_lse, _  = flash_attn_func(
                        query,
                        key,
                        value,
                        dropout_p=0.0,
                        return_attn_probs=True,
                    )
                if q_block_id == self.target_block_id and intra_node_step == 0:  # 计算了自身
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse) # 不存自己
                else:
                    fresh_out, fresh_lse = update_out_and_lse(fresh_out, fresh_lse, block_out, block_lse)
                    
                merge_block_cnt += 1
                
                if intra_node_step + 1 != curr_isp_stride or inter_node_step + 1 != large_ring:
                    key, value = self.async_ring_p2p_wait_and_update((next_k,next_v))
                    
                # =============== update Cache ================
                if merge_block_cnt == next_isp_stride:
                    # if self.global_rank == 0 and self.layer_id == 15:
                    #     logger.info(f"{self.cache.timestep}-{self.layer_id} | trying to store {self.target_block_id=}")
                    if fresh_lse is not None:
                        fresh_lse = fresh_lse.squeeze(-1).transpose(-1,-2)
                        if self.cache is not None:
                            self.cache.store_block(self.layer_id, self.target_block_id, fresh_out, fresh_lse)
                        out, lse = update_out_and_lse(out, lse, fresh_out, fresh_lse)
                    fresh_out, fresh_lse = None, None
                    merge_block_cnt = 0
                    self.target_block_id = (self.target_block_id + 1) % total_block_num
                    # logger.debug(f"{self.cache.timestep}-{self.layer_id} | {self.isp_rank=} {total_block_num=} , Updated {self.target_block_id=}")
        if self.global_rank == 0 and self.cache.timestep == 0 and self.layer_id == 10:
            logger.info(f"******************* Processing out_shape: {out.shape=}*******************")
        if self.global_rank == 0 and get_diff_sensor() is not None:
            get_diff_sensor().log_layer(current_step=self.cache.timestep,
                                        layer_id=self.layer_id,
                                        current_out=out,
                                        current_lse=lse
                                        )
            max_memory = torch.cuda.max_memory_allocated() 
            memory_bytes = out.element_size() * out.nelement()
            logger.info(f"{self.cache.timestep}-{self.layer_id} |GPU内存: {memory_bytes/1024/1024:.2f}MB | max: {max_memory / 1024**2:.2f} MB")
        return out
    
    @get_timer("varlen oAttn")
    def varlen_attn(
        self,
        query,
        key,
        value,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        batch_size=1,
    ):
        if cu_seqlens_q is None and cu_seqlens_kv is None and max_seqlen_q is None and max_seqlen_kv is None:
            return self.attn(query, key, value)
        # Note: For varlen_attn, input qkv shape should be (bxs, a, d)
        
        # =================== Initialize =================
        next_k, next_v = None, None
        out, lse = None, None
        
        if self.cache is None:
            curr_isp_stride = self.isp_size
            next_isp_stride = self.isp_size
            q_block_id = self.isp_rank // self.isp_size
            total_block_num = 1
        
        # =================== update prev =================
        if self.cache is not None:
            self.cache.set_isp_size(isp_size=self.isp_size)
            curr_isp_stride = self.cache.get_curr_isp_stride(self.layer_id)
            next_isp_stride = self.cache.get_next_isp_stride(self.layer_id)
            
            assert next_isp_stride == self.isp_size or curr_isp_stride == self.isp_size or next_isp_stride == curr_isp_stride
            
            if next_isp_stride == self.isp_size:
                total_block_num = self.isp_size // curr_isp_stride
                q_block_id = self.isp_rank // curr_isp_stride
            elif next_isp_stride < self.isp_size:
                total_block_num = self.isp_size // next_isp_stride
                q_block_id = self.isp_rank // next_isp_stride

            if self.isp_size == curr_isp_stride: # 本轮要算整个ring，直接从头开始算
                self.target_block_id = self.isp_rank // next_isp_stride 
            
            else: # 本轮只算部分ring，先更新存好的内容
                for i in range(total_block_num - 1):
                    cached_block_id = (self.target_block_id + 1 + i) % total_block_num
                    cache_out, cache_lse = self.cache.get_block(self.layer_id, cached_block_id)
                    if cache_out is not None:
                        out, lse = update_out_and_lse(out, lse, cache_out, cache_lse)
                    # else:
                    #     if self.global_rank == 0 and self.layer_id == 15:
                    #         logger.error(f"{self.cache.timestep}-{self.layer_id} | trying to get {cached_block_id=} but found None.")
                        
            # Get INFO: 下一轮有几个block，哪些要算，哪些要取
            self.cache.purge_cache(self.layer_id, self.target_block_id, total_block_num)
            
        # ================== get first kv ======================= 
        if self.cache is not None and self.target_block_id != q_block_id:
            # 准备数据和确认目标进程
            first_key, first_value, first_cu_seqlens_kv = None, None, None   
            block_id_stride = (self.target_block_id - q_block_id) % total_block_num
            send_block_id = (q_block_id - block_id_stride) % total_block_num
            isp_block_rank = self.isp_rank % curr_isp_stride
            
            send_rank = curr_isp_stride * send_block_id + isp_block_rank
            recv_rank = self.target_block_id * curr_isp_stride + isp_block_rank
            
            first_key, first_value, first_cu_seqlens_kv = self.async_ring_p2p_commit(
                (key, value, cu_seqlens_kv),
                src_rank=recv_rank,
                dst_rank=send_rank,
            )
            
            # if self.global_rank == 0 and self.layer_id == 15:
            #     logger.debug(f"{self.cache.timestep}-{self.layer_id} | Try to get target block {self.target_block_id}'s kv |  {recv_rank} -> rank:{self.isp_rank} -> {send_rank}")
            
            block_out, block_lse, _  = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    dropout_p=0.0,
                    return_attn_probs=True,
                )
            
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            key, value, cu_seqlens_kv = self.async_ring_p2p_wait_and_update(
                (first_key, first_value, first_cu_seqlens_kv)
            )
        
        # ================= ISP Loop ====================
        large_ring = 1
        full_isp_stride = curr_isp_stride
        if curr_isp_stride > gpus_per_node and self.use_overlap_ring:
            assert curr_isp_stride % gpus_per_node == 0, f"Does not support this ISP stride {curr_isp_stride} for overlap ring strategy"
            large_ring = curr_isp_stride // gpus_per_node
            curr_isp_stride = gpus_per_node
            
        q_offset = self.isp_rank // curr_isp_stride * curr_isp_stride # stride小环起始chunk的id
        large_q_offset = self.isp_rank // full_isp_stride * full_isp_stride # stride大环起始chunk的id
        next_rank = (self.isp_rank - 1) % curr_isp_stride + q_offset
        prev_rank = (self.isp_rank + 1) % curr_isp_stride + q_offset
        # if self.rank in [0] and self.layer_id == 20:
        #     logger.debug(f"{cache.timestep} | ISP-{self.kv_block_id=} | {q_offset=} {prev_rank} -> rank:{self.rank} -> {next_rank}")
        fresh_out, fresh_lse = None, None
        merge_block_cnt = 0
        for inter_node_step in range(large_ring):
            for intra_node_step in range(curr_isp_stride):
                if intra_node_step + 1 != curr_isp_stride: 
            
                    next_k, next_v, next_cu_seqlens_kv = self.async_ring_p2p_commit(
                        (key, value, cu_seqlens_kv),
                        src_rank=prev_rank,
                        dst_rank=next_rank
                    )
                    
                elif inter_node_step + 1 != large_ring:
                    
                    inter_next_rank =  (self.isp_rank - gpus_per_node) % full_isp_stride + large_q_offset
                    inter_prev_rank = (self.isp_rank + gpus_per_node) % full_isp_stride + large_q_offset
                    next_k, next_v, next_cu_seqlens_kv = self.async_ring_p2p_commit(
                        (key, value, cu_seqlens_kv),
                        src_rank=inter_prev_rank,
                        dst_rank=inter_next_rank
                    )
                
                block_out, block_lse, _  = flash_attn_varlen_func(
                        query,
                        key,
                        value,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        dropout_p=0.0,
                        return_attn_probs=True,
                    )
                
                if q_block_id == self.target_block_id and intra_node_step == 0:  # 计算了自身
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse) # 不存自己
                else:
                    fresh_out, fresh_lse = update_out_and_lse(fresh_out, fresh_lse, block_out, block_lse)
                    
                merge_block_cnt += 1
                
                if intra_node_step + 1 != curr_isp_stride or inter_node_step + 1 != large_ring:
                    key, value, cu_seqlens_kv = self.async_ring_p2p_wait_and_update(
                        (next_k, next_v, next_cu_seqlens_kv)
                    )
                    
                # =============== update Cache ================
                if merge_block_cnt == next_isp_stride:
                    if fresh_lse is not None:
                        fresh_lse = fresh_lse.squeeze(-1).transpose(-1,-2)
                        if self.cache is not None:
                            self.cache.store_block(self.layer_id, self.target_block_id, fresh_out, fresh_lse)
                        out, lse = update_out_and_lse(out, lse, fresh_out, fresh_lse)
                    fresh_out, fresh_lse = None, None
                    merge_block_cnt = 0
                    self.target_block_id = (self.target_block_id + 1) % total_block_num
                
        # ================= store output and lse for analysis ===================
        if self.global_rank == 0 and get_diff_sensor() is not None:
            get_diff_sensor().log_layer(current_step=self.cache.timestep,
                                        layer_id=self.layer_id,
                                        current_out=out,
                                        current_lse=lse
                                        )
            max_memory = torch.cuda.max_memory_allocated() 
            memory_bytes = out.element_size() * out.nelement()
            logger.info(f"{self.cache.timestep}-{self.layer_id} |GPU内存: {memory_bytes/1024/1024:.2f}MB | max: {max_memory / 1024**2:.2f} MB")
        
        return out
    
