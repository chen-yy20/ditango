import torch
from typing import Optional, Tuple, List
import torch.distributed
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from .group_coordinate import GroupCoordinator

from .cache import proCache
from .config import get_config
from .stride_map import get_stride_map

from ..logger import init_logger
from ..timer import get_timer
from ..utils import update_out_and_lse, get_timestep


logger = init_logger(__name__)
    
    
class proAttention:
    def __init__(self, layer_id: int, isp_group: GroupCoordinator):
        self.group = isp_group
        self.isp_size = self.group.world_size
        self.global_rank = torch.distributed.get_rank()
        self.local_chunk_id = self.group.rank_in_group
        self.layer_id = layer_id
        self.target_chunk_id = self.local_chunk_id // self.isp_size # 初始化chunk的指针，表示计算到了哪个chunk，从0到isp_size-1一直循环
        self.cache = proCache(isp_size=self.isp_size, layer_id=layer_id)
        
        self.use_ringfusion = get_config().use_ringfusion
        self.small_ring_stride = 8
        self.measure_memory = False
        if self.layer_id == 0:
            logger.info(f"R{self.global_rank}L{layer_id} | Using RingFusion Attn.")
        
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
        timestep = get_timestep()
        # block size 就是 isp stride， 每次计算只计算一个block
        
        # =================== 0. Memory Check  =================
        curr_isp_stride = get_stride_map().get_curr_isp_stride(timestep, self.layer_id)
        next_isp_stride = get_stride_map().get_next_isp_stride(timestep, self.layer_id)
        
        # self.cache.pass_memory_constraint(next_isp_stride)
        if not self.cache.pass_memory_check(next_isp_stride, self.layer_id):
            next_isp_stride = self.isp_size
            
        assert self.isp_size % curr_isp_stride == 0, f"Does not support this ISP stride {curr_isp_stride} for SP size {self.isp_size}"
        
        total_block_num = self.isp_size // curr_isp_stride
        local_block_id = self.local_chunk_id // curr_isp_stride # 当前进程属于哪个block
        target_block_id = self.target_chunk_id // curr_isp_stride # 表示本轮需要计算的block id
        
        # =================== 1. Update Cached blocks =================
        if curr_isp_stride == self.isp_size: # 满，刷新指针, 细粒度存储
            self.target_chunk_id = self.local_chunk_id
        else: # 不满，先更新cached block
            for i in range(total_block_num - 1):
                cached_block_id = (target_block_id + 1 + i) % total_block_num
                cache_out, cache_lse = self.cache.get_block(cached_block_id)
                # if self.layer_id == 0:
                #     logger.debug(f"{timestep}-{self.layer_id} | trying to get {cached_block_id=}, {total_block_num=}")
                if cache_lse is not None:
                    out, lse = update_out_and_lse(out, lse, cache_out, cache_lse)
                    
        
        # ================== 2. Get First KV Chunk ======================= 
        if self.target_chunk_id != self.local_chunk_id: # 手头上的chunk不是目标chunk，要计算的是相对应的block
            # 准备数据和确认目标进程
            first_key, first_value = None, None   
            block_id_stride = (target_block_id - local_block_id) % total_block_num # 接收block和当前block的距离
            send_block_id = (local_block_id - block_id_stride) % total_block_num # 发送block的id    
            inner_block_rank = self.local_chunk_id % curr_isp_stride # 当前进程在block内的rank
        
            recv_rank = target_block_id * curr_isp_stride + inner_block_rank
            send_rank = send_block_id * curr_isp_stride + inner_block_rank
            
            first_key, first_value = self.async_ring_p2p_commit(
                (key, value),
                src_rank=recv_rank,
                dst_rank=send_rank
            )
            
            # ============ overlap with local KV calculate ==============
            block_out, block_lse, _  = flash_attn_func(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    return_attn_probs=True,
                )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            key, value = self.async_ring_p2p_wait_and_update((first_key, first_value))
            
        # ================= 3. Target KV Block Calculate ====================
        num_chunks_to_update = 1 if curr_isp_stride == self.isp_size else curr_isp_stride
        
        num_large_ring = 1
        small_ring_stride = curr_isp_stride
        if curr_isp_stride > self.small_ring_stride and self.use_ringfusion:
            assert curr_isp_stride % self.small_ring_stride == 0, f"Does not support this ISP stride {curr_isp_stride} for overlap ring strategy"
            num_large_ring = curr_isp_stride // self.small_ring_stride
            small_ring_stride = self.small_ring_stride
            
        small_ring_offset= self.local_chunk_id // small_ring_stride * small_ring_stride # stride小环起始chunk的id
        large_ring_offset = self.local_chunk_id // curr_isp_stride * curr_isp_stride # stride大环起始chunk的id
        
        intra_next_rank = (self.local_chunk_id - 1) % small_ring_stride + small_ring_offset
        intra_prev_rank = (self.local_chunk_id + 1) % small_ring_stride + small_ring_offset
        inter_next_rank =  (self.local_chunk_id - small_ring_stride) % curr_isp_stride + large_ring_offset
        inter_prev_rank = (self.local_chunk_id + small_ring_stride) % curr_isp_stride + large_ring_offset
        
        fresh_out, fresh_lse = None, None
        finished_chunk_cnt = 0
        for inter_node_step in range(num_large_ring):
            for intra_node_step in range(small_ring_stride):
                if intra_node_step + 1 != small_ring_stride: 
                    next_k, next_v = self.async_ring_p2p_commit(
                        (key, value),
                        src_rank=intra_prev_rank,
                        dst_rank=intra_next_rank,
                    )
                    # if timestep > 4 and self.global_rank==14:
                    #     logger.debug(f"{timestep}-{self.layer_id} |IR-{self.local_chunk_id} | SMALL_STEP {intra_node_step} |  {intra_prev_rank} -> rank:{self.local_chunk_id} -> {intra_next_rank}")
                elif inter_node_step + 1 != num_large_ring:
                    
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
                
                if local_block_id == target_block_id and intra_node_step == 0 and inter_node_step ==0:  # 计算了自身
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse) # 直接更新，不做储存
                else:
                    fresh_out, fresh_lse = update_out_and_lse(fresh_out, fresh_lse, block_out, block_lse)
                    
                finished_chunk_cnt += 1
                
                if intra_node_step + 1 != small_ring_stride or inter_node_step + 1 != num_large_ring:
                    key, value = self.async_ring_p2p_wait_and_update((next_k,next_v))
                    
                # =============== 4. Update Cache ================
                if finished_chunk_cnt == num_chunks_to_update: # 完成了本block的计算 
                    if fresh_lse is not None:
                        fresh_lse = fresh_lse.squeeze(-1).transpose(-1,-2)
                        if next_isp_stride != self.isp_size:
                            if curr_isp_stride == self.isp_size:
                                self.cache.store_block(self.target_chunk_id, fresh_out, fresh_lse) # 细粒度存储
                            else:
                                self.cache.store_block(target_block_id, fresh_out, fresh_lse) # 只存储单个block
                                
                        out, lse = update_out_and_lse(out, lse, fresh_out, fresh_lse)
                    self.target_chunk_id = (self.target_chunk_id + finished_chunk_cnt) % self.isp_size
                    fresh_out, fresh_lse = None, None
                    finished_chunk_cnt = 0
                    
        # ================= 5. Cache Management =================
        next_target_block_id = self.target_chunk_id // next_isp_stride
        self.cache.update_cache_blocks(next_isp_stride, next_target_block_id)
        # if self.use_ringfusion and  self.layer_id == 0 and self.global_rank == 0:
        #     print(f"R{self.global_rank}L{self.layer_id} | Cache updated with {next_isp_stride=}, {next_target_block_id=}", flush=True)
        #     self.cache.report_cache_status(self.layer_id)
                    
        if self.global_rank == 0 and timestep == 0 and self.layer_id == 10:
            logger.info(f"******************* Processing out_shape: {out.shape=}*******************")
    
        stride_map = get_stride_map()
        stride_map.record_out_redundancy(timestep=timestep,
                                            layer_id=self.layer_id,
                                            curr_out=out)
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
        # Note: Varlen version is design for HunyuanVideo
        if cu_seqlens_q is None and cu_seqlens_kv is None and max_seqlen_q is None and max_seqlen_kv is None:
            logger.warning("cu_seqlens_q not provided, using normal attention instead.")
            return self.attn(query, key, value)
        # Note: For varlen_attn, input qkv shape should be (bxs, a, d)
        
        next_k, next_v, next_cu_seqlens_kv = None, None, None
        out, lse = None, None
        timestep = get_timestep()
        # if self.measure_memory and timestep == 0:
        #     torch.cuda.reset_peak_memory_stats()
        
        # block size 就是 isp stride， 每次计算只计算一个block
        
        # =================== 0. Initialize  =================
        curr_isp_stride = get_stride_map().get_curr_isp_stride(timestep, self.layer_id)
        next_isp_stride = get_stride_map().get_next_isp_stride(timestep, self.layer_id)
        
        if not self.cache.pass_memory_check(next_isp_stride, self.layer_id):
            next_isp_stride = self.isp_size
        
        assert self.isp_size % curr_isp_stride == 0, f"Does not support this ISP stride {curr_isp_stride} for SP size {self.isp_size}"
        total_block_num = self.isp_size // curr_isp_stride
        local_block_id = self.local_chunk_id // curr_isp_stride # 当前进程属于哪个block
        target_block_id = self.target_chunk_id // curr_isp_stride # 表示本轮需要计算的block id
        
        # self.target_chunk_id = self.local_chunk_id # fix: breakdown evaluate
        # =================== 1. Update Cached blocks =================
        if curr_isp_stride == self.isp_size: # 满，刷新指针, 细粒度存储
            self.target_chunk_id = self.local_chunk_id
        else: # 不满，先更新cached block
            for i in range(total_block_num - 1):
                cached_block_id = (target_block_id + 1 + i) % total_block_num
                cache_out, cache_lse = self.cache.get_block(cached_block_id)
                # if self.layer_id == 0:
                #     logger.debug(f"{timestep}-{self.layer_id} | trying to get {cached_block_id=}, {total_block_num=}")
                if cache_lse is not None:
                    # if cache_lse.dim() == 4:
                    #     cache_lse = cache_lse.squeeze(-1).transpose(-1,-2) # [seqlen, head, 1] -> [head, seqlen]
                    out, lse = update_out_and_lse(out, lse, cache_out, cache_lse)
                    if lse.dim() == 4:
                        lse = lse.squeeze(-1).transpose(-1,-2)
                    
                    
        
        # ================== 2. Get First KV Chunk ======================= 
        if self.target_chunk_id != self.local_chunk_id: # 手头上的chunk不是目标chunk，要计算的是相对应的block
            # 准备数据和确认目标进程
            first_key, first_value, first_cu_seqlens_kv = None, None, None   
            block_id_stride = (target_block_id - local_block_id) % total_block_num # 接收block和当前block的距离
            send_block_id = (local_block_id - block_id_stride) % total_block_num # 发送block的id    
            inner_block_rank = self.local_chunk_id % curr_isp_stride # 当前进程在block内的rank
           
            recv_rank = target_block_id * curr_isp_stride + inner_block_rank
            send_rank = send_block_id * curr_isp_stride + inner_block_rank
            
            first_key, first_value, first_cu_seqlens_kv = self.async_ring_p2p_commit(
                (key, value, cu_seqlens_kv),
                src_rank=recv_rank,
                dst_rank=send_rank
            )
            
            # ============ overlap with local KV calculate ==============
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
            key, value, cu_seqlens_kv = self.async_ring_p2p_wait_and_update((first_key, first_value, first_cu_seqlens_kv))
            
        # ================= 3. Target KV Block Calculate ====================
        num_chunks_to_update = 1 if curr_isp_stride == self.isp_size else curr_isp_stride
        
        num_large_ring = 1
        small_ring_stride = curr_isp_stride
        if curr_isp_stride > self.small_ring_stride and self.use_ringfusion:
            assert curr_isp_stride % self.small_ring_stride == 0, f"Does not support this ISP stride {curr_isp_stride} for overlap ring strategy"
            num_large_ring = curr_isp_stride // self.small_ring_stride
            small_ring_stride = self.small_ring_stride
            
        small_ring_offset= self.local_chunk_id // small_ring_stride * small_ring_stride # stride小环起始chunk的id
        large_ring_offset = self.local_chunk_id // curr_isp_stride * curr_isp_stride # stride大环起始chunk的id
        
        intra_next_rank = (self.local_chunk_id - 1) % small_ring_stride + small_ring_offset
        intra_prev_rank = (self.local_chunk_id + 1) % small_ring_stride + small_ring_offset
        inter_next_rank =  (self.local_chunk_id - small_ring_stride) % curr_isp_stride + large_ring_offset
        inter_prev_rank = (self.local_chunk_id + small_ring_stride) % curr_isp_stride + large_ring_offset
        
        fresh_out, fresh_lse = None, None
        finished_chunk_cnt = 0
        for inter_node_step in range(num_large_ring):
            for intra_node_step in range(small_ring_stride):
                if intra_node_step + 1 != small_ring_stride: 
                    # logger.info(f"{timestep}-{self.layer_id} |IR-{self.local_chunk_id} | SMALL_STEP {intra_node_step} |  {intra_prev_rank} -> rank:{self.local_chunk_id} -> {intra_next_rank}, trying to send {key.shape=}, {value.shape=}, {cu_seqlens_kv.shape=}")
                    next_k, next_v, next_cu_seqlens_kv = self.async_ring_p2p_commit(
                        (key, value, cu_seqlens_kv),
                        src_rank=intra_prev_rank,
                        dst_rank=intra_next_rank,
                    )
                elif inter_node_step + 1 != num_large_ring:
                    
                    next_k, next_v, next_cu_seqlens_kv = self.async_ring_p2p_commit(
                        (key, value, cu_seqlens_kv),
                        src_rank=inter_prev_rank,
                        dst_rank=inter_next_rank,
                    )
                
                # logger.debug(f"{timestep}-{self.layer_id} |IR-{self.local_chunk_id} | SMALL_STEP {intra_node_step} |  {query.shape=}, {key.shape=}, {value.shape=}, {cu_seqlens_q=}, {cu_seqlens_kv=}, {max_seqlen_q=}, {max_seqlen_kv=}")
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
                
                if local_block_id == target_block_id and intra_node_step == 0 and inter_node_step == 0:  # 计算了自身
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse) # 直接更新，不做储存
                else:
                    fresh_out, fresh_lse = update_out_and_lse(fresh_out, fresh_lse, block_out, block_lse)
                    
                finished_chunk_cnt += 1
                
                if intra_node_step + 1 != small_ring_stride or inter_node_step + 1 != num_large_ring:
                    key, value, cu_seqlens_kv = self.async_ring_p2p_wait_and_update((next_k, next_v, next_cu_seqlens_kv))
                    # logger.info(f"{timestep}-{self.layer_id} |IR-{self.local_chunk_id} | SMALL_STEP {intra_node_step} |  {intra_prev_rank} -> rank:{self.local_chunk_id} -> {intra_next_rank}, Succesfully recv {key.shape=}, {value.shape=}, {cu_seqlens_kv.shape=}")
                # =============== 4. Update Cache ================
                if finished_chunk_cnt == num_chunks_to_update: # 完成了本block的计算 
                    if fresh_lse is not None:
                        fresh_lse = fresh_lse.squeeze(-1).transpose(-1,-2)
                        if next_isp_stride != self.isp_size:
                            if curr_isp_stride == self.isp_size:
                                self.cache.store_block(self.target_chunk_id, fresh_out, fresh_lse) # 细粒度存储
                            else:
                                self.cache.store_block(target_block_id, fresh_out, fresh_lse) # 只存储单个block
                                
                        out, lse = update_out_and_lse(out, lse, fresh_out, fresh_lse)
                    self.target_chunk_id = (self.target_chunk_id + finished_chunk_cnt) % self.isp_size
                    fresh_out, fresh_lse = None, None
                    finished_chunk_cnt = 0
                    
        # ================= 5. Cache Management =================
        next_target_block_id = self.target_chunk_id // next_isp_stride
        self.cache.update_cache_blocks(next_isp_stride, next_target_block_id)
        # if self.use_ringfusion and self.global_rank == 0 and self.layer_id == 40:
        #     print(f"R{self.global_rank}L{self.layer_id} | Cache updated with {next_isp_stride=}, {next_target_block_id=}", flush=True)
        #     self.cache.report_cache_status(self.layer_id)
                    
        # if self.global_rank == 0 and timestep == 0 and self.layer_id == 10:
        #     logger.info(f"******************* Processing out_shape: {out.shape=}*******************")
        if self.measure_memory and timestep == 0:
            peak_memory = torch.cuda.max_memory_allocated()
            logger.info(f"Peak Attn memory usage: {peak_memory / 1024**2:.2f} MB")
        stride_map = get_stride_map()
        stride_map.record_out_redundancy(timestep=timestep,
                                            layer_id=self.layer_id,
                                            curr_out=out)
        return out
    
