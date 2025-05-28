from typing import Dict, Optional, Union, Tuple, List
import torch

from ..logger import init_logger
from ..utils import update_out_and_lse, get_timestep
from .stride_map import get_stride_map

logger = init_logger(__name__)
class proCache:
    
    def __init__(self, isp_size, layer_id):
        self.isp_size = isp_size
        self.curr_isp_stride = isp_size
        self.curr_block_num = isp_size
        self.layer_id = layer_id
       
        self.out_block_cache = [None] * self.curr_block_num
        self.lse_block_cache = [None] * self.curr_block_num
        self.full_out_cache = [None]
                
        self.block_size_mb = None
        self.memory_constraint = 0.72 * torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        
    def get_curr_isp_stride(self, layer_id: int):
        isp_stride = int(get_stride_map()[get_timestep(), layer_id].item())
        return isp_stride
    
    def _merge_blocks(self, new_block_num: int):
        # logger.debug(f"Merging blocks from {self.curr_block_num} to {new_block_num}")
        assert self.curr_block_num % new_block_num == 0, f"Invalid block number {new_block_num} for merging."
        chunks_to_merge = self.curr_block_num // new_block_num
        new_out_block_cache, new_lse_block_cache = [None] * new_block_num, [None] * new_block_num
        for i in range(new_block_num):
            merged_block_out, merged_block_lse = None, None
            for j in range(chunks_to_merge):
                block_id = i * chunks_to_merge + j
                cached_out = self.out_block_cache[block_id]
                cached_lse = self.lse_block_cache[block_id]
                if cached_lse is not None:
                    if cached_lse.dim() == 4:
                        cached_lse = cached_lse.squeeze(-1).transpose(-1,-2)
                    merged_block_out, merged_block_lse = update_out_and_lse(merged_block_out, merged_block_lse, cached_out, cached_lse)
            new_out_block_cache[i] = merged_block_out
            new_lse_block_cache[i] = merged_block_lse
        self.out_block_cache = new_out_block_cache
        self.lse_block_cache = new_lse_block_cache
        # logger.debug(f"Blocks merged successfully, New cache len {len(self.out_block_cache)}")
        
    def update_cache_blocks(self, new_isp_stride: int, next_target_block_id: int):
        # if self.layer_id == 0:
        #     logger.info(f"Updating cache blocks: {self.curr_isp_stride=}, new_isp_stride {new_isp_stride}, next_target_block_id {next_target_block_id}")
        # next_target_block_id = 0 # USAGE: Uncomment when processing Time-first break down evaluate
        if new_isp_stride == self.isp_size: # 全面刷新
            self.curr_isp_stride = self.isp_size
            self.curr_block_num = self.isp_size
            self.out_block_cache = [None] * self.curr_block_num
            self.lse_block_cache = [None] * self.curr_block_num
        elif new_isp_stride == 0: # 只留一个输出位
            self.curr_isp_stride = 0
            self.curr_block_num = 1
            self.out_block_cache = [None] * self.curr_block_num
            self.lse_block_cache = [None] * self.curr_block_num
        else: 
            new_block_num = self.isp_size // new_isp_stride
            assert new_block_num <= self.curr_block_num, "Invalid ISP stride."
            self.curr_isp_stride = new_isp_stride
            if new_block_num != self.curr_block_num:
                # merge blocks
                self._merge_blocks(new_block_num)
                self.curr_block_num = new_block_num
            self.out_block_cache[next_target_block_id] = None
            self.lse_block_cache[next_target_block_id] = None
                
   
    def get_block(self, block_id: int):
        if block_id >= len(self.out_block_cache):
            logger.error(f"{get_timestep()} | Cache is not ready but trying to get cached value.")
        cached_out = self.out_block_cache[block_id] 
        cached_lse = self.lse_block_cache[block_id]
        return cached_out, cached_lse
    
    def store_block(self, block_id, out, lse):
        self.out_block_cache[block_id] = out
        self.lse_block_cache[block_id] = lse
    
    def clear(self):
        logger.warning("============== Clear oCache =================")
    
    def set_block_size_mb(self, tensor):
        original_shape = list(tensor.shape)
        
        new_shape = original_shape.copy()
        new_shape[-1] += 1
        
        expanded_tensor = torch.zeros(new_shape, dtype=torch.float32, device=tensor.device)
        
        if self.block_size_mb is None:
            block_size_mb = expanded_tensor.element_size() * expanded_tensor.nelement() / (1024 ** 2)
            self.block_size_mb = block_size_mb
        logger.debug(f"{expanded_tensor.shape=} | Block size MB: {self.block_size_mb}")
        return self.block_size_mb
    
    def pass_memory_check(self, next_isp_stride: int, layer_id: int):
        if next_isp_stride == self.isp_size:
            return True
        curr_active_block_num = 0
        for i, block in enumerate(self.out_block_cache):
            if block is not None:
                curr_active_block_num += 1
                # calculate block_size_mb
                if self.block_size_mb is None:
                    block_size_mb = block.element_size() * block.nelement() / (1024 ** 2)
                    if self.lse_block_cache[i] is not None:
                        block_size_mb += self.lse_block_cache[i].element_size() * self.lse_block_cache[i].nelement() / (1024 ** 2)
                    self.block_size_mb = block_size_mb
        if self.block_size_mb is None:
            return True
        predict_active_block_num = self.isp_size // next_isp_stride
        if predict_active_block_num <= curr_active_block_num:
            return True
        else:
            predict_add_mem = self.block_size_mb * (predict_active_block_num - curr_active_block_num)
            curr_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            # logger.info(f"T{get_timestep()}L{layer_id} |Curr_Stride {self.curr_isp_stride} | Next_Stride {next_isp_stride} | Memory check: Add {predict_add_mem} MB")
            if curr_memory + predict_add_mem > self.memory_constraint:
                logger.warning(f"T{get_timestep()}L{layer_id} | Memory check failed: {curr_memory + predict_add_mem} > {self.memory_constraint}, setting stride {next_isp_stride} to {self.isp_size}")
                return False
            return True
        
        
    def report_cache_status(self, layer_id: int):
        """
        Report the current status of the cache including:
        - Cache configuration
        - Block storage status
        - Memory usage information
        - Shape information of cached blocks
        """
        # 获取当前时间步
        timestep = get_timestep()
        
        # 计算已缓存的块数
        cached_blocks = sum(1 for out in self.out_block_cache if out is not None)
        
        # 计算单个块大小并找到第一个非空块作为示例
        block_size_mb = 0
        example_out = None
        example_lse = None
        
        for block_id in range(len(self.out_block_cache)):
            out = self.out_block_cache[block_id]
            lse = self.lse_block_cache[block_id]
            if out is not None:
                example_out = out
                example_lse = lse
                
                block_size_bytes = out.element_size() * out.nelement()
                if lse is not None:
                    block_size_bytes += lse.element_size() * lse.nelement()
                block_size_mb = block_size_bytes / (1024 * 1024)
                if self.block_size_mb != block_size_mb:
                    self.block_size_mb = block_size_mb
                break
        
        # 计算总缓存大小
        total_cache_size_mb = block_size_mb * cached_blocks
        
        # 获取当前GPU内存使用情况
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        total_memory = self.memory_constraint
        
        # 输出报告标题
        logger.info(f"===== Cache Status: Layer {layer_id}, Timestep {timestep} =====")
        
        # 基本配置信息
        logger.info(f"Maximum ISP Size: {self.isp_size}")
        logger.info(f"Current ISP Stride: {self.curr_isp_stride}")
        logger.info(f"Current Block Count: {self.curr_block_num}")
        
        # 缓存利用率信息
        logger.info(f"Filled Blocks: {cached_blocks}/{self.curr_block_num} ({cached_blocks/self.curr_block_num*100:.1f}% full)")
        
        # 生成块状态可视化 (O:有数据 X:空)
        block_status = ['O' if out is not None else 'X' for out in self.out_block_cache]
        block_status_str = ' '.join(block_status)
        logger.info(f"Block Status: [{block_status_str}]")
        
        # 内存使用信息
        logger.info(f"Block size: {block_size_mb:.2f} MB")
        logger.info(f"Total cache size: {total_cache_size_mb:.2f} MB")
        logger.info(f"GPU memory: {current_memory:.2f}/{total_memory:.2f} MB ({current_memory/total_memory*100:.1f}%)")
        
        if total_cache_size_mb > 0:
            logger.info(f"Cache percentage: {total_cache_size_mb/current_memory*100:.1f}% of GPU usage")
        
        # 张量形状信息 (直接报告，不需要额外参数)
        if cached_blocks > 0 and example_out is not None:
                
            logger.info("Block Tensor Shapes:")
            logger.info(f"  OUT tensor: {tuple(example_out.shape)}")
            if example_lse is not None:
                logger.info(f"  LSE tensor: {tuple(example_lse.shape)}")
            else:
                logger.info("  LSE tensor: None")
        
        logger.info("================================================")
        

