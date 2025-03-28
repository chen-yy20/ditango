from typing import Dict, Optional, Union, Tuple, List
import torch

from ..logger import init_logger
from ..utils import update_out_and_lse, get_timestep
from .pro_map import get_stride_map

logger = init_logger(__name__)
class proCache:
    
    def __init__(self, isp_size, layer_id):
        self.isp_size = isp_size
        self.curr_isp_stride = isp_size
        self.curr_block_num = isp_size
        self.layer_id = layer_id
       
        self.out_block_cache = [None] * self.curr_block_num
        self.lse_block_cache = [None] * self.curr_block_num
        
        logger.info(f"proCache initialized with ISP size {self.isp_size}")
        
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
        if new_isp_stride == self.isp_size: # 全面刷新
            self.curr_isp_stride = self.isp_size
            self.curr_block_num = self.isp_size
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
        
    def report_cache_status(self, layer_id: int):
        """
        Report the current status of the cache including:
        - Cache configuration
        - Block storage status (filled or empty)
        - Shape information of cached blocks
        """
        # 缓存的基本配置信息
        logger.info(f"===== Cache Status: Layer {layer_id}, Timestep {get_timestep()} =====")
        logger.info(f"Maximum ISP Size: {self.isp_size}")
        logger.info(f"Current ISP Stride: {self.curr_isp_stride}")
        logger.info(f"Current Block Count: {self.curr_block_num}")
        
        # 计算已缓存的块数
        cached_blocks = sum(1 for out in self.out_block_cache if out is not None)
        logger.info(f"Filled Blocks: {cached_blocks}/{self.curr_block_num} ({cached_blocks/self.curr_block_num*100:.1f}% full)")
        
        # 生成块状态可视化 (O:有数据 X:空)
        block_status = ['O' if out is not None else 'X' for out in self.out_block_cache]
        block_status_str = ' '.join(block_status)
        logger.info(f"Block Status: [{block_status_str}]")
        
        # 报告块的形状信息
        if cached_blocks > 0:
            # 找到第一个非空块作为示例
            example_block_id = next((i for i, out in enumerate(self.out_block_cache) if out is not None), None)
            if example_block_id is not None:
                example_out = self.out_block_cache[example_block_id]
                example_lse = self.lse_block_cache[example_block_id]
                
                logger.info("Block Tensor Shapes:")
                logger.info(f"  OUT tensor: {tuple(example_out.shape)}")
                if example_lse is not None:
                    logger.info(f"  LSE tensor: {tuple(example_lse.shape)}")
                else:
                    logger.info("  LSE tensor: None")
        
        logger.info("================================================")
    
    def report_memory_usage(self, layer_id: int):
        """
        Report memory usage information for the current layer:
        - Number of cached blocks
        - Total cache size
        - Current GPU memory usage
        """
        # 计算非空块的数量
        cached_blocks = sum(1 for out in self.out_block_cache if out is not None)
        
        # 计算单个块大小（只计算第一个非空块）
        block_size_mb = 0
        for block_id in range(len(self.out_block_cache)):
            out = self.out_block_cache[block_id]
            lse = self.lse_block_cache[block_id]
            if out is not None:
                block_size_bytes = out.element_size() * out.nelement()
                if lse is not None:
                    block_size_bytes += lse.element_size() * lse.nelement()
                block_size_mb = block_size_bytes / (1024 * 1024)
                break
        
        # 计算总缓存大小
        total_cache_size_mb = block_size_mb * cached_blocks
        
        # 获取当前GPU内存使用情况
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
        
        # 输出简化报告
        logger.info(f"===== Cache Report: Layer {layer_id}, Timestep {get_timestep()} =====")
        logger.info(f"ISP stride: {self.curr_isp_stride}, Block count: {self.curr_block_num}")
        logger.info(f"Cached blocks: {cached_blocks}/{self.curr_block_num}")
        logger.info(f"Block size: {block_size_mb:.2f} MB")
        logger.info(f"Total cache size: {total_cache_size_mb:.2f} MB")
        logger.info(f"GPU memory: {current_memory:.2f}/{total_memory:.2f} MB ({current_memory/total_memory*100:.1f}%)")
        
        if total_cache_size_mb > 0:
            logger.info(f"Cache percentage: {total_cache_size_mb/current_memory*100:.1f}% of GPU usage")
        
        logger.info("================================================")
        

