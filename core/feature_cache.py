from typing import Dict, Optional, Union, Tuple, List
from math import gcd
import torch
import os
import warnings
import numpy as np
import torch.distributed as dist

from .redundancy_map import get_redundancy_map
from ..baseline.cache import easyCache, DistriFusionKVCache
from ..logger import init_logger

logger = init_logger(__name__)
        
class oCache:
    
    def __init__(self, num_timesteps, num_layers):
       self.timestep = 0
       self.curr_isp_size = -1
       self.num_layers = num_layers
       self.num_timesteps = num_timesteps
       
       self.out_cache = [None] * num_layers
       self.lse_cache = [None] * num_layers
       
       self.next_calc_block_id=0
    
    def set_isp_size(self, isp_size:int):
        self.curr_isp_size = isp_size       
    
    def get_curr_isp_stride(self, layer_id: int):
        isp_stride = int(get_redundancy_map()[self.timestep, layer_id].item())
        return isp_stride if isp_stride < self.curr_isp_size else self.curr_isp_size
    
    def get_next_isp_stride(self, layer_id: int):
        if self.timestep + 1 >= self.num_timesteps:
            return self.curr_isp_size
        isp_stride = int(get_redundancy_map()[self.timestep + 1, layer_id].item())
        return isp_stride if isp_stride < self.curr_isp_size else self.curr_isp_size
   
    def get_block(self, layer_id, block_id):
        # if dist.get_rank()==0 and layer_id == 15:
        #     logger.info(f"{self.timestep}-{layer_id} | Trying to get {block_id=}")
        if block_id >= len(self.out_cache[layer_id]):
            logger.error(f"{self.timestep}-{layer_id} | Cache is not ready but trying to get cached value.")
        cached_out = self.out_cache[layer_id][block_id] 
        cached_lse = self.lse_cache[layer_id][block_id]
        # if cached_out is None:
        #     if dist.get_rank()==0 and layer_id == 15:
        #         logger.error(f"{self.timestep}-{layer_id} | Get None type cached value.")
        return cached_out, cached_lse
    
    def purge_cache(self, layer_id, target_block_id, block_num):
        if self.get_next_isp_stride(layer_id) == self.curr_isp_size: # 下一轮算满，不用存了
            self.out_cache[layer_id] = []
            self.lse_cache[layer_id] = []
        else: 
            if len(self.out_cache[layer_id]) != block_num:
                self.out_cache[layer_id] = [None] * block_num
                self.lse_cache[layer_id] = [None] * block_num
                if self.get_curr_isp_stride(layer_id) != self.curr_isp_size:
                    logger.error(f"{self.timestep}-{layer_id} | Cache is not ready but trying to get cached value.")
            else:
                self.next_calc_block_id = (target_block_id + 1) % block_num
                self.out_cache[layer_id][self.next_calc_block_id] = None
                self.lse_cache[layer_id][self.next_calc_block_id] = None
                # if dist.get_rank()==0 and layer_id == 15:
                #     logger.info(f"{self.timestep}-{layer_id} | Released block {next_calc_block_id}/{block_num}")
    
    def store_block(self, layer_id, block_id, out, lse):
        if self.get_next_isp_stride(layer_id) == self.curr_isp_size: # 下一轮算满，不用存了
            return 
        elif block_id == self.next_calc_block_id: # 下一轮要算，不用存了
            return
        else:
            # if block_id >= len(self.out_cache[layer_id]):
            #     logger.error(f"{self.timestep}-{layer_id} | Trying to store block{block_id} but cache is not ready.")
            
            self.out_cache[layer_id][block_id] = out
            self.lse_cache[layer_id][block_id] = lse
            # if dist.get_rank()==0:
            #     logger.info(f"{self.timestep}-{layer_id} | Stored {block_id} in cache. Mem={torch.cuda.memory_allocated()}")
    
    def clear(self):
        logger.warning("============== Clear oCache =================")
    
    def update_timestep(self, timestep: int):
        """Update current timestep"""
        self.timestep = timestep
    
    def report_memory_usage(self):
        """
        Report detailed memory usage information including:
        - Total number of cached blocks
        - Size of each cached block
        - Current GPU memory usage
        """
        total_blocks = 0
        total_cache_size_bytes = 0
        block_sizes = {}
        
        # 收集每层缓存的块数量和大小
        for layer_id in range(self.num_layers):
            if self.out_cache[layer_id] is None:
                continue
                
            layer_blocks = 0
            layer_size_bytes = 0
            
            for block_id in range(len(self.out_cache[layer_id])):
                out = self.out_cache[layer_id][block_id]
                lse = self.lse_cache[layer_id][block_id]
                
                if out is not None:
                    layer_blocks += 1
                    # 计算块大小(out + lse)
                    block_size_bytes = out.element_size() * out.nelement()
                    if lse is not None:
                        block_size_bytes += lse.element_size() * lse.nelement()
                    
                    layer_size_bytes += block_size_bytes
                    
                    # 记录块大小信息
                    if f"Layer {layer_id}" not in block_sizes:
                        block_sizes[f"Layer {layer_id}"] = []
                    block_sizes[f"Layer {layer_id}"].append(f"Block {block_id}: {block_size_bytes / (1024 * 1024):.2f} MB")
            
            total_blocks += layer_blocks
            total_cache_size_bytes += layer_size_bytes
        
        # 获取当前GPU内存使用情况
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
        
        # 缓存大小
        total_cache_size_mb = total_cache_size_bytes / (1024 * 1024)
        
        # 输出报告
        logger.info(f"===== Memory Usage Report at timestep {self.timestep} =====")
        logger.info(f"Total cached blocks: {total_blocks}")
        logger.info(f"Total cache size: {total_cache_size_mb:.2f} MB ({total_cache_size_bytes} bytes)")
        logger.info(f"Cache breakdown by layer:")
        
        for layer, blocks in block_sizes.items():
            layer_total = sum([float(b.split(": ")[1].split(" MB")[0]) for b in blocks])
            logger.info(f"  {layer}: {len(blocks)} blocks, {layer_total:.2f} MB")
            # 如果需要详细到每个块的大小，可以取消下面的注释
            # for block_info in blocks:
            #     logger.info(f"    {block_info}")
        
        logger.info(f"Current GPU memory: {current_memory:.2f} MB / {total_memory:.2f} MB ({current_memory/total_memory*100:.1f}%)")
        logger.info(f"Peak GPU memory: {max_memory:.2f} MB / {total_memory:.2f} MB ({max_memory/total_memory*100:.1f}%)")
        logger.info(f"Cache memory percentage: {total_cache_size_mb/current_memory*100:.1f}% of current usage")
        logger.info("================================================")
        
# Global variable and thread lock
_GLOBAL_KV_CACHE: Optional[Union[oCache, easyCache, DistriFusionKVCache]] = None

def init_cache(args) -> Union[oCache, easyCache, DistriFusionKVCache]:
    """Initialize or reinitialize the global KV cache"""
    global _GLOBAL_KV_CACHE
    assert get_redundancy_map() is not None, "Stride map must be initialized before init_cache()"
    if args.use_easy_cache:
        logger.warning(f"====== You are using baseline easyCache instead of DiTango Cache, threshold={args.cache_threshold}======")
        assert args.cache_threshold is not None, "You should set '--cache-threshold' for easyCache baseline "
        _GLOBAL_KV_CACHE = easyCache(args.num_inference_steps, args.num_layers, args.cache_threshold)
    elif args.use_distrifusion:
        logger.warning(f"====== You are using baseline DistriFusion_KVCache instead of DiTango Cache======")
        _GLOBAL_KV_CACHE = DistriFusionKVCache()
    else:
        if args.rank == 0:
            logger.info(f"====== Init DiTango Cache======")
        _GLOBAL_KV_CACHE = oCache(args.num_inference_steps, args.num_layers)
    if args.rank == 0:
        logger.info("====== Finished Init Cache======")
    return _GLOBAL_KV_CACHE

def get_cache() -> Optional[Union[oCache, easyCache, DistriFusionKVCache]]:
    """Get the global KV cache instance"""
    if _GLOBAL_KV_CACHE is None:
        raise RuntimeError("KV cache not initialized. Call init_cache() first.")
    return _GLOBAL_KV_CACHE

def exist_oCache():
    return isinstance(_GLOBAL_KV_CACHE, oCache)

def exist_cache():
    """Check if the global KV cache exists"""
    return _GLOBAL_KV_CACHE is not None

def clear_cache():
    """Clear the global KV cache"""
    if _GLOBAL_KV_CACHE is not None:
        _GLOBAL_KV_CACHE.clear()

def reset_cache():
    """Reset (remove) the global KV cache"""
    global _GLOBAL_KV_CACHE
    _GLOBAL_KV_CACHE = None