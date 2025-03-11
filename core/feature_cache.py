from typing import Dict, Optional, Union, Tuple, List
from math import gcd
import torch
import os
import warnings
import numpy as np
import torch.distributed as dist

from . import get_redundancy_map, get_usp_group, get_isp_group
from ..logger import init_logger

logger = init_logger("__main__")
do_analyse = True if int(os.getenv("ANALYSIS"))==1 else False
        
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
            if dist.get_rank()==0:
                logger.info(f"{self.timestep}-{layer_id} | Stored {block_id} in cache. Mem={torch.cuda.memory_allocated()}")
    
    def clear(self):
        logger.warning("============== Clear oCache =================")
    
    def update_timestep(self, timestep: int):
        """Update current timestep"""
        self.timestep = timestep
    
    
class easyCache:
    
    def __init__(self, num_timesteps, num_layers, threshold):
       self.timestep = 0
       self.num_layers = num_layers
       self.num_timesteps = num_timesteps
       self.threshold = threshold
       self.cache = {}
       
   
    def is_important(self, layer_id: int):
        importance = int(get_redundancy_map()[self.timestep, layer_id].item())
        return importance >= self.threshold
   
    def get_feature(self, layer_id, name):
        if name not in self.cache.keys():
            logger.debug(f"Didn't find tensor {name} in cache")
        if dist.get_rank()==0:
            logger.info(f"{self.timestep}-{layer_id} | Trying to get feature {name}")
        cached_feature = self.cache[name][layer_id]
        if cached_feature is None:
            if dist.get_rank():
                logger.error(f"{self.timestep}-{layer_id} | Get None type cached value.")
        return cached_feature
    
    def store_feature(self, layer_id, name, feature):
        if name not in self.cache.keys():
            self.cache[name] = [None] * self.num_layers
        self.cache[name][layer_id] = feature
        if dist.get_rank()==0:
            logger.info(f"{self.timestep}-{layer_id} | Stored tensor {name} in cache. Mem={torch.cuda.memory_allocated()}")
    
    def clear(self):
        logger.warning("============== Clear oCache =================")
        self.cache.clear()
    
    def update_timestep(self, timestep: int):
        """Update current timestep"""
        self.timestep = timestep
        
        
# Global variable and thread lock
_GLOBAL_KV_CACHE: Optional[Union[oCache, easyCache]] = None

def init_cache(num_timesteps, num_layers, threshold = None) -> Union[oCache, easyCache]:
    """Initialize or reinitialize the global KV cache"""
    global _GLOBAL_KV_CACHE
    assert get_redundancy_map() is not None, "Stride map must be initialized before init_cache()"
    if threshold is not None:
        _GLOBAL_KV_CACHE = easyCache(num_timesteps, num_layers, threshold)
    else:
        _GLOBAL_KV_CACHE = oCache(num_timesteps, num_layers)
    return _GLOBAL_KV_CACHE

def get_cache() -> Optional[Union[oCache, easyCache]]:
    """Get the global KV cache instance"""
    if _GLOBAL_KV_CACHE is None:
        raise RuntimeError("KV cache not initialized. Call init_cache() first.")
    return _GLOBAL_KV_CACHE

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