from typing import Dict, Optional, Union, Tuple, List
import torch
import torch.distributed as dist

from ..core.group_coordinate import  GroupCoordinator
from ..core.redundancy_map import get_redundancy_map
from ..core.parallel_state import get_usp_group
from ..logger import init_logger

logger = init_logger(__name__)

class DistriFusionKVCache:
    def __init__(self):
        """
        Initialize the KV caching system for DistriFusion models
        
        Args:
            stride_map: Binary tensor indicating whether to cache at each timestep and layer
        """
        self.timestep = 0
        self.stride_map = get_redundancy_map()
        self.num_timesteps, self.num_layers = get_redundancy_map().shape
        self.full_sp_size = get_usp_group().world_size
        
        # Cache structure: {layer_id: {"k": Tensor, "v": Tensor}}
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
    def should_cache(self, layer_id: int) -> bool:
        """Check if current timestep and layer should be cached"""
        if self.timestep + 1 >= self.num_timesteps:
            return False
        return self.stride_map[self.timestep + 1, layer_id].item() < self.full_sp_size # next step need cache
    
    def is_cached(self, layer_id: int) -> bool:
        """Check if features for given layer are cached"""
        return layer_id in self.cache
    
    def store(self, layer_id: int, key: torch.Tensor, value: torch.Tensor):
        """
        Store K,V features
        
        Args:
            layer_id: Layer identifier
            key: Key tensor to cache
            value: Value tensor to cache
        """
        if not self.should_cache(layer_id):
            return
        # Store detached copies of key and value
        self.cache[layer_id] = {
            "k": key.detach().clone(),
            "v": value.detach().clone()
        }
    
    def get_kv(self, layer_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve cached K,V for given layer
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            Dict with "k" and "v" tensors if cached, None otherwise
        """
        if not self.is_cached(layer_id):
            return None
        
        return {
            "k": self.cache[layer_id]["k"].clone(),
            "v": self.cache[layer_id]["v"].clone()
        }
    
    def async_gather(self, layer_id: int, dim: int, group: GroupCoordinator) -> None:
        """
        Asynchronously gather cached KV across distributed processes
        
        Args:
            layer_id: Layer identifier
            dim: Dimension to gather on
            group: Process group for distributed communication
        """
        if not self.is_cached(layer_id):
            # logger.warning(f"No cached KV found for layer={layer_id}")
            return 

        if group is None or group.world_size == 1:
            # logger.info(f"t{self.timestep} l{layer_id} | Single process, no need to gather")
            return 
        # logger.debug("Tried to gather kv")
        # Gather K and V separately
        k_gathered = group.all_gather(self.cache[layer_id]["k"], dim=dim)
        v_gathered = group.all_gather(self.cache[layer_id]["v"], dim=dim)
        
        self.cache[layer_id]["k"] = k_gathered
        self.cache[layer_id]["v"] = v_gathered
        
        # logger.info(f"l{layer_id} t{self.timestep} | Gathered KV, shapes: k={k_gathered.shape}, v={v_gathered.shape}, mem={torch.cuda.max_memory_allocated()}")

    def get_cache_status(self) -> Dict:
        """Get current cache status"""
        status = {}
        for layer_id, kv_dict in self.cache.items():
            status[f'layer_{layer_id}'] = {
                'k_shape': tuple(kv_dict["k"].shape),
                'v_shape': tuple(kv_dict["v"].shape)
            }
        return status
    
    def print_status(self):
        """Print current cache status"""
        status = self.get_cache_status()
        logger.info(f"Timestep {self.timestep} | Cache: {status}")
    
    def clear(self):
        """Clear all cached features"""
        self.cache.clear()
    
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
        # logger.info(f"{self.timestep}-{layer_id} | {importance=} {self.threshold=}")
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
        logger.warning("============== Clear easyCache =================")
        self.cache.clear()
    
    def update_timestep(self, timestep: int):
        """Update current timestep"""
        self.timestep = timestep