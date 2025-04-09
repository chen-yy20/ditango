from typing import Dict, Optional, Union, Tuple, List
import torch
import torch.distributed as dist

from ..core.group_coordinate import  GroupCoordinator
from ..core.stride_map import get_stride_map
from ..core.parallel_state import get_usp_group
from ..core.config import get_config
from ..logger import init_logger
from ..utils import get_timestep

logger = init_logger(__name__)
Fusion_Cache = None
Easy_Cache = None

class DistriFusionKVCache:
    def __init__(self):
        """
        Initialize the KV caching system for DistriFusion models
        
        Args:
            stride_map: Binary tensor indicating whether to cache at each timestep and layer
        """
        self.num_timesteps, self.num_layers = get_config().num_inference_steps, get_config().num_layers
        self.full_sp_size = get_usp_group().world_size
        self.warmup_steps = 3
        
        # Cache structure: {layer_id: {"k": Tensor, "v": Tensor}}
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
    def warmup(self, layer_id: int) -> bool:
        """Check if current timestep and layer should be cached"""
        timestep = get_timestep()
        if timestep < self.warmup_steps:
            return True
        else:
            return False
    
    def should_cache(self) -> bool:
        if get_timestep() < self.warmup_steps -1 :
            return False
        else:
            return True
        
        # if get_timestep() + 1 >= self.num_timesteps:
        #     return False
        # return get_stride_map().get_next_isp_stride(get_timestep(), layer_id) < self.full_sp_size # next step need cache
    
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
        if not self.should_cache():
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
            # logger.info(f"t{get_timestep()} l{layer_id} | Single process, no need to gather")
            return 
        # logger.debug("Tried to gather kv")
        # Gather K and V separately
        k_gathered = group.all_gather(self.cache[layer_id]["k"], dim=dim)
        v_gathered = group.all_gather(self.cache[layer_id]["v"], dim=dim)
        
        self.cache[layer_id]["k"] = k_gathered
        self.cache[layer_id]["v"] = v_gathered
        
        # logger.info(f"l{layer_id} t{get_timestep()} | Gathered KV, shapes: k={k_gathered.shape}, v={v_gathered.shape}, mem={torch.cuda.max_memory_allocated()}")

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
        logger.info(f"Timestep {get_timestep()} | Cache: {status}")
    
    def clear(self):
        """Clear all cached features"""
        self.cache.clear()
    

class easyCache:
    def __init__(self):
       self.num_layers = get_config().num_layers
       self.num_timesteps = get_config().num_inference_steps
       self.skip = 3
       self.cache = {}
       logger.info(f"Using easyCache, skip={self.skip}")
       
   
    def is_important(self):
        # logger.info(f"{get_timestep()}-{layer_id} | {importance=} {self.threshold=}")
        timestep = get_timestep()
        if timestep < 3:
            return True
        return get_timestep() % self.skip == 0
    
    def should_cache(self, layer_id: int):
        if get_timestep() == self.total_steps -1:
            return False
        else:
            next_step_redundancy = int(get_stride_map()[get_timestep() + 1, layer_id].item())
            return next_step_redundancy <= self.threshold # will skip
   
    def get_feature(self, layer_id, name):
        if name not in self.cache.keys():
            logger.debug(f"Didn't find tensor {name} in cache")
        # if dist.get_rank()==0:
        #     logger.info(f"{get_timestep()}-{layer_id} | Trying to get feature {name}")
        cached_feature = self.cache[name][layer_id]
        if cached_feature is None:
            if dist.get_rank():
                logger.error(f"{get_timestep()}-{layer_id} | Get None type cached value.")
        return cached_feature
    
    def store_feature(self, layer_id, name, feature):
        if name not in self.cache.keys():
            self.cache[name] = [None] * self.num_layers
        self.cache[name][layer_id] = feature
        # if dist.get_rank()==0:
        #     logger.info(f"{get_timestep()}-{layer_id} | Stored tensor {name} in cache. Mem={torch.cuda.memory_allocated()}")
    
    def clear(self):
        logger.warning("============== Clear easyCache =================")
        self.cache.clear()
    
        

def init_fusion_cache():
    """
    Initialize the global DistriFusionKVCache instance
    """
    global Fusion_Cache
    if Fusion_Cache is None:
        Fusion_Cache = DistriFusionKVCache()
        logger.info("Initialized DistriFusionKVCache")
    else:
        logger.warning("DistriFusionKVCache already initialized")

def get_fusion_cache() -> DistriFusionKVCache:
    """
    Get the global DistriFusionKVCache instance
    """
    global Fusion_Cache
    if Fusion_Cache is None:
        logger.error("DistriFusionKVCache not initialized. Please call init_fusion_cache() first.")
        return None
    return Fusion_Cache


def init_easy_cache():
    config = get_config()
    global Easy_Cache
    if Easy_Cache is None:
        Easy_Cache = easyCache()
        logger.info("Initialized easyCache")
    else:
        logger.warning("easyCache already initialized")
        
def get_easy_cache() -> easyCache:
    """
    Get the global easyCache
    """
    global Easy_Cache
    if Easy_Cache is None:
        logger.error("easyCache not initialized. Please call init_easy_cache() first.")
        return None
    return Easy_Cache

def clear_cache():
    if get_easy_cache() is not None:
        get_easy_cache().clear()
        logger.info("Cleared Easy Cache")
    if get_fusion_cache() is not None:
        get_fusion_cache().clear()
        logger.info("Cleared Fusion Cache")


