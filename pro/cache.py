from typing import Dict, Optional, Union, Tuple, List
from math import gcd
import torch
import os
import warnings
import numpy as np
import torch.distributed as dist
from itertools import combinations

from ..core.redundancy_map import get_redundancy_map
from ..core.arguments import get_config
from ..logger import init_logger

logger = init_logger(__name__)
        
class proCache:
    def __init__(self):
       self.cache_size = 4 # TODO: Autoset
       self.cached_chunk_ids = []
       self.decay_list = [] # 维护腐烂度
       self.value_list = [] # 维护价值
       self.out_cache = []
       self.lse_cache = []
       
    def _chunk_num(self, chunk_id: str):
        return len(chunk_id.split('&'))
    
    def _cid2id(self, cid: str) -> int:
        if cid not in self.cached_chunk_ids:
            logger.error(f"{self.cached_chunk_ids=}")
            assert False, f"Something went wrong when trying to get {cid} from cache"
        return self.cached_chunk_ids.index(cid)
    
       
    def provide_chunk_id_list(self, max_redundancy: int):
        def total_value_and_decay(chunk_ids: List):
            value = 0
            decay = 0
            for cid in chunk_ids:
                 value += self.value_list[self._cid2id(cid)]
                 decay += self.decay_list[self._cid2id(cid)]
            return value, decay
        
        # 遍历 cached_chunk_ids 的可能组合
        good_value = 0
        good_combo = []
        for r in range(len(self.cached_chunk_ids), -1, -1):
            for combo in combinations(self.cached_chunk_ids, r):
                # 计算组合的最大价值
                value, decay = total_value_and_decay(combo)
                if value > good_value and decay <= max_redundancy:
                    good_value = value
                    good_combo = combo
        return good_combo    
    
    def get_chunk(self, chunk_id: str) -> Tuple:
        
        cached_out = self.out_cache[self._cid2id(chunk_id)]
        cached_lse = self.lse_cache[self._cid2id(chunk_id)]
        
        return cached_out, cached_lse
        
    
    def update_decay_and_value(self):
        for i, chunk_id in enumerate(self.cached_chunk_ids):
            self.decay_list[i] += self._chunk_num(chunk_id)
            self.value_list[i] -= self._chunk_num(chunk_id) # TODO: 添加调整参数
    
    
    
    
        
        
