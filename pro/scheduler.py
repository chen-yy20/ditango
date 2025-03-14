import torch
import torch.distributed as dist
from typing import Optional, Tuple, List

from ..core.group_coordinate import GroupCoordinator
from ..core.redundancy_map import get_redundancy_map
from .cache import proCache
from ..utils import update_out_and_lse

from ..logger import init_logger

logger = init_logger(__name__)


class chunk_scheduler():
    def __init__(self, layer_id: int, isp_group: GroupCoordinator) -> None:
        self.group = isp_group
        self.isp_size = isp_group.world_size
        self.isp_rank = isp_group.rank_in_group
        self.global_rank = dist.get_rank()
        self.layer_id = layer_id
        self.chunk_map = [0] * self.isp_size
        self.cache = proCache()
        self.timestep = 0
        
    def update_timestep(self, timestep: int) -> None:
        self.timestep = timestep
        
    def _reset_chunk_map(self) -> None:
        logger.info(f"Step {self.timestep} - Reset chunk map")
        self.chunk_map = [0] * self.isp_size
        
    def _get_curr_redundancy(self) -> int:
        return int(get_redundancy_map()[self.timestep][self.layer_id])

    def get_cached_output(self) -> Tuple:
        # Decide which chunk to get
        max_redundancy = self._get_curr_redundancy()
        
        chunk_id_list = self.cache.provide_chunk_id_list(max_redundancy)
        merged_cached_out, merged_cached_lse = self._fetch(chunk_id_list)
        
        # update chunk_map
        for chunk_id in chunk_id_list:
            int_ids = [int(num) for num in chunk_id.split('&')]
            for int_id in int_ids:
                assert self.chunk_map[int_id] == 0, "Something went wrong with scheduler.get_cached_output()..."
                self.chunk_map[int_id] = 1
                
        return merged_cached_out, merged_cached_lse
    
   
    def merge(out1: Optional[torch.Tensor],
        lse1: Optional[torch.Tensor],
        out2: torch.Tensor,
        lse2: torch.Tensor) -> Tuple:
        '''
        merge two intermediate blocks into one, including out and lse
        '''
        if out1 is None:
            return out2, lse2
        else:
            # merge 2 to 1
            lse2 = lse2.squeeze(-1).transpose(-1, -2) # reshape for merge
            logger.info(f"Merging blocks | {out1.shape=} {out2.shape=} {lse1.shape=} {lse2.shape=}")
            out, lse = update_out_and_lse(out1, lse1, out2, lse2)
        
        return out, lse
    
    def get_exchange_rank_list(self):
        '''
        Get rank list of exchange tensor chunks with p2p send_recv between GPUs
        '''
        pass
    
    def _fetch(self, chunk_id_list: List) -> Tuple:
        '''
        Fetch tensor chunks (out and lse) from local cache, merge them for attention calculate
        '''
        out, lse = None, None
        for cid in chunk_id_list:
            cached_out, cached_lse = self.cache.get_chunk(cid)
            out, lse = self.merge((out, lse), (cached_out, cached_lse))
            
        return out, lse
    
    def store():
        '''
        Store tensor chunks (out and lse) to local cache
        '''
        
        
