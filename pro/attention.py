import torch

from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from ..core.group_coordinate import GroupCoordinator

from .scheduler import chunk_scheduler

from ..logger import init_logger


logger = init_logger(__name__)


class proAttention:
    def __init__(self, layer_id: int, isp_group: GroupCoordinator) -> None:
        self.scheduler = chunk_scheduler(layer_id, isp_group)
        
    def attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        '''
            Attention with flash attn func (static sequence len), input size should be (b, s, a, d)
        '''
        out, lse = None, None
        # 1. get chunks from cache
        out, lse = self.scheduler.get_cached_output()
        
        # 2. exchange and calculate other chunks
        send_rank_list, recv_rank_list = self.scheduler.get_exchange_rank_list()
        
        # 3. store new chunks

        return out
        
        
