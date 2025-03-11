import torch
import numpy as np
from typing import List, Optional, Dict, Tuple
from . import get_usp_group

class StrideMap:
    def __init__(self, num_timesteps: int = 50, num_layers: int = 30):
        """
        Initialize StrideMap
        
        Args:
            num_timesteps: Number of diffusion timesteps
            num_layers: Number of transformer layers 
            base_weight: Base weight value to initialize the map
        """
        self.num_timesteps = num_timesteps
        self.num_layers = num_layers
        self.base_weight = get_usp_group().world_size
        self.redundancy_map = self._init_map()

    def _init_map(self) -> torch.Tensor:
        """Initialize stride map with base weight"""
        return torch.ones(self.num_timesteps, self.num_layers, dtype=torch.long) * self.base_weight
    
    def set_pattern_for_row(self, row_idx: int, split_list: List, pattern_list: List):
        """
        Set pattern (a,b,a) for a specific row
        
        Args:
            row_idx: Index of the row to modify
            pattern: Tuple of (left_value, middle_value, left_value)
        """
        if row_idx >= self.num_timesteps:
            raise ValueError(f"Row index {row_idx} exceeds num_timesteps {self.num_timesteps}")
        else:
            if pattern_list is None or len(pattern_list) == 0:
                return
            assert len(split_list) + 1 == len(pattern_list), "Number of patterns does not match number of split chunks. Expected {} patterns for {} splits.".format(len(split_list) + 1, len(pattern_list))
        if split_list is None or len(split_list) == 0:
            assert pattern_list[0] <= self.base_weight, f"isp size {pattern_list[0]} should be less than osp size {self.base_weight}"
            self.redundancy_map[row_idx, :] = pattern_list[0]
            return
        
        # Set first stride
        assert pattern_list[0] <= self.base_weight, f"isp size {pattern_list[0]} should be less than osp size {self.base_weight}"
        self.redundancy_map[row_idx, :split_list[0]] = pattern_list[0]
        # Set middle stride 
        for i in range(len(split_list) - 1):
            assert pattern_list[i+1] <= self.base_weight, f"isp size {pattern_list[i+1]} should be less than osp size {self.base_weight}"
            self.redundancy_map[row_idx, split_list[i]:split_list[i+1]] = pattern_list[i+1]
        # Set right stride
        assert pattern_list[-1] <= self.base_weight, f"isp size {pattern_list[-1]} should be less than osp size {self.base_weight}"
        self.redundancy_map[row_idx, split_list[-1]:] = pattern_list[-1]
        
    def set_pattern_for_rows(self, timestep_indices, split_list: List = None, pattern_list: List = None):
        """
        Set pattern for multiple rows specified by indices
        
        Args:
            timestep_indices: List of row indices or range object
            pattern: Tuple of (left_value, middle_value, left_value)
        """
        if isinstance(timestep_indices, range):
            timestep_indices = list(timestep_indices)
        self.set_pattern_for_row(timestep_indices[0], [], [self.base_weight])
        for idx in timestep_indices[1:]:
            self.set_pattern_for_row(idx, split_list, pattern_list)
        if torch.distributed.get_rank() == 0:
            print_redundancy_map(get_redundancy_map())
    
    def get_map(self) -> torch.Tensor:
        """Return the stride map tensor"""
        return self.redundancy_map
    
    def reset_map(self):
        """Reset stride map to initial state"""
        self.redundancy_map = self._init_map()


def print_redundancy_map(redundancy_map: torch.Tensor, logger=None):
    """
    Print total stride map
    
    Args:
        redundancy_map: stride map tensor
        logger: logger to print info
    """
    # 设置numpy打印选项
    np.set_printoptions(threshold=np.inf, linewidth=1000)
    
    # 转换为numpy数组并格式化
    formatted_output = '\n'.join([' '.join(map(str, row)) for row in redundancy_map.numpy()])
    
    if logger:
        logger.debug(f"redundancy_map:\n{formatted_output}")
    else:
        print(f"redundancy_map:\n{formatted_output}", flush=True)

_redundancy_map: Optional[torch.Tensor] = None
def init_redundancy_map(
    num_timesteps: int,
    num_layers: int,
):
  global _redundancy_map
  assert _redundancy_map is None, ("stride map is already initialized")
  map = StrideMap(num_timesteps, num_layers)
  _redundancy_map = map
  return map
  

def get_redundancy_map() -> torch.Tensor:
  if _redundancy_map is None:
     return None
  return _redundancy_map.get_map()