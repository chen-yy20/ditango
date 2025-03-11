import os
from math import gcd
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union

import torch
import torch.distributed

from ditango.logger import init_logger
from . import GroupCoordinator

logger = init_logger(__name__)

_WORLD: Optional[GroupCoordinator] = None
_DP: Optional[GroupCoordinator] = None
_CFG: Optional[GroupCoordinator] = None
_USP: Optional[GroupCoordinator] = None
_IOSP_GROUP_DICT: Optional[Dict[int, GroupCoordinator]] = None


def get_world_group() -> GroupCoordinator:
  assert _WORLD is not None, ("world group is not initialized")
  return _WORLD


def get_dp_group() -> GroupCoordinator:
  assert _DP is not None, ("data parallel group is not initialized")
  return _DP


def get_cfg_group() -> GroupCoordinator:
  assert _CFG is not None, ("classifier free guidance parallel group is not initialized")
  return _CFG


def get_usp_group() -> GroupCoordinator:
  assert _USP is not None, ("Unified sequence parallel group is not initialized")
  return _USP

def get_iosp_group(stride: Optional[int] = None) -> Tuple[GroupCoordinator, GroupCoordinator]:
    """
    Get the ISP and OSP parallel groups for specified stride
    
    Args:
        stride: Sparsity parameter n
        
    Returns:
        tuple: (osp_group, isp_group) - The outer and inner sequence parallel groups
        
    Raises:
        AssertionError: If IOSP groups are not initialized or specified stride doesn't exist
    """
    assert _IOSP_GROUP_DICT is not None, "IOSP groups are not initialized"
    
    if stride is None:
        assert len(_IOSP_GROUP_DICT) == 1, "Found more than 1 iosp group pair, need to provide stride"
        groups = next(iter(_IOSP_GROUP_DICT.values()))
    else:
        assert stride in _IOSP_GROUP_DICT, f"No IOSP groups found for stride={stride}"
        groups = _IOSP_GROUP_DICT[stride]
    return groups['osp'], groups['isp']
  
def get_isp_group(stride: Optional[int] = None) -> GroupCoordinator:
    _, isp = get_iosp_group(stride)
    return isp

def get_osp_group(stride: Optional[int] = None) -> GroupCoordinator:
    osp, _ = get_iosp_group(stride)
    return osp


def init_world_group(ranks: List[int], local_rank: int, backend: str) -> GroupCoordinator:
  return GroupCoordinator(
    group_ranks=[ranks],
    local_rank=local_rank,
    torch_distributed_backend=backend,
    group_name="world",
  )


def init_model_parallel_group(
  group_ranks: List[List[int]],
  local_rank: int,
  backend: str,
  group_name: Optional[str] = None,
) -> GroupCoordinator:
  if group_name == 'cfg':
    for group in group_ranks:
      assert len(group) <= 2, f'cfg_size can only be 1 or 2'
  return GroupCoordinator(
    group_ranks=group_ranks,
    local_rank=local_rank,
    torch_distributed_backend=backend,
    group_name=group_name,
  )


def init_distributed_environment(
  world_size: int = -1,
  rank: int = -1,
  distributed_init_method: str = "env://",
  local_rank: int = -1,
  backend: str = "nccl",
):
  logger.debug("world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s", world_size, rank, local_rank, distributed_init_method, backend)
  if not torch.distributed.is_initialized():
    assert distributed_init_method is not None, ("distributed_init_method must be provided when initializing "
                                                 "distributed environment")
    # this backend is used for WORLD
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_method, world_size=world_size, rank=rank)
  # set the local rank
  # local_rank is not available in torch ProcessGroup,
  # see https://github.com/pytorch/pytorch/issues/122816
  if local_rank == -1:
    # local rank not set, this usually happens in single-node
    # setting, where we can use rank as local rank
    if distributed_init_method == "env://":
      local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
      local_rank = rank
  global _WORLD
  if _WORLD is None:
    ranks = list(range(torch.distributed.get_world_size()))
    _WORLD = init_world_group(ranks, local_rank, backend)
  else:
    assert _WORLD.world_size == torch.distributed.get_world_size(), ("world group already initialized with a different world size")


def generate_parallel_groups(world_size: int, do_cfg_guidance: bool = True, sparse_type: Union[List, str, None] = 'full'):
    """
    Generate group ranks for model parallel initialization based on world size.
    When do_cfg_guidance=True:
        - USP groups split into [0:half] and [half:world_size]
        - CFG groups pair ranks from first and second half
    When do_cfg_guidance=False:
        - All ranks in single USP group
        - No CFG groups
    
    Args:
        world_size (int): Total number of processes
        do_cfg_guidance (bool): If True, prioritize CFG groups formation
        gpus_per_node (int): Number of GPUs per node
        
    Returns:
        tuple: (cfg_group_ranks, usp_group_ranks, iosp_dict)
    """
    if world_size < 1:
        raise ValueError("World size must be positive")
      
    if isinstance(sparse_type, str):
        assert sparse_type in ['full', 'sequential'], f"Sparse type can only be 'full' or 'sequential'! Found {sparse_type}."
    
    if isinstance(sparse_type, List):
        stride_list = sparse_type
    else:
        stride_list = [1]
    
      
    iosp_dict = {}
    if world_size == 1:
        for stride in stride_list:
            iosp_dict[stride] = (None, None)
        return None, None, iosp_dict

    if do_cfg_guidance:
        half_size = world_size // 2
        
        # USP groups - split into first and second half
        usp_group_ranks = [
            list(range(0, half_size)),           # First half
            list(range(half_size, world_size))   # Second half
        ]
        
        # CFG groups - pair ranks from first and second half
        cfg_group_ranks = []
        for i in range(half_size):
            cfg_group_ranks.append([i, i + half_size])
            
    else:
        # No CFG groups
        cfg_group_ranks = None
        # All ranks in single USP group
        usp_group_ranks = [list(range(world_size))]

    usp_size = len(usp_group_ranks[0])
    if sparse_type == 'sequential':
        stride_list = [1]
    
    # Set iosp group
    for stride in stride_list:
        assert stride % usp_size == 0 or usp_size % stride == 0, f"Does not support {stride=} but {usp_size=}"
        isp_size = gcd(usp_size, stride)
        osp_size = usp_size // isp_size
        osp_groups = []
        isp_groups = []
        for rank_list in usp_group_ranks:
            for i in range(0, usp_size, osp_size):
                osp_groups.append(rank_list[i:i+osp_size])
            for i in range(osp_size):
                isp_groups.append(rank_list[i::osp_size])
            iosp_dict[stride] = (osp_groups, isp_groups)
            
    return cfg_group_ranks, usp_group_ranks, iosp_dict

def init_model_parallel(
  dp_group_ranks: Optional[List[List[int]]] = None,
  cfg_group_ranks: Optional[List[List[int]]] = None,
  usp_group_ranks: Optional[List[List[int]]] = None,
  iosp_dict: Optional[Dict] = None,
  backend: Optional[str] = None,
):
  assert torch.distributed.is_initialized()
  world_size = torch.distributed.get_world_size()
  backend = backend or torch.distributed.get_backend(get_world_group().device_group)

  if dp_group_ranks is None:
    dp_group_ranks = [[rank] for rank in range(world_size)]
  if cfg_group_ranks is None:
    cfg_group_ranks = [[rank] for rank in range(world_size)]
  if usp_group_ranks is None:
    usp_group_ranks = [[rank] for rank in range(world_size)]

  global _DP
  assert _DP is None, ("data parallel group is already initialized")
  _DP = init_model_parallel_group(dp_group_ranks, get_world_group().local_rank, backend, group_name="dp")

  global _CFG
  assert _CFG is None, ("classifier free guidance parallel group is already initialized")
  _CFG = init_model_parallel_group(cfg_group_ranks, get_world_group().local_rank, backend, group_name="cfg")

  global _USP
  assert _USP is None, ("unified sequence parallel group is already initialized")
  _USP = init_model_parallel_group(usp_group_ranks, get_world_group().local_rank, backend, group_name="usp")

  global _IOSP_GROUP_DICT
  if iosp_dict is not None:
    assert _IOSP_GROUP_DICT is None, ("iosp sequence parallel group is already initialized")
    _IOSP_GROUP_DICT = {}
    for n, (osp_ranks, isp_ranks) in iosp_dict.items():
      if osp_ranks is None:
        osp_ranks = [[rank] for rank in range(world_size)]
      if isp_ranks is None:
        isp_ranks = usp_group_ranks
      _IOSP_GROUP_DICT[n] = {
        'osp': init_model_parallel_group(
          osp_ranks,
          get_world_group().local_rank,
          backend,
          group_name=f"osp_{n}"
        ),
        'isp': init_model_parallel_group(
          isp_ranks,
          get_world_group().local_rank,
          backend,
          group_name=f"isp_{n}"
        )
      }
  if torch.distributed.get_rank() == 0:
      # logger.debug(f"{dp_group_ranks=}")
      logger.debug(f"{cfg_group_ranks=}")
      logger.debug(f"{usp_group_ranks=}")
      for stride, (osp_ranks, isp_ranks) in iosp_dict.items():
          logger.debug(f"stride={stride}, osp_ranks={osp_ranks}, isp_ranks={isp_ranks}")
      

def destroy_model_parallel():
  """Set the groups to none and destroy them."""
  global _DP
  if _DP:
    _DP.destroy()
  _DP = None

  global _CFG
  if _CFG:
    _CFG.destroy()
  _CFG = None

  global _USP
  if _USP:
    _USP.destroy()
  _USP = None


def destroy_distributed_environment():
  global _WORLD
  if _WORLD:
    _WORLD.destroy()
  _WORLD = None
  if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
