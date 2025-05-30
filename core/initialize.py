import torch
from typing import List
from argparse import ArgumentParser
# from .redundancy_map import init_redundancy_map, print_redundancy_map
from .config import init_config, print_config
from .parallel_state import init_distributed_environment, init_model_parallel, get_world_group, generate_parallel_groups
from .redundancy_map import init_redundancy_map

from ditango.baseline.cache import init_fusion_cache, init_easy_cache

from ..timer import init_timer
from ..logger import init_logger

logger = init_logger(__name__)
USE_DITANGO = False

def init_ditango(
    config_path: str = None,
):
    # 1. init arguments
    config = init_config(config_path)
    
    # 2. init parallel
    init_distributed_environment(world_size=config.world_size, rank=config.rank, local_rank=config.local_rank)
    print_config()
    local_rank = get_world_group().local_rank
    torch.cuda.set_device(local_rank)
    sparse_type = 'full'
    # stride_list = sorted(set(pattern_list))
    # if config.use_distrifusion:
    #     assert config.model_name not in ['latte', 'opensora'], f"Unsupported model type {config.model_name} for DistriFusion Baseline."
    #     sparse_type = stride_list
    # elif config.model_name == 'opensora':
    #     sparse_type = stride_list # sparse_n=1
    #     for stride in stride_list: # sparse_n=4
    #         if stride > 4:
    #             assert stride % 4 == 0
    #             new_stride = stride // 4
    #             if new_stride not in sparse_type:
    #                 sparse_type.append(new_stride)
    # elif config.model_name == 'latte':
    #     sparse_type = 'sequential'
    # else:
    #     sparse_type = 'full' 
        
    cfg_ranks, usp_ranks, iosp_dict = generate_parallel_groups(config.world_size, do_cfg_guidance=config.do_cfg_parallel, sparse_type=sparse_type)
    init_model_parallel(cfg_group_ranks=cfg_ranks, 
                    usp_group_ranks=usp_ranks,
                    iosp_dict=iosp_dict)
    
    
    # 3. init stride map
    init_redundancy_map(config)
    
    if config.use_distrifusion:
        init_fusion_cache()
    elif config.use_easy_cache:
        init_easy_cache()

    # 4. init timer
    if config.enable_timing:
        init_timer(enable=False)
    
    global USE_DITANGO
    USE_DITANGO = True
    logger.info(f"***************************************** RANK {config.rank} - Initialized DiTango! *****************************************")
    
def is_ditango_initialized():
    global USE_DITANGO
    return USE_DITANGO