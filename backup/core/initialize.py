import torch
from typing import List
from argparse import ArgumentParser
# from .redundancy_map import init_redundancy_map, print_redundancy_map
from .arguments import init_config, print_config
from .feature_cache import init_cache
from .parallel_state import init_distributed_environment, init_model_parallel, get_world_group, generate_parallel_groups
from ..diff_sensor import init_diff_sensor
from ..timer import init_timer
from ..logger import init_logger
from ..pro.pro_map import init_stride_map
from ..pro.sensor import init_redundancy_sensor

logger = init_logger(__name__)
USE_DITANGO = False

def init_ditango(
    config_path: str = None,
    timestep_indices: List = [], 
    split_list: List = [], 
    pattern_list: List = [],
    use_diff_sensor: bool = False,
    use_timer: bool = False,
):
    # 1. init arguments
    config = init_config(config_path)
    
    # 2. init parallel
    init_distributed_environment(world_size=config.world_size, rank=config.rank, local_rank=config.local_rank)
    print_config()
    local_rank = get_world_group().local_rank
    torch.cuda.set_device(local_rank)
    stride_list = sorted(set(pattern_list))
    if config.use_distrifusion:
        assert config.model_name not in ['latte', 'opensora'], f"Unsupported model type {config.model_name} for DistriFusion Baseline."
        sparse_type = stride_list
    elif config.model_name == 'opensora':
        sparse_type = stride_list # sparse_n=1
        for stride in stride_list: # sparse_n=4
            if stride > 4:
                assert stride % 4 == 0
                new_stride = stride // 4
                if new_stride not in sparse_type:
                    sparse_type.append(new_stride)
    elif config.model_name == 'latte':
        sparse_type = 'sequential'
    else:
        sparse_type = 'full' 
        
    cfg_ranks, usp_ranks, iosp_dict = generate_parallel_groups(config.world_size, do_cfg_guidance=config.do_cfg_parallel, sparse_type=sparse_type)
    init_model_parallel(cfg_group_ranks=cfg_ranks, 
                    usp_group_ranks=usp_ranks,
                    iosp_dict=iosp_dict)
    
    
    # 3. init stride map
    map = init_stride_map(config)
    
    # init_redundancy_sensor(
    #     model_name=config.model_name,
    # )
    
    # map.set_pattern_for_rows(
    #     timestep_indices,
    #     split_list,
    #     pattern_list,
    # )
    # map.set_pattern_for_rows(
    #     timestep_indices=list(range(1,46,4)),
    #     split_list=[],
    #     pattern_list=[8]
    # )
    
    if config.use_easy_cache:
        map.set_full_stride_for_rows(
            timestep_indices=list(range(5,48,4))
        )
    
    # print_redundancy_map(logger)
    
    # 4. init feature cache
    # init_cache(config)
    
    # 5. init timer
    if use_timer:
        init_timer(enable=False)
    # 5. init redundancy sensor
    if use_diff_sensor:
        init_diff_sensor(f"./exp/diff/{config.model_name}_diff.csv") 
    global USE_DITANGO
    USE_DITANGO = True
    logger.info(f"***************************************** RANK {config.rank} - Initialized DiTango! *****************************************")
    
def is_ditango_initialized():
    global USE_DITANGO
    return USE_DITANGO