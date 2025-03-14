from typing import List

from .redundancy_map import init_redundancy_map, print_redundancy_map
from .arguments import init_args, print_args
from .feature_cache import init_cache
from .parallel_state import init_distributed_environment, init_model_parallel, get_world_group, generate_parallel_groups
from ..diff_sensor import init_diff_sensor
from ..timer import init_timer
from ..logger import init_logger

logger = init_logger(__name__)

def init_ditango(
    timestep_indices: List = [], 
    split_list: List = [], 
    pattern_list: List = [],
    use_diff_sensor: bool = False,
    use_timer: bool = False,
):
    # 1. init arguments
    args = init_args()
    print_args(args)
    
    # 2. init parallel
    init_distributed_environment(world_size=args.world_size, rank=args.rank, local_rank=args.local_rank)
    stride_list = sorted(set(pattern_list))
    if args.use_distrifusion:
        assert args.model_type not in ['latte', 'opensora'], f"Unsupported model type {args.model_type} for DistriFusion Baseline."
        sparse_type = stride_list
    elif args.model_type == 'opensora':
        sparse_type = stride_list # sparse_n=1
        for stride in stride_list: # sparse_n=4
            if stride > 4:
                assert stride % 4 == 0
                new_stride = stride // 4
                if new_stride not in sparse_type:
                    sparse_type.append(new_stride)
    elif args.model_type == 'latte':
        sparse_type = 'sequential'
    else:
        sparse_type = 'full' 
        
    cfg_ranks, usp_ranks, iosp_dict = generate_parallel_groups(args.world_size, do_cfg_guidance=args.do_cfg_parallel, sparse_type=sparse_type)
    init_model_parallel(cfg_group_ranks=cfg_ranks, 
                    usp_group_ranks=usp_ranks,
                    iosp_dict=iosp_dict)
    
    
    # 3. init stride map
    map = init_redundancy_map(args)
    
    map.set_pattern_for_rows(
        timestep_indices,
        split_list,
        pattern_list,
    )
    
    if args.use_easy_cache:
        map.set_full_stride_for_rows(
            timestep_indices=list(range(5,48,4))
        )
    
    print_redundancy_map(logger)
    
    # 4. init feature cache
    init_cache(args)
    
    # 5. init timer
    if use_timer:
        init_timer(enable=False)
    # 5. init redundancy sensor
    if use_diff_sensor:
        init_diff_sensor(f"./exp/diff/{args.model_type}_diff.csv") 
    
    logger.info(f"***************************************** RANK {args.rank} - Initialized DiTango! *****************************************")