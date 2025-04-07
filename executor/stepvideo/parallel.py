import torch.distributed as dist
# import xfuser
import torch
import os
from ditango.core.parallel_state import init_distributed_environment, init_model_parallel, get_world_group, generate_parallel_groups, get_usp_group
from ditango.utils import split_tensor_uneven, remove_padding_after_gather

def initialize_parall_group():
    pass
    # world_size = int(os.getenv("WORLD_SIZE"))
    # # 获取本地rank而不是全局rank
    # local_rank = int(os.getenv("LOCAL_RANK"))
    # rank = int(os.getenv("RANK"))
    # dist.init_process_group("nccl", world_size=world_size, rank=rank)
    # torch.cuda.set_device(local_rank)
    # init_distributed_environment(
    #     rank=dist.get_rank(), 
    #     world_size=dist.get_world_size()
    # )
    # world_size = get_world_group().world_size
    # cfg_ranks, usp_ranks, iosp_dict = generate_parallel_groups(world_size, do_cfg_guidance=True, sparse_type='full')
    # init_model_parallel(cfg_group_ranks=cfg_ranks, 
    #                     usp_group_ranks=usp_ranks,
    #                     iosp_dict=iosp_dict)
    
    # torch.cuda.set_device(local_rank)

def get_parallel_group():
    return get_world_group()

def get_sequence_parallel_world_size():
    return get_usp_group().world_size

def get_sequence_parallel_rank():
    return get_usp_group().rank_in_group

def get_sp_group():
    return get_usp_group()



def parallel_forward(fn_):
    def wrapTheFunction(_, hidden_states, *args, **kwargs):
        if kwargs['parallel']:            
            # hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            # kwargs['attn_mask'] = torch.chunk(kwargs['attn_mask'], get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            hidden_states = split_tensor_uneven(hidden_states, get_sequence_parallel_world_size(), dim=-2, tensor_name="hidden")[get_sequence_parallel_rank()]
            kwargs['attn_mask'] = split_tensor_uneven(kwargs['attn_mask'], get_sequence_parallel_world_size(), dim=-2, tensor_name='mask')[get_sequence_parallel_rank()]
        output = fn_(_, hidden_states, *args, **kwargs)
        
        if kwargs['parallel']:
            output = get_sp_group().all_gather(output.contiguous(), dim=-2)
            output = remove_padding_after_gather(output, "hidden")
        return output
     
    return wrapTheFunction