import numpy as np
from PIL import Image
import math
import torch, torchvision
from einops import rearrange
import imageio
import os
from typing import Optional, Dict, List

_PADDING_DICT: Dict[str, List[int]] = {}

def split_tensor_uneven(tensor: torch.Tensor, world_size: int, dim: int = 1, tensor_name: Optional[str] = None) -> List[torch.Tensor]:
    """Split tensor with padding to ensure equal sizes across all ranks.
    
    Args:
        tensor: Input tensor to split
        world_size: Number of splits
        dim: Dimension to split on
        tensor_name: Name of tensor for padding tracking
    
    Returns:
        List of split tensors with padding
    """
    seq_len = tensor.shape[dim]
    base_len = seq_len // world_size
    remainder = seq_len % world_size
    
    # 如果有余数，将base_len加1以确保所有分片大小相同
    target_len = base_len + (1 if remainder > 0 else 0)
    
    splits = []
    start_idx = 0
    padding_sizes = []
    
    for i in range(world_size):
        length = base_len + (1 if i < remainder else 0)
        end_idx = start_idx + length
        
        # 获取当前分片
        split = tensor.narrow(dim, start_idx, length)
        
        # 计算需要的padding大小
        pad_size = target_len - length
        padding_sizes.append(pad_size)
        
        if pad_size > 0:
            # 创建padding tensor
            pad_shape = list(split.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=split.dtype, device=split.device)
            
            # 拼接原始tensor和padding
            split = torch.cat([split, padding], dim=dim)
            
        splits.append(split)
        start_idx = end_idx
    
    # 如果提供了tensor_name，记录padding信息
    if tensor_name is not None:
        _PADDING_DICT[tensor_name] = padding_sizes
        
    return splits

def get_padding_dict() -> Dict[str, List[int]]:
    """获取所有tensor的padding信息"""
    return _PADDING_DICT

def remove_padding_after_gather(tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
    """从gather后的tensor中移除padding"""
    if tensor_name not in _PADDING_DICT:
        return tensor
        
    padding_sizes = _PADDING_DICT[tensor_name]
    world_size = len(padding_sizes)
    
    # 计算每个rank的实际长度（不包含padding）
    actual_lengths = []
    for i in range(world_size):
        total_len = tensor.size(1) // world_size
        actual_len = total_len - padding_sizes[i]
        actual_lengths.append(actual_len)
    
    # 创建索引列表
    indices = []
    start_idx = 0
    for rank in range(world_size):
        segment_len = tensor.size(1) // world_size
        valid_len = segment_len - padding_sizes[rank]
        indices.extend(range(start_idx, start_idx + valid_len))
        start_idx += segment_len
    
    # 使用索引选择有效数据
    return tensor.index_select(1, torch.tensor(indices, device=tensor.device))
def save_hwc_tensor(tensor, save_path):
    # 确保tensor在CPU上
    tensor = tensor.cpu()
    
    # 转numpy
    img = tensor.numpy()
    
    # 处理值范围
    if img.dtype in [np.float32, np.float64]:
        if img.max() <= 1:
            img = img * 255
            img = np.clip(img, 0, 255)  # 裁剪到[0,255]
    
    img = img.astype(np.uint8)
    
    # 转PIL并保存
    img = Image.fromarray(img)
    img.save(save_path)

# ============= Hunyuan ==============
import collections.abc
from itertools import repeat
def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def align_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)