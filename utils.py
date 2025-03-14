import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Dict, List, Tuple

_PADDING_DICT: Dict[str, List[int]] = {}

def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
      if slice_ is not None:
        raise RuntimeError("first update_out_and_lse should not pass slice_ args")
      out = block_out.to(torch.float32)
      lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
      slice_out, slice_lse = out[slice_], lse[slice_]
      slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
      out[slice_], lse[slice_] = slice_out, slice_lse
    else:
      out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse

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


