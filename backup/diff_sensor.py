import csv
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from threading import Lock


class LayerWiseLogger:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, log_file: str = "layer_log.csv", clear_existing: bool = True):
        if hasattr(self, '_initialized'):
            return
        
        # 初始化配置
        self.log_file = Path(log_file)
        self.layer_data: Dict[str, Tuple[int, torch.Tensor, torch.Tensor]] = {}  # {layer_id: (last_step, out, lse)}
        self._write_lock = Lock()
        
        # 初始化CSV文件头
        mode = 'w' if clear_existing or not self.log_file.exists() else 'a'
        with self._write_lock, open(self.log_file, mode, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "layer", "mse_out", "mse_lse"])
        
        self._initialized = True
    
    def log_layer(self,
                 current_step: int,
                 layer_id: str,
                 current_out: torch.Tensor,  # shape [1, 4107, 48, 64]
                 current_lse: torch.Tensor): # shape [1, 4107, 48, 1]
        """
        记录当前层输出并自动计算与上一次记录的差异
        
        :param current_step: 当前全局时间步 (由调用方维护)
        :param layer_id: 层唯一标识符
        :param current_out: 当前输出张量
        :param current_lse: 当前对数求和指数张量
        """
        # 数据预处理
        current_out = current_out.detach().cpu().float()
        current_lse = current_lse.squeeze(-1).detach().cpu().float()
        
        # 计算差异（如果存在有效历史数据）
        mse_out, mse_lse = None, None
        if layer_id in self.layer_data:
            last_step, last_out, last_lse = self.layer_data[layer_id]
            
            # 仅当上一步是当前步的前一步时计算差异
            if last_step == current_step - 1:
                mse_out = torch.mean((current_out - last_out)**2).item()
                mse_lse = torch.mean((current_lse - last_lse)**2).item()
        
        # 更新存储（总是覆盖旧数据）
        self.layer_data[layer_id] = (current_step, current_out, current_lse)
        
        # 写入日志（仅当有有效差异时）
        if mse_out is not None:
            with self._write_lock, open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    current_step,
                    layer_id,
                    f"{mse_out:.6e}",
                    f"{mse_lse:.6e}"
                ])

# 全局访问接口
def get_diff_sensor() -> LayerWiseLogger:
    if LayerWiseLogger._instance is None:
        return None
    return LayerWiseLogger._instance

def init_diff_sensor(log_file: str = "layer_log.csv", clear_existing: bool = True):
    """
    初始化全局层级日志记录器
    
    :param log_file: 日志文件路径
    :param clear_existing: 是否清空已存在的文件内容
    """
    print(f"================= Using Redundancy sensor! Inference process would be slow down! ============", flush=True)
    LayerWiseLogger(log_file=log_file, clear_existing=clear_existing)

# 使用示例 -------------------------------------------------------------------
if __name__ == "__main__":
    # 初始化日志器，默认会清空已有文件
    init_diff_sensor("example_log.csv")
    logger = get_diff_sensor()
    
    # 模拟全局时间步管理
    global_step = 0
    
    # 时间步0记录（无历史数据）
    for layer_idx in range(3):
        logger.log_layer(
            current_step=global_step,
            layer_id=f"block_{layer_idx}",
            current_out=torch.randn(1, 4107, 48, 64),
            current_lse=torch.randn(1, 4107, 48, 1)
        )
    
    # 时间步1记录（计算差异）
    global_step += 1
    for layer_idx in range(3):
        logger.log_layer(
            current_step=global_step,
            layer_id=f"block_{layer_idx}",
            current_out=torch.randn(1, 4107, 48, 64),
            current_lse=torch.randn(1, 4107, 48, 1)
        )