import torch
import numpy as np
import os
import csv
import time
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Optional, DefaultDict
from pathlib import Path
from ..logger import init_logger

logger = init_logger(__name__)

class RedundancyRecorder:
    """
    记录各时间步、层的平均冗余度，以简洁易读的格式输出到单个文件
    忽略零值冗余度（不参与平均值计算），并提供热图可视化
    """
    
    def __init__(
        self, 
        output_dir: str = "./redundancy_logs",
        file_prefix: str = "redundancy",
        save_interval: int = 10,
        enable_recording: bool = True
    ):
        """
        初始化冗余度记录器
        
        Args:
            output_dir: 输出目录路径
            file_prefix: 日志文件前缀
            save_interval: 数据保存间隔(步数)
            enable_recording: 是否启用记录功能
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.file_prefix = file_prefix
        self.save_interval = save_interval
        self.enable_recording = enable_recording
        
        # 使用二级字典存储：{timestep: {layer_id: [redundancy_values]}}
        self.redundancy_values = collections.defaultdict(lambda: collections.defaultdict(list))
        
        # 存储计算好的平均值：{timestep: {layer_id: avg_redundancy}}
        self.avg_redundancy = collections.defaultdict(dict)
        
        # 记录器状态
        self.last_save_step = -1
        self.current_step = 0
        
        # 创建CSV文件
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.timestamp = timestamp
        self.log_file = self.output_dir / f"{self.file_prefix}_{timestamp}.csv"
        
        # 记录所有已发现的层ID
        self.known_layers = set()
        
        # 初始化文件（稍后会写入实际数据）
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestep", "Layer", "AvgRedundancy", "NonZeroCount", "TotalCount"])
        
        logger.info(f"RedundancyRecorder initialized. Log file: {self.log_file}")
        
    def record_redundancy(
        self, 
        timestep: int, 
        layer_id: int, 
        block_id: int, 
        out_distance: float, 
        lse_distance: float = None  # 不使用，但保留参数兼容性
    ):
        """
        记录冗余度数据，忽略零值
        
        Args:
            timestep: 时间步
            layer_id: 层ID
            block_id: 块ID
            out_distance: 输出张量的相对L1距离
            lse_distance: LSE张量的相对L1距离（不使用）
        """
        if not self.enable_recording:
            return
            
        # 更新当前步
        self.current_step = max(self.current_step, timestep)
        
        # 记录该层该时间步的冗余度值（即使是0也记录，因为需要用于统计总块数）
        # 在计算平均值时会过滤零值
        self.redundancy_values[timestep][layer_id].append(float(out_distance))
        
        # 记录已知层
        self.known_layers.add(layer_id)
        
        # 检查是否需要保存
        if self.current_step - self.last_save_step >= self.save_interval:
            self.compute_averages()
            self.flush_buffer()
            self.last_save_step = self.current_step
    
    def compute_averages(self):
        """计算每个时间步每层的平均冗余度，忽略零值"""
        for ts, layers in self.redundancy_values.items():
            for layer_id, values in layers.items():
                if values:
                    # 过滤掉零值
                    non_zero_values = [v for v in values if v > 0]
                    
                    # 计算非零值的平均值
                    if non_zero_values:
                        avg_value = np.mean(non_zero_values)
                        self.avg_redundancy[ts][layer_id] = {
                            'avg': avg_value,
                            'non_zero_count': len(non_zero_values),
                            'total_count': len(values)
                        }
                    else:
                        # 如果所有值都是零，记录平均值为0
                        self.avg_redundancy[ts][layer_id] = {
                            'avg': 0.0,
                            'non_zero_count': 0,
                            'total_count': len(values)
                        }
    
    def flush_buffer(self):
        """将计算好的平均冗余度写入文件"""
        if not self.avg_redundancy:
            return
            
        # 收集需要写入的数据
        rows_to_write = []
        for ts in sorted(self.avg_redundancy.keys()):
            for layer_id in sorted(self.avg_redundancy[ts].keys()):
                data = self.avg_redundancy[ts][layer_id]
                rows_to_write.append([
                    ts, 
                    layer_id, 
                    f"{data['avg']:.6f}",
                    data['non_zero_count'],
                    data['total_count']
                ])
        
        # 写入文件
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows_to_write)
            
            logger.debug(f"Flushed {len(rows_to_write)} redundancy records to file")
            
            # 清空已处理的数据
            self.redundancy_values.clear()
            self.avg_redundancy.clear()
            
        except Exception as e:
            logger.error(f"Error writing to redundancy log: {e}")
    
    def generate_matrix_file(self):
        """
        生成一个矩阵格式的冗余度文件，行是时间步，列是层
        """
        # 确保所有数据已处理并写入
        self.compute_averages()
        self.flush_buffer()
        
        # 从CSV读取所有数据
        all_data = {}  # {(timestep, layer_id): (redundancy, non_zero_count, total_count)}
        timesteps = set()
        layers = sorted(self.known_layers)
        
        try:
            with open(self.log_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                
                for row in reader:
                    if len(row) >= 5:
                        ts = int(row[0])
                        layer = int(row[1])
                        redundancy = float(row[2])
                        non_zero_count = int(row[3])
                        total_count = int(row[4])
                        
                        all_data[(ts, layer)] = (redundancy, non_zero_count, total_count)
                        timesteps.add(ts)
        except Exception as e:
            logger.error(f"Error reading redundancy data: {e}")
            return None
        
        # 创建矩阵格式文件
        matrix_file = self.output_dir / f"{self.file_prefix}_{self.timestamp}_matrix.csv"
        
        try:
            with open(matrix_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # 写入表头：Timestep和所有层
                header = ["Timestep"] + [f"Layer{layer}" for layer in layers]
                writer.writerow(header)
                
                # 按时间步写入每行
                for ts in sorted(timesteps):
                    row = [ts]
                    for layer in layers:
                        if (ts, layer) in all_data:
                            redundancy, non_zero_count, total_count = all_data[(ts, layer)]
                            # 只有当有非零冗余度时才记录数值
                            if non_zero_count > 0:
                                row.append(f"{redundancy:.6f}")
                            else:
                                row.append("0")  # 明确标记为零
                        else:
                            row.append("")  # 没有数据
                    writer.writerow(row)
                
            logger.info(f"Matrix format redundancy file generated: {matrix_file}")
            return matrix_file
            
        except Exception as e:
            logger.error(f"Error generating matrix file: {e}")
            return None

    def generate_heatmap(self, data_dict, output_file=None):
        """
        根据数据生成传统热图并保存 - 修复版本
        
        Args:
            data_dict: 字典 {(timestep, layer): value}
            output_file: 输出文件路径，如果为None则使用默认路径
        
        Returns:
            保存的文件路径
        """
        # 如果没有指定输出文件，使用默认路径
        if output_file is None:
            output_file = self.output_dir / f"{self.file_prefix}_{self.timestamp}_heatmap.png"
        
        # 整理数据为矩阵格式
        timesteps = sorted(set(key[0] for key in data_dict.keys()))
        layers = sorted(set(key[1] for key in data_dict.keys()))
        
        # 如果数据太多，采样以减少尺寸
        if len(timesteps) > 50:
            step = max(1, len(timesteps) // 50)
            timesteps = timesteps[::step]
        
        if len(layers) > 30:
            step = max(1, len(layers) // 30)
            layers = layers[::step]
        
        # 创建数据矩阵
        data_matrix = np.zeros((len(timesteps), len(layers)))
        
        # 填充数据矩阵，无数据的位置保持为0
        for i, ts in enumerate(timesteps):
            for j, layer in enumerate(layers):
                if (ts, layer) in data_dict:
                    redundancy, non_zero_count, _ = data_dict[(ts, layer)]
                    if non_zero_count > 0:
                        data_matrix[i, j] = redundancy
        
        # 设置Matplotlib风格
        plt.style.use('default')
        mpl.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16
        })
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
        
        # 自定义色彩映射，处理零值
        cmap = plt.cm.viridis.copy()
        cmap.set_under('lightgray')  # 零值以下的颜色（用于无数据区域）
        
        # 确保非零的最小值略大于0，使得零值可以被正确识别
        vmin = 0.00001
        vmax = max(0.3, np.max(data_matrix)) if np.max(data_matrix) > 0 else 0.3
        
        # 绘制热图，使用set_under处理零值
        im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', interpolation='none',
                    vmin=vmin, vmax=vmax)
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, pad=0.02, extend='both')
        cbar.set_label('Redundancy (L1 Distance)')
        cbar.cmap.set_under('lightgray')
        
        # 设置坐标轴刻度
        # 为了避免拥挤，仅显示部分时间步刻度
        if len(timesteps) > 10:
            step = max(1, len(timesteps) // 10)
            yticks = np.arange(0, len(timesteps), step)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{timesteps[i]}" for i in yticks])
        else:
            ax.set_yticks(np.arange(len(timesteps)))
            ax.set_yticklabels([f"{ts}" for ts in timesteps])
        
        # 同样处理层刻度
        if len(layers) > 10:
            step = max(1, len(layers) // 10)
            xticks = np.arange(0, len(layers), step)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"L{layers[i]}" for i in xticks])
        else:
            ax.set_xticks(np.arange(len(layers)))
            ax.set_xticklabels([f"L{layer}" for layer in layers])
        
        # 设置标题和标签
        ax.set_title('Redundancy Heatmap (Output Tensor L1 Distance)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Timestep')
        
        # 添加网格线
        ax.set_xticks(np.arange(-.5, len(layers), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(timesteps), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        try:
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Heatmap saved to {output_file}")
            plt.close(fig)
            return output_file
        except Exception as e:
            logger.error(f"Error saving heatmap: {e}")
            plt.close(fig)
            return None
    def generate_stridemap(self, data_dict, output_file=None):
        """
        根据数据生成基于分位数的四色分布图(Stridemap)
        每种颜色大致包含相同数量的数据点
        
        Args:
            data_dict: 字典 {(timestep, layer): value}
            output_file: 输出文件路径，如果为None则使用默认路径
        
        Returns:
            保存的文件路径
        """
        # 如果没有指定输出文件，使用默认路径
        if output_file is None:
            output_file = self.output_dir / f"{self.file_prefix}_{self.timestamp}_stridemap.png"
        
        # 整理数据为矩阵格式
        timesteps = sorted(set(key[0] for key in data_dict.keys()))
        layers = sorted(set(key[1] for key in data_dict.keys()))
        
        # 如果数据太多，采样以减少尺寸
        if len(timesteps) > 50:
            step = max(1, len(timesteps) // 50)
            timesteps = timesteps[::step]
        
        if len(layers) > 30:
            step = max(1, len(layers) // 30)
            layers = layers[::step]
        
        # 创建数据矩阵
        data_matrix = np.zeros((len(timesteps), len(layers)))
        # 保存所有非零值
        non_zero_values = []
        
        # 填充数据矩阵，同时收集所有非零值
        for i, ts in enumerate(timesteps):
            for j, layer in enumerate(layers):
                if (ts, layer) in data_dict:
                    redundancy, non_zero_count, _ = data_dict[(ts, layer)]
                    if non_zero_count > 0:
                        data_matrix[i, j] = redundancy
                        non_zero_values.append(redundancy)
        
        # 如果没有足够的非零值，使用默认颜色方案
        if len(non_zero_values) < 4:
            logger.warning("Not enough non-zero values for quartile coloring, using default color scheme")
            # 设置默认的边界值
            boundaries = [0.00001, 0.05, 0.1, 0.2, 0.3]
            color_labels = ["No Data", "Very Low", "Low", "Medium", "High"]
        else:
            # 计算四分位数（将非零值分成近似相等的四组）
            non_zero_values = np.array(non_zero_values)
            q1 = np.percentile(non_zero_values, 20)
            q2 = np.percentile(non_zero_values, 50)
            q3 = np.percentile(non_zero_values, 80)
            max_val = np.max(non_zero_values)
            
            # 确保边界值不会重复
            if q1 == q2:
                q1 = 0.9 * q2
            if q2 == q3:
                q3 = 1.1 * q2
            if q3 == max_val:
                max_val = 1.1 * q3
                
            # 设置边界值
            boundaries = [0.00001, q1, q2, q3, max_val]
            
            color_labels = ["No Data", "Q1 (0-20%)", "Q2 (20-50%)", "Q3 (50-80%)", "Q4 (80-100%)"]
        
        # 设置Matplotlib风格
        plt.style.use('default')
        mpl.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16
        })
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
        
        # 创建自定义的颜色映射
        colors = ['lightgray', 'royalblue', 'green', 'orange', 'crimson']
        cmap = mpl.colors.ListedColormap(colors[1:])  # 不包括灰色（用于零值）
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
        
        # 创建掩码，使零值区域为灰色
        zero_mask = data_matrix <= 0.00001
        
        # 为了让灰色区域显示正确，先绘制灰色背景
        ax.imshow(np.ones_like(data_matrix), cmap=mpl.colors.ListedColormap(['lightgray']), 
                aspect='auto', interpolation='none')
        
        # 绘制热图（非零值）
        im = ax.imshow(np.ma.masked_where(zero_mask, data_matrix), 
                    cmap=cmap, norm=norm, aspect='auto', interpolation='none')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, pad=0.02, ticks=[(boundaries[i] + boundaries[i+1])/2 for i in range(len(boundaries)-1)])
        cbar.set_label('Redundancy (L1 Distance)')
        cbar.ax.set_yticklabels(color_labels[1:])  # 不包括"No Data"
        
        # 设置坐标轴刻度
        # 为了避免拥挤，仅显示部分时间步刻度
        if len(timesteps) > 10:
            step = max(1, len(timesteps) // 10)
            yticks = np.arange(0, len(timesteps), step)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{timesteps[i]}" for i in yticks])
        else:
            ax.set_yticks(np.arange(len(timesteps)))
            ax.set_yticklabels([f"{ts}" for ts in timesteps])
        
        # 同样处理层刻度
        if len(layers) > 10:
            step = max(1, len(layers) // 10)
            xticks = np.arange(0, len(layers), step)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"L{layers[i]}" for i in xticks])
        else:
            ax.set_xticks(np.arange(len(layers)))
            ax.set_xticklabels([f"L{layer}" for layer in layers])
        
        # 设置标题和标签
        ax.set_title('Redundancy Stridemap (Equal Quartile Distribution)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Timestep')
        
        # 添加网格线
        ax.set_xticks(np.arange(-.5, len(layers), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(timesteps), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.3)
        
        # 添加图例说明零值区域
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightgray', label='No Data (0)')]
        for i in range(len(boundaries)-1):
            legend_elements.append(Patch(facecolor=colors[i+1], label=f'{color_labels[i+1]} ({boundaries[i]:.6f}-{boundaries[i+1]:.6f})'))
        
        ax.legend(handles=legend_elements, loc='upper center', 
                bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
        
        # 调整布局，为图例留出空间
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        # 保存图形
        try:
            plt.savefig(output_file, bbox_inches='tight')
            logger.info(f"Stridemap saved to {output_file}")
            
            # 同时保存分布信息到文本文件
            dist_file = str(output_file).replace('.png', '_distribution.txt')
            with open(dist_file, 'w') as f:
                f.write(f"Redundancy Distribution Statistics:\n\n")
                f.write(f"Total data points: {len(timesteps) * len(layers)}\n")
                f.write(f"Non-zero points: {len(non_zero_values)}\n")
                f.write(f"Zero/No-data points: {len(timesteps) * len(layers) - len(non_zero_values)}\n\n")
                
                f.write(f"Distribution boundaries (quartiles):\n")
                for i in range(len(boundaries)-1):
                    if i == 0:
                        f.write(f"No Data: 0.0\n")
                    count = np.sum((data_matrix > boundaries[i]) & (data_matrix <= boundaries[i+1]))
                    percentage = 100 * count / len(non_zero_values) if len(non_zero_values) > 0 else 0
                    f.write(f"{color_labels[i+1]}: {boundaries[i]:.6f} - {boundaries[i+1]:.6f} ({count} points, {percentage:.2f}%)\n")
                
            logger.info(f"Distribution information saved to {dist_file}")
            
            plt.close(fig)
            return output_file, dist_file
        except Exception as e:
            logger.error(f"Error saving stridemap: {e}")
            plt.close(fig)
            return None, None    
    def generate_summary(self):
        """生成摘要报告、热图和分布图，忽略零值"""
        # 确保所有数据已处理
        self.compute_averages()
        self.flush_buffer()
        
        # 创建矩阵格式文件（便于查看）
        self.generate_matrix_file()
        
        # 创建摘要文件
        summary_file = self.output_dir / f"{self.file_prefix}_{self.timestamp}_summary.txt"
        
        try:
            # 从CSV读取所有数据
            all_data = {}  # {(timestep, layer_id): (redundancy, non_zero_count, total_count)}
            timesteps = set()
            layers = sorted(self.known_layers)
            
            with open(self.log_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                
                for row in reader:
                    if len(row) >= 5:
                        ts = int(row[0])
                        layer = int(row[1])
                        redundancy = float(row[2])
                        non_zero_count = int(row[3])
                        total_count = int(row[4])
                        
                        all_data[(ts, layer)] = (redundancy, non_zero_count, total_count)
                        timesteps.add(ts)
            
            # 生成两种可视化图表
            heatmap_file = self.generate_heatmap(all_data)
            stridemap_file, dist_file = self.generate_stridemap(all_data)
            
            heatmap_filename = os.path.basename(heatmap_file) if heatmap_file else "heatmap generation failed"
            stridemap_filename = os.path.basename(stridemap_file) if stridemap_file else "stridemap generation failed"
            
            # 计算各种统计信息
            layer_stats = {}  # {layer: [min, max, avg, std, non_zero_ratio]}
            timestep_stats = {}  # {timestep: [min, max, avg, std, non_zero_ratio]}
            
            # 计算每层的统计信息（只考虑非零值）
            for layer in layers:
                # 收集该层所有时间步的非零冗余度
                values = []
                non_zero_total = 0
                total_blocks = 0
                
                for ts in timesteps:
                    if (ts, layer) in all_data:
                        redundancy, non_zero_count, total_count = all_data[(ts, layer)]
                        if non_zero_count > 0:  # 只有当有非零值时才加入统计
                            values.append(redundancy)
                        non_zero_total += non_zero_count
                        total_blocks += total_count
                
                if values:
                    # 计算非零比例
                    non_zero_ratio = non_zero_total / total_blocks if total_blocks > 0 else 0
                    
                    layer_stats[layer] = [
                        np.min(values),
                        np.max(values),
                        np.mean(values),
                        np.std(values) if len(values) > 1 else 0,
                        non_zero_ratio
                    ]
            
            # 计算每个时间步的统计信息（只考虑非零值）
            for ts in sorted(timesteps):
                # 收集该时间步所有层的非零冗余度
                values = []
                non_zero_total = 0
                total_blocks = 0
                
                for layer in layers:
                    if (ts, layer) in all_data:
                        redundancy, non_zero_count, total_count = all_data[(ts, layer)]
                        if non_zero_count > 0:  # 只有当有非零值时才加入统计
                            values.append(redundancy)
                        non_zero_total += non_zero_count
                        total_blocks += total_count
                
                if values:
                    # 计算非零比例
                    non_zero_ratio = non_zero_total / total_blocks if total_blocks > 0 else 0
                    
                    timestep_stats[ts] = [
                        np.min(values),
                        np.max(values),
                        np.mean(values),
                        np.std(values) if len(values) > 1 else 0,
                        non_zero_ratio
                    ]
            
            # 写入摘要文件
            with open(summary_file, 'w') as f:
                f.write("= Redundancy Analysis Summary =\n")
                f.write("Note: All statistics exclude zero values\n\n")
                
                # 提到生成的两种可视化图表
                f.write(f"Visualizations:\n")
                f.write(f"- Heatmap (continuous scale): {heatmap_filename}\n")
                f.write(f"- Stridemap (quartile distribution): {stridemap_filename}\n\n")
                
                # 整体统计信息
                all_values = []
                total_non_zero = 0
                total_blocks = 0
                
                for key, (redundancy, non_zero_count, total_count) in all_data.items():
                    if non_zero_count > 0:
                        all_values.append(redundancy)
                    total_non_zero += non_zero_count
                    total_blocks += total_count
                
                non_zero_ratio = total_non_zero / total_blocks if total_blocks > 0 else 0
                
                f.write(f"Total Records: {len(all_data)}\n")
                f.write(f"Total Blocks: {total_blocks}\n")
                f.write(f"Non-Zero Blocks: {total_non_zero} ({non_zero_ratio*100:.2f}%)\n")
                
                if all_values:
                    f.write(f"Overall Average Redundancy (non-zero only): {np.mean(all_values):.6f}\n")
                    f.write(f"Overall Min Redundancy: {np.min(all_values):.6f}\n")
                    f.write(f"Overall Max Redundancy: {np.max(all_values):.6f}\n")
                    f.write(f"Overall Std Deviation: {np.std(all_values):.6f}\n")
                else:
                    f.write("No non-zero redundancy values found.\n")
                f.write("\n")
                
                # 每层统计信息
                f.write("= Layer Statistics =\n\n")
                f.write("Layer | Min      | Max      | Avg      | Std Dev  | Non-Zero %\n")
                f.write("------+----------+----------+----------+----------+----------\n")
                for layer in layers:
                    if layer in layer_stats:
                        stats = layer_stats[layer]
                        f.write(f"{layer:5d} | {stats[0]:.6f} | {stats[1]:.6f} | {stats[2]:.6f} | {stats[3]:.6f} | {stats[4]*100:8.2f}%\n")
                f.write("\n")
                
                # 时间步统计信息（只显示部分典型时间步）
                f.write("= Timestep Statistics (selected) =\n\n")
                f.write("TS    | Min      | Max      | Avg      | Std Dev  | Non-Zero %\n")
                f.write("------+----------+----------+----------+----------+----------\n")
                
                # 选择有代表性的时间步展示
                step_list = sorted(timesteps)
                if len(step_list) > 20:
                    step = max(1, len(step_list) // 20)
                    display_steps = step_list[::step]
                else:
                    display_steps = step_list
                
                for ts in display_steps:
                    if ts in timestep_stats:
                        stats = timestep_stats[ts]
                        f.write(f"{ts:5d} | {stats[0]:.6f} | {stats[1]:.6f} | {stats[2]:.6f} | {stats[3]:.6f} | {stats[4]*100:8.2f}%\n")
                f.write("\n")
                
                # 简单的ASCII热图（时间步×层）
                f.write("= Simple ASCII Redundancy Heatmap =\n")
                f.write("(For detailed visualization, see the generated heatmap and stridemap images)\n\n")
                
                # 选择要显示的层和时间步
                if len(layers) > 10:
                    display_layers = layers[::max(1, len(layers)//10)]
                else:
                    display_layers = layers
                    
                if len(step_list) > 20:
                    step = max(1, len(step_list) // 20)
                    display_steps = step_list[::step]
                else:
                    display_steps = step_list
                
                # 打印热图表头
                f.write("      |")
                for layer in display_layers:
                    f.write(f" L{layer:2d} |")
                f.write("\n")
                
                f.write("------+" + "-----+" * len(display_layers) + "\n")
                
                # 打印热图内容
                for ts in display_steps:
                    f.write(f"T{ts:4d} |")
                    for layer in display_layers:
                        if (ts, layer) in all_data:
                            redundancy, non_zero_count, total_count = all_data[(ts, layer)]
                            
                            # 只有当有非零冗余度时才显示强度
                            if non_zero_count > 0:
                                # 使用简单的ASCII符号表示强度
                                if redundancy < 0.05:
                                    symbol = "  . "
                                elif redundancy < 0.1:
                                    symbol = "  : "
                                elif redundancy < 0.2:
                                    symbol = "  * "
                                elif redundancy < 0.3:
                                    symbol = "  # "
                                else:
                                    symbol = "  @ "
                            else:
                                symbol = "  0 "  # 明确标记为零
                            f.write(f"{symbol}|")
                        else:
                            f.write("     |")
                    f.write("\n")
                
                f.write("\n")
                f.write("Symbol legend: 0 (zero/no data)  . (<0.05)  : (<0.1)  * (<0.2)  # (<0.3)  @ (>=0.3)\n")
                
            logger.info(f"Redundancy summary generated: {summary_file}")
            return {
                "summary": str(summary_file),
                "heatmap": str(heatmap_file) if heatmap_file else None,
                "stridemap": str(stridemap_file) if stridemap_file else None,
                "dist_file": str(dist_file) if dist_file else None
            }
            
        except Exception as e:
            logger.error(f"Error generating redundancy summary: {e}")
            return None
    
    def clear(self):
        """清空缓冲区"""
        self.redundancy_values.clear()
        self.avg_redundancy.clear()
        logger.info("Redundancy recorder buffer cleared")
    
    def close(self):
        """关闭记录器，确保所有数据已写入，并生成摘要和可视化"""
        self.compute_averages()
        self.flush_buffer()
        
        # 生成矩阵格式文件和摘要
        matrix_file = self.generate_matrix_file()
        summary_results = self.generate_summary()
        
        logger.info("Redundancy recorder closed")
        self.clear()
        
        return {
            "csv": str(self.log_file),
            "matrix": str(matrix_file) if matrix_file else None,
            "summary": summary_results["summary"] if summary_results else None,
            "heatmap": summary_results["heatmap"] if summary_results else None,
            "stridemap": summary_results["stridemap"] if summary_results else None
        }


# 全局冗余记录器实例
REDUNDANCY_RECORDER = None

def init_redundancy_recorder(
    output_dir: str = "./redundancy_logs",
    file_prefix: str = "redundancy",
    save_interval: int = 10,
    enable_recording: bool = True
):
    """初始化全局冗余记录器"""
    global REDUNDANCY_RECORDER
    REDUNDANCY_RECORDER = RedundancyRecorder(
        output_dir=output_dir,
        file_prefix=file_prefix,
        save_interval=save_interval,
        enable_recording=enable_recording
    )
    return REDUNDANCY_RECORDER

def get_redundancy_recorder():
    """获取全局冗余记录器"""
    global REDUNDANCY_RECORDER
    if REDUNDANCY_RECORDER is None:
        REDUNDANCY_RECORDER = RedundancyRecorder()
    return REDUNDANCY_RECORDER