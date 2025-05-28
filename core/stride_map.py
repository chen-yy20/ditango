import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Optional, Dict, Tuple

import torch.distributed
from .parallel_state import get_usp_group
from ..utils import rel_l1, get_timestep
from ..logger import init_logger
from .config import get_config

logger = init_logger(__name__)

class StrideMap:
    def __init__(self, args):
        """
        Initialize StrideMap
        
        Args:
            num_timesteps: Number of diffusion timesteps
            num_layers: Number of transformer layers 
            base_weight: Base weight value to initialize the map
        """
        self.num_timesteps = args.num_inference_steps
        self.num_layers = args.num_layers
        self.base_weight = get_usp_group().world_size
        
        self.divider_map = self._init_map()
        self.stride_divider = [8, 4, 2, 1]
        self.map_percentage = [20, 50, 80]
        
        # Auto setting 
        self.auto_setting = False
        self.calc_round = 0
        self.prev_out = [None] * self.num_layers
        self.redundancy_map = None

    def _init_map(self) -> torch.Tensor:
        """Initialize stride map with base weight"""
        return torch.ones(self.num_timesteps, self.num_layers, dtype=torch.long)
    
    def is_auto_setting(self):
        """Check if auto setting mode is enabled"""
        return self.auto_setting
    
    def get_curr_isp_stride(self, timestep: int, layer_id: int) -> int:
        """
        Get current ISP stride for a specific timestep and layer
        
        Args:
            timestep: Current diffusion timestep index
            layer_id: Layer index in transformer
            
        Returns:
            Stride value to use for current ISP communication
        """
        if self.auto_setting:
            return self.base_weight   
        divider = self.divider_map[timestep, layer_id].item()
        if self.base_weight < divider:
            return 1
        else:
            assert self.base_weight % divider == 0, f"Invalid divider {divider} for base weight {self.base_weight}"
        return max(self.base_weight // divider, 1)
    
    def get_next_isp_stride(self, timestep: int, layer_id: int) -> int:
        """
        Get next ISP stride for a specific timestep and layer
        
        Args:
            timestep: Current diffusion timestep index
            layer_id: Layer index in transformer
            
        Returns:
            Stride value to use for next timestep's ISP communication
        """
        if self.auto_setting or timestep == self.num_timesteps - 1:
            return self.base_weight 
        divider = self.divider_map[timestep+1, layer_id].item()
        if self.base_weight < divider:
            return 1
        else:
            assert self.base_weight % divider == 0, f"Invalid divider {divider} for base weight {self.base_weight}"
        return self.base_weight // divider
    
    # Manual setting of stride map
    def set_pattern_for_row(self, row_idx: int, split_list: List, pattern_list: List):
        """
        Set pattern (a,b,a) for a specific row
        
        Args:
            row_idx: Index of the row to modify
            split_list: List of indices where to split the layer range
            pattern_list: List of pattern values to set for each chunk
        """
        if row_idx >= self.num_timesteps:
            raise ValueError(f"Row index {row_idx} exceeds num_timesteps {self.num_timesteps}")
        else:
            if pattern_list is None or len(pattern_list) == 0:
                return
            assert len(split_list) + 1 == len(pattern_list), "Number of patterns does not match number of split chunks. Expected {} patterns for {} splits.".format(len(split_list) + 1, len(pattern_list))
        if split_list is None or len(split_list) == 0:
            assert pattern_list[0] <= self.base_weight, f"isp size {pattern_list[0]} should be less than osp size {self.base_weight}"
            self.divider_map[row_idx, :] = pattern_list[0]
            return
        
        # Set first stride
        assert pattern_list[0] <= self.base_weight, f"isp size {pattern_list[0]} should be less than osp size {self.base_weight}"
        self.divider_map[row_idx, :split_list[0]] = pattern_list[0]
        # Set middle stride 
        for i in range(len(split_list) - 1):
            assert pattern_list[i+1] <= self.base_weight, f"isp size {pattern_list[i+1]} should be less than osp size {self.base_weight}"
            self.divider_map[row_idx, split_list[i]:split_list[i+1]] = pattern_list[i+1]
        # Set right stride
        assert pattern_list[-1] <= self.base_weight, f"isp size {pattern_list[-1]} should be less than osp size {self.base_weight}"
        self.divider_map[row_idx, split_list[-1]:] = pattern_list[-1]
        
    def set_pattern_for_rows(self, timestep_indices, split_list: List = [], pattern_list: List = []):
        """
        Set pattern for multiple rows specified by indices
        
        Args:
            timestep_indices: List of row indices or range object
            split_list: List of indices where to split the layer range
            pattern_list: List of pattern values to set for each chunk
        """
        if len(timestep_indices) == 0 or len(pattern_list) == 0:
            return
        if isinstance(timestep_indices, range):
            timestep_indices = list(timestep_indices)

        for idx in timestep_indices:
            self.set_pattern_for_row(idx, split_list, pattern_list)
            
    def set_full_stride_for_rows(self, timestep_indices):
        """
        Set full stride (default) for specified rows
        
        Args:
            timestep_indices: List of row indices or range object
        """
        if isinstance(timestep_indices, range):
            timestep_indices = list(timestep_indices)
        self.set_pattern_for_row(timestep_indices[0], [], [self.base_weight])
        for idx in timestep_indices[1:]:
            self.set_pattern_for_row(idx, [], [self.base_weight])
    
    def get_map(self) -> torch.Tensor:
        """Return the stride map tensor"""
        return self.divider_map
    
    def reset_map(self):
        """Reset stride map to initial state"""
        self.divider_map = self._init_map()
        
    # Preprocess to Autoset StrideMap
    def enable_auto_setting(self):
        """Enable auto setting mode - collects redundancy data during inference"""
        try:
            current_timestep = get_timestep()
            if current_timestep != 0:
                logger.warning(f"Auto setting enabled at timestep {current_timestep}, should be 0")
        except:
            pass
        
        self.auto_setting = True
        self.redundancy_map = torch.zeros(self.num_timesteps, self.num_layers, dtype=torch.float)
        self.prev_out = [None] * self.num_layers
        logger.info("Auto setting enabled for StrideMap")
        
    def record_out_redundancy(self, timestep: int, layer_id: int, curr_out: torch.Tensor):
        """
        Record redundancy between current and previous outputs
        
        Args:
            timestep: Current diffusion timestep index
            layer_id: Layer index in transformer
            curr_out: Current tensor output from the layer
        """
        if not self.auto_setting:
            return
            
        prev_out = self.prev_out[layer_id]
        if prev_out is None:
            self.prev_out[layer_id] = curr_out
            logger.info(f"T{timestep} L{layer_id} | {curr_out.shape=}, memory={torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
            return
        # Calculate relative L1 distance
        # logger.debug(f"T{timestep} L{layer_id} R{torch.distributed.get_rank()} | {curr_out[:2]=}, {prev_out[:2]=}")
        rel_l1_distance = rel_l1(curr_out, prev_out)
        # logger.debug(f"T{timestep} L{layer_id} R{torch.distributed.get_rank()}| Auto Setting: {self.redundancy_map=}, {self.redundancy_map[timestep, layer_id]}, add: {rel_l1_distance}")
        self.redundancy_map[timestep, layer_id] += rel_l1_distance
        self.prev_out[layer_id] = curr_out
        
    def generate_divider_map_from_redundancy(self, custom_percentages=None, custom_dividers=None):
        """
        Generate divider map from redundancy data with customizable parameters
        
        Args:
            custom_percentages: Optional custom percentile thresholds [low, mid, high]
            custom_dividers: Optional custom divider values [lowest, low, mid, high]
            
        Returns:
            percentiles: The calculated percentile values used for mapping
        """
        if self.redundancy_map is None:
            logger.warning("Cannot generate divider map: no redundancy data available")
            return None
            
        # Use custom parameters if provided, otherwise use defaults
        percentages = custom_percentages if custom_percentages is not None else self.map_percentage
        dividers = custom_dividers if custom_dividers is not None else self.stride_divider
        
        # Collect all non-zero redundancy values
        non_zero_values = []
        for t in range(self.num_timesteps):
            for l in range(self.num_layers):
                redundancy = self.redundancy_map[t, l].item()
                if redundancy > 0:
                    non_zero_values.append(redundancy)
        
        if len(non_zero_values) == 0:
            logger.warning("No non-zero redundancy values found, keeping default divider map")
            return None
        
        # Calculate percentiles for redundancy distribution
        non_zero_values = np.array(non_zero_values)
        percentiles = [np.percentile(non_zero_values, p) for p in percentages]
        
        logger.info(f"Generating divider map with percentiles: {percentiles}")
        logger.info(f"Corresponding to percentages: {percentages}")
        logger.info(f"Using dividers: {dividers}")
        
        # Reset divider map to base values
        self.divider_map = self._init_map()
        
        # Set divider based on redundancy value
        for t in range(self.num_timesteps):
            for l in range(self.num_layers):
                if t == self.num_timesteps - 1:
                    self.divider_map[t, l] = 1
                    continue
                redundancy = self.redundancy_map[t, l].item()
                
                # Only process non-zero redundancy values
                if redundancy > 0:
                    # Map redundancy to appropriate divider
                    if redundancy < percentiles[0]:  # 0-low%
                        divider = dividers[0]  # Lowest divider (largest stride)
                    elif redundancy < percentiles[1]:  # low-mid%
                        divider = dividers[1]
                    elif redundancy < percentiles[2]:  # mid-high%
                        divider = dividers[2]
                    else:  # high-100%
                        divider = dividers[3]  # Highest divider (smallest stride)
                    
                    # Ensure divider is valid for base_weight
                    if self.base_weight < divider:
                        divider = self.base_weight
                    elif self.base_weight % divider != 0:
                        # Find closest valid divider
                        for d in sorted(dividers):
                            if self.base_weight % d == 0 and d <= divider:
                                divider = d
                                break
                        else:
                            divider = 1
                            
                    # Set divider in map (ensure it's an integer)
                    self.divider_map[t, l] = int(divider)
        
        logger.info(f"Divider map generation complete")
        
        # Return percentiles for visualization
        return percentiles

    def _refine_divider_map(self):
        """
        Refine the divider map to ensure valid transitions between timesteps.
        Valid transitions:
        1. Large divider to smaller divider (e.g., 8->4, 4->2, 2->1)
        2. 1 to any other divider (1->2, 1->4, 1->8)
        
        For invalid transitions, the first invalid value is changed to 1 to make the transition legal.
        """
        if self.divider_map is None:
            logger.warning("Cannot refine divider map: map not initialized")
            return
        
        # Make a copy of the map for checking changes
        original_map = self.divider_map.clone()
        
        # Process each column (layer) independently
        for layer in range(self.num_layers):
            for t in range(1, self.num_timesteps):
                current_val = self.divider_map[t, layer].item()
                prev_val = self.divider_map[t-1, layer].item()
                if t < 3 or t > self.num_timesteps - 3:
                    self.divider_map[t, layer] = 1
                    continue
                
                # Check if transition is valid
                # Case 1: Previous value is 1, can transition to any value
                if prev_val == 1:
                    # Valid transition, do nothing
                    continue
                
                # Case 2: Transition from large to small divider (decreasing stride)
                if current_val <= prev_val:
                    # Valid transition, do nothing
                    continue
                    
                # Case 3: Invalid transition (small to large divider, increasing stride)
                # Change current value to 1 to make transition legal
                self.divider_map[t, layer] = 1
        
        # Count changes made
        changes = (original_map != self.divider_map).sum().item()
        if changes > 0:
            logger.info(f"Refined divider map: {changes} values changed to ensure valid transitions")
        
        return changes
    
    def visualize_redundancy_map(self, output_dir, file_prefix="redundancy"):
        """
        Generate and save visualization for redundancy map
        
        Args:
            output_dir: Directory to save the redundancy map visualization
            file_prefix: Prefix for the saved file
            
        Returns:
            Path to the generated visualization file
        """
        if self.redundancy_map is None:
            logger.warning("Cannot visualize redundancy map: no data available")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better visualization
        plt.style.use('ggplot')
        
        # Create file path without timestamp
        redundancy_file = os.path.join(output_dir, f"{file_prefix}_map.png")
        
        # Convert to numpy array
        redundancy_data = self.redundancy_map.cpu().numpy()
        
        # Collect all non-zero redundancy values
        non_zero_values = redundancy_data[redundancy_data > 0]
        
        if len(non_zero_values) > 0:
            fig, ax = plt.subplots(figsize=(14, 10), dpi=120)
            
            # Create custom colormap with special handling for zero values
            cmap = plt.cm.viridis.copy()
            cmap.set_under('#f0f0f0')  # Light gray for values below minimum (zero)
            
            # Set non-zero minimum and maximum values
            vmin = 0.00001
            vmax = max(0.3, np.percentile(non_zero_values, 95))  # Use 95th percentile for better contrast
            
            # Draw heatmap
            im = ax.imshow(redundancy_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            
            # Add colorbar
            cbar = fig.colorbar(im, pad=0.02, extend='both')
            cbar.set_label('Redundancy (L1 Distance)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
            # Set axis labels and title
            ax.set_xlabel('Layer ID', fontsize=12)
            ax.set_ylabel('Timestep', fontsize=12)
            ax.set_title('Redundancy Map', fontsize=14, fontweight='bold')
            
            # Add grid lines
            ax.grid(False)  # Remove default grid
            
            # Add timestep and layer ticks
            # For larger dimensions, only show a subset of ticks
            timesteps = range(self.num_timesteps)
            layers = range(self.num_layers)
            
            if self.num_timesteps > 10:
                step = max(1, self.num_timesteps // 10)
                ax.set_yticks(range(0, self.num_timesteps, step))
                ax.set_yticklabels([f"{t}" for t in timesteps[::step]])
            else:
                ax.set_yticks(range(self.num_timesteps))
                ax.set_yticklabels([f"{t}" for t in timesteps])
                
            if self.num_layers > 10:
                step = max(1, self.num_layers // 10)
                ax.set_xticks(range(0, self.num_layers, step))
                ax.set_xticklabels([f"L{l}" for l in layers[::step]])
            else:
                ax.set_xticks(range(self.num_layers))
                ax.set_xticklabels([f"L{l}" for l in layers])
            
            plt.tight_layout()
            plt.savefig(redundancy_file)
            plt.close(fig)
            
            logger.info(f"Redundancy map visualization saved to {redundancy_file}")
            return redundancy_file
        
        return None

    def visualize_divider_map(self, output_dir=None, file_prefix="stride"):
        """
        Generate and save visualization for divider map with timestamps
        
        Args:
            output_dir: Directory to save visualization files
            file_prefix: Prefix for saved files
            
        Returns:
            Dictionary with paths to generated visualization files
        """
        # Ensure output directory exists
        if output_dir is None:
            output_dir = "./result/"
        file_prefix += f'_{get_config().tag}'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        visualization_files = {}
        
        # Set better style
        plt.style.use('ggplot')
        
        # Generate divider map visualization with improved aesthetics
        divider_file = os.path.join(output_dir, f"{file_prefix}_{timestamp}_divider.png")
        
        # Convert to numpy array (ensure it's integer type)
        divider_data = self.divider_map.cpu().numpy().astype(np.int64)
        
        # Create a figure with better proportions
        fig, ax = plt.subplots(figsize=(14, 10), dpi=120)
        
        # Create discrete colormap where each divider gets a unique color
        # Using a more visually pleasing colormap
        unique_dividers = np.unique(divider_data)
        unique_dividers = np.sort(unique_dividers)
        
        # Create color mapping using a better color scheme
        # Using viridis colormap for better color distinction
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_dividers)))
        cmap = mpl.colors.ListedColormap(colors)
        
        # Create normalized mapping for visualization
        divider_to_index = {divider: i for i, divider in enumerate(unique_dividers)}
        normalized_data = np.zeros_like(divider_data)
        for i in range(self.num_timesteps):
            for j in range(self.num_layers):
                normalized_data[i,j] = divider_to_index[divider_data[i,j]]
        
        # Draw heatmap
        im = ax.imshow(normalized_data, cmap=cmap, aspect='auto')
        
        # Create a single colorbar with both divider and stride information
        if all(self.base_weight % d == 0 for d in unique_dividers):
            # Create better labels that include both divider and stride
            cbar_labels = [f"Divider={d} (Stride={self.base_weight//d})" for d in unique_dividers]
            
            # Add a single, more informative colorbar
            cbar = fig.colorbar(im, ticks=range(len(unique_dividers)), pad=0.02)
            cbar.ax.set_yticklabels(cbar_labels)
            cbar.set_label('Communication Configuration', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
        else:
            # Fallback to simple divider colorbar
            cbar = fig.colorbar(im, ticks=range(len(unique_dividers)), pad=0.02)
            cbar.set_label('Divider Value', fontsize=12)
            cbar.ax.set_yticklabels([f"{d}" for d in unique_dividers])
        
        # Set axis labels and title
        ax.set_xlabel('Layer ID', fontsize=12)
        ax.set_ylabel('Timestep', fontsize=12)
        ax.set_title('Communication Strategy Map', fontsize=14, fontweight='bold')
        
        # Remove grid
        ax.grid(False)
        
        # Add timestep and layer ticks
        # For larger dimensions, only show a subset of ticks
        timesteps = range(self.num_timesteps)
        layers = range(self.num_layers)
        
        if self.num_timesteps > 10:
            step = max(1, self.num_timesteps // 10)
            ax.set_yticks(range(0, self.num_timesteps, step))
            ax.set_yticklabels([f"{t}" for t in timesteps[::step]])
        else:
            ax.set_yticks(range(self.num_timesteps))
            ax.set_yticklabels([f"{t}" for t in timesteps])
            
        if self.num_layers > 10:
            step = max(1, self.num_layers // 10)
            ax.set_xticks(range(0, self.num_layers, step))
            ax.set_xticklabels([f"L{l}" for l in layers[::step]])
        else:
            ax.set_xticks(range(self.num_layers))
            ax.set_xticklabels([f"L{l}" for l in layers])
        
        # Add a subtle grid to help with reading
        for i in range(1, self.num_timesteps):
            if i % 5 == 0:  # Only draw grid line every 5 steps
                ax.axhline(i - 0.5, color='white', linewidth=0.8, alpha=0.5)
                
        for j in range(1, self.num_layers):
            if j % 5 == 0:  # Only draw grid line every 5 layers
                ax.axvline(j - 0.5, color='white', linewidth=0.8, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(divider_file)
        plt.close(fig)
        
        visualization_files['divider_map'] = divider_file
        logger.info(f"Divider map visualization saved to {divider_file}")
        
        # Generate information text file
        info_file = os.path.join(output_dir, f"{file_prefix}_{timestamp}_info.txt")
        
        with open(info_file, 'w') as f:
            f.write("=== StrideMap Configuration ===\n\n")
            f.write(f"Number of timesteps: {self.num_timesteps}\n")
            f.write(f"Number of layers: {self.num_layers}\n")
            f.write(f"AutoSet round: {self.calc_round}\n")
            f.write(f"Base weight (osp size): {self.base_weight}\n\n")
            
            f.write("Stride divider options: " + str(self.stride_divider) + "\n")
            f.write("Redundancy percentile thresholds: " + str(self.map_percentage) + "%\n\n")
            
            f.write("=== Divider Distribution ===\n\n")
            unique, counts = np.unique(divider_data, return_counts=True)
            total_cells = self.num_timesteps * self.num_layers
            
            f.write("Divider | Count | Percentage | Stride\n")
            f.write("--------|-------|------------|-------\n")
            for div, count in zip(unique, counts):
                percentage = 100 * count / total_cells
                div_int = int(div)  # Ensure div is integer
                stride = self.base_weight // div_int if self.base_weight % div_int == 0 else "N/A"
                f.write(f"{div_int:7d} | {count:5d} | {percentage:8.2f}% | {stride}\n")
            
            if self.redundancy_map is not None:
                # Add redundancy to divider mapping information
                f.write("\n=== Redundancy to Divider Mapping ===\n\n")
                
                # Calculate percentiles
                non_zero_values = []
                for t in range(self.num_timesteps):
                    for l in range(self.num_layers):
                        redundancy = self.redundancy_map[t, l].item()
                        if redundancy > 0:
                            non_zero_values.append(redundancy)
                
                if len(non_zero_values) > 0:
                    non_zero_values = np.array(non_zero_values)
                    percentiles = [np.percentile(non_zero_values, p) for p in self.map_percentage]
                    
                    f.write(f"Redundancy Range | Percentile | Assigned Divider | Resulting Stride\n")
                    f.write(f"----------------|------------|-----------------|----------------\n")
                    
                    # 动态生成冗余度与分割器映射的说明
                    # 处理第一个区间 (0 到第一个百分位)
                    stride = self.base_weight // self.stride_divider[0] if self.base_weight % self.stride_divider[0] == 0 else "N/A"
                    f.write(f"0.0 - {percentiles[0]:.6f} | 0-{self.map_percentage[0]}% | {self.stride_divider[0]:17d} | {stride}\n")
                    
                    # 处理中间区间
                    for i in range(len(self.map_percentage)-1):
                        stride = self.base_weight // self.stride_divider[i+1] if self.base_weight % self.stride_divider[i+1] == 0 else "N/A"
                        f.write(f"{percentiles[i]:.6f} - {percentiles[i+1]:.6f} | {self.map_percentage[i]}-{self.map_percentage[i+1]}% | {self.stride_divider[i+1]:17d} | {stride}\n")
                    
                    # 处理最后一个区间 (最后一个百分位到无穷大)
                    stride = self.base_weight // self.stride_divider[-1] if self.base_weight % self.stride_divider[-1] == 0 else "N/A"
                    f.write(f"{percentiles[-1]:.6f} - inf | {self.map_percentage[-1]}-100% | {self.stride_divider[-1]:17d} | {stride}\n")
        
        visualization_files['info'] = info_file
        logger.info(f"StrideMap information saved to {info_file}")

        return visualization_files

    def finished_one_round(self):
        """Mark completion of one processing round"""
        self.calc_round += 1
        self.prev_out = [None] * self.num_layers
        if torch.distributed.get_rank() in [0, 7]:
            logger.debug(f"{self.redundancy_map=}")
        logger.info(f"Finished StrideMap auto-setting round {self.calc_round}")
    
    def finish_auto_setting(self):
        """
        Disable auto setting and collect redundancy data.
        All processes average their redundancy maps using all_reduce.
        Does NOT generate divider map automatically - this is now done separately.
        """
        if not self.auto_setting:
            logger.info("Auto setting is not enabled, nothing to finish")
            return
            
        # All-reduce to average the redundancy maps from all processes
        device = torch.device(f"cuda:{get_config().local_rank}")
        redun = self.redundancy_map.to(device)
        if torch.distributed.is_initialized() and self.redundancy_map is not None:
            # Make a copy of the redundancy map for all_reduce
            # No need to convert to float since redundancy_map is already float type
            torch.distributed.all_reduce(redun, op=torch.distributed.ReduceOp.AVG)
            logger.info(f"All-reduced redundancy map from all processes")
            self.redundancy_map = redun.to(self.redundancy_map.device)
        
        # Only rank 0 generates visualizations and saves the redundancy map
        if torch.distributed.get_rank() == 0:
            # Save the redundancy map to disk for future use
            save_path = save_redundancy_map(override=True)
            logger.info(f"Auto-setting complete for round {self.calc_round}, redundancy map saved to {save_path}")
        
        # Clean up and disable auto-setting
        self.auto_setting = False
        self.prev_out = [None] * self.num_layers
    
def print_stride_map(logger=None):
    """
    Print total stride map
    
    Args:
        logger: logger to print info
    """
    # Set numpy print options
    np.set_printoptions(threshold=np.inf, linewidth=1000)
    
    stride_map = get_stride_map()
    if stride_map is None:
        print("Stride map not initialized", flush=True)
        return
        
    divider_map = stride_map.divider_map
    world_size = get_usp_group().world_size
    base_weight_map = torch.ones_like(divider_map) * world_size
    # 确保 divider_map 不会超过 base_weight_map
    clamped_divider_map = torch.clamp_max(divider_map, world_size)
    stride_values = base_weight_map // clamped_divider_map
    
    # Format the output as a readable text grid
    formatted_output = '\n'.join([' '.join([f"{int(num):2d}" for num in row]) for row in stride_values.cpu().numpy()])
    
    # Only rank 0 prints the map
    if torch.distributed.get_rank() == 0:
        if logger:
            logger.info(f"===== DiTango Stride Map =====\n{formatted_output}\n=================================")
        else:
            print(f"===== DiTango Stride Map =====\n{formatted_output}\n=================================", flush=True)

_stride_map: Optional[StrideMap] = None

def init_stride_map(args, redun_map_path=None, custom_percentages=None, custom_dividers=None):
    """
    Initialize global stride map
    
    Args:
        args: Config object containing parameters
        redun_map_path: Optional path to load a pre-computed redundancy map
        custom_percentages: Optional custom percentile thresholds [low, mid, high]
        custom_dividers: Optional custom divider values [lowest, low, mid, high]
        
    Returns:
        Initialized StrideMap instance
    """
    global _stride_map
    assert _stride_map is None, ("Stride map is already initialized")
    
    # Initialize StrideMap with config
    stride_map = StrideMap(args)
    _stride_map = stride_map
    
    # Use config values if custom parameters not provided
    if custom_percentages is None and hasattr(args, 'stride_percentiles'):
        custom_percentages = args.stride_percentiles
    
    if custom_dividers is None and hasattr(args, 'stride_dividers'):
        custom_dividers = args.stride_dividers
    
    if args.use_easy_cache:
        print_stride_map()
        return stride_map
    
    # Try to load pre-computed redundancy map if path is provided or check default location
    if redun_map_path is None:
        # Check default location
        default_path = os.path.join(f"{args.ditango_base}/configs/{args.model_name}/", 
                                  f"redundancy_map_{args.num_inference_steps}_{args.num_layers}.pt")
        if os.path.exists(default_path):
            redun_map_path = default_path
        else:
            logger.warning(f"No pre-computed redundancy map found at default location: {default_path}, Using full stride map instead")
    
    if redun_map_path is not None and os.path.exists(redun_map_path):
        # Only rank 0 prints the loading message
        if torch.distributed.get_rank() == 0:
            logger.info(f"Loading pre-computed redundancy map from {redun_map_path}")
        
        try:
            # Load the redundancy map tensor
            loaded_map = torch.load(redun_map_path, map_location='cpu')
            
            # Verify the loaded map has correct dimensions
            if (loaded_map.shape[0] == args.num_inference_steps and 
                loaded_map.shape[1] == args.num_layers):
                # Set the redundancy map
                stride_map.redundancy_map = loaded_map
                
                # Generate divider map with parameters from config or custom parameters
                percentiles = stride_map.generate_divider_map_from_redundancy(
                    custom_percentages=custom_percentages, 
                    custom_dividers=custom_dividers
                )
                
                # Refine divider map to ensure valid transitions
                stride_map._refine_divider_map()
                
                # Only rank 0 prints success message and visualization
                if torch.distributed.get_rank() == 0:
                    logger.info(f"Successfully loaded redundancy map with shape {loaded_map.shape}")
                    logger.info(f"Generated divider map with" + 
                             (f" percentages {custom_percentages}" if custom_percentages else " default percentages") +
                             (f" and dividers {custom_dividers}" if custom_dividers else " and default dividers"))
                    
                    # Generate visualizations for divider map in logs directory
                    stride_map.visualize_divider_map(output_dir=args.output_dir)
            else:
                # If dimensions don't match, warn and use default
                if torch.distributed.get_rank() == 0:
                    logger.warning(f"Loaded map dimensions {loaded_map.shape} don't match "
                                 f"required dimensions ({args.num_inference_steps}, {args.num_layers}). "
                                 f"Using default map instead.")
        except Exception as e:
            # If loading fails, warn and use default
            if torch.distributed.get_rank() == 0:
                logger.warning(f"Failed to load redundancy map: {str(e)}. Using default map instead.")
                
    print_stride_map()
    return stride_map

def save_redundancy_map(path=None, override=False):
    """
    Save current redundancy map to disk and generate visualization
    
    Args:
        path: Path to save the redundancy map. If None, uses default path.
        override: Whether to override existing file if it exists
        
    Returns:
        Path where map was saved, or None if saving failed
    """
    args = get_config()
    stride_map = get_stride_map()
    if stride_map is None or stride_map.redundancy_map is None:
        logger.warning("Cannot save redundancy map: stride map not initialized or no redundancy data available")
        return None
    
    # Only rank 0 needs to save the map
    if torch.distributed.get_rank() != 0:
        return None
    
    # Generate default path if not provided
    if path is None:
        # Create directory if it doesn't exist
        config_dir = f"{args.ditango_base}/configs/{args.model_name}/"
        os.makedirs(config_dir, exist_ok=True)
        path = os.path.join(config_dir, 
                          f"redundancy_map_{stride_map.num_timesteps}_{stride_map.num_layers}.pt")
    
    # Check if file exists and handle override
    if os.path.exists(path) and not override:
        logger.warning(f"File {path} already exists. Use override=True to replace it.")
        return None
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the redundancy map
        torch.save(stride_map.redundancy_map, path)
        logger.info(f"Redundancy map saved to {path}")
        
        # Generate visualization in the same directory as the redundancy map
        vis_path = stride_map.visualize_redundancy_map(os.path.dirname(path))
        if vis_path:
            logger.info(f"Redundancy map visualization saved to {vis_path}")
            
        return path
    except Exception as e:
        logger.error(f"Failed to save redundancy map: {str(e)}")
        return None

def save_divider_map(path=None, override=False):
    """
    Save current divider map to disk (kept for backward compatibility)
    
    Args:
        path: Path to save the divider map. If None, uses default path.
        override: Whether to override existing file if it exists
        
    Returns:
        Path where map was saved, or None if saving failed
    """
    args = get_config()
    stride_map = get_stride_map()
    if stride_map is None:
        logger.warning("Cannot save divider map: stride map not initialized")
        return None
    
    # Only rank 0 needs to save the map
    if torch.distributed.get_rank() != 0:
        return None
    
    # Generate default path if not provided
    if path is None:
        # Create directory if it doesn't exist
        os.makedirs(f"{args.ditango_base}/configs/{args.model_name}/", exist_ok=True)
        path = os.path.join(f"{args.ditango_base}/configs/{args.model_name}/", 
                          f"divider_map_{stride_map.num_timesteps}_{stride_map.num_layers}.pt")
    
    # Check if file exists and handle override
    if os.path.exists(path) and not override:
        logger.warning(f"File {path} already exists. Use override=True to replace it.")
        return None
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the divider map
        torch.save(stride_map.divider_map, path)
        logger.info(f"Divider map saved to {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save divider map: {str(e)}")
        return None
  
def get_stride_map() -> StrideMap:
    """
    Get global stride map instance
    
    Returns:
        Global StrideMap instance or None if not initialized
    """
    if _stride_map is None:
        return None
    return _stride_map

def is_preprocessing():
    if _stride_map is None:
        return False
    return _stride_map.auto_setting

def preprocess_for_stridemap(func, func_args, prompt_list):
    """
    Preprocess for generating stride map with auto-setting.
    Only rank 0 will set the map, then broadcast to all processes.
    
    Args:
        pipe: The generation pipeline to use for collecting redundancy data
    """
    if prompt_list is None:
        prompt_list = ["A kitten wearing a red bow tie is dancing, the camera slowly zooms in from a distance"]
    config = get_config()
    # Display start message on rank 0
    if config.rank == 0:    
        logger.info(f"====================== Start StrideMap AutoSetting! round={len(prompt_list)} ==============")
    
    # Enable auto setting on all processes (only rank 0 will collect data)
    get_stride_map().enable_auto_setting()
    
    # Process each prompt to collect redundancy data
    for prompt in prompt_list:
        if config.local_rank == 0:
            print(f"prompt: {prompt}\n Preprocessing...", flush=True)
            
        # Generate video to collect redundancy data
        video = func(prompt=prompt, **func_args)
        
        # Mark round as finished
        get_stride_map().finished_one_round()
    
    # Finish auto setting - rank 0 will process data and broadcast results
    get_stride_map().finish_auto_setting()
    
    # Print final stride map (only on rank 0)
    if config.rank == 0:    
        print_stride_map()

# Function to generate and save divider map with custom settings
def generate_divider_map_with_custom_settings(custom_percentages=None, custom_dividers=None, output_dir=None):
    """
    Generate a new divider map with custom settings from an existing redundancy map
    
    Args:
        custom_percentages: Custom percentile thresholds [low, mid, high]
        custom_dividers: Custom divider values [lowest, low, mid, high]
        output_dir: Directory to save visualization files
    
    Returns:
        Paths to generated visualization files
    """
    stride_map = get_stride_map()
    if stride_map is None or stride_map.redundancy_map is None:
        logger.warning("Cannot generate custom divider map: stride map not initialized or no redundancy data")
        return None
    
    # Only rank 0 generates the divider map
    if torch.distributed.get_rank() != 0:
        return None
    
    config = get_config()
    # Use config values if custom values are not provided
    if custom_percentages is None:
        custom_percentages = config.stride_percentiles
    if custom_dividers is None:
        custom_dividers = config.stride_dividers
    
    # Set output directory to logs if not specified
    if output_dir is None:
        output_dir = config.output_dir
    
    # Generate divider map with custom parameters
    stride_map.generate_divider_map_from_redundancy(
        custom_percentages=custom_percentages,
        custom_dividers=custom_dividers
    )
    
    # Refine the divider map
    stride_map._refine_divider_map()
    
    # Generate visualizations with custom prefix to distinguish them
    percentages_str = "_".join([str(p) for p in custom_percentages]) if custom_percentages else "default"
    dividers_str = "_".join([str(d) for d in custom_dividers]) if custom_dividers else "default"
    file_prefix = f"stride_custom_p{percentages_str}_d{dividers_str}"
    
    # Generate and save visualizations
    if config.use_ringfusion:
        visualization_files = stride_map.visualize_divider_map(output_dir=output_dir, file_prefix=file_prefix)
    
    logger.info(f"Generated custom divider map with percentages {custom_percentages if custom_percentages else 'default'} and dividers {custom_dividers if custom_dividers else 'default'}")
    
    return visualization_files