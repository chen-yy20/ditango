import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Optional, Dict, Tuple
from contextlib import contextmanager

import torch.distributed
from ..core.parallel_state import get_usp_group
from ..utils import rel_l1, get_timestep
from ..logger import init_logger
from ..core.config import get_config

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
        return self.base_weight // divider
    
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
            return
        
        # Calculate relative L1 distance
        rel_l1_distance = rel_l1(curr_out, prev_out)
        self.redundancy_map[timestep, layer_id] += rel_l1_distance
        self.prev_out[layer_id] = curr_out
        
    def _auto_set_divider_map(self):
        """
        Automatically set divider map based on recorded redundancy
        - Higher redundancy -> higher divider (smaller stride)
        - Redundancy ranges are determined by map_percentage percentiles
        - Only non-zero redundancy values are considered
        
        Returns:
            percentiles: The calculated percentile values used for mapping
        """
        if not self.auto_setting or self.redundancy_map is None:
            logger.warning("Cannot auto-set divider map: auto_setting is not enabled or no redundancy data")
            return

        # Collect all non-zero redundancy values
        non_zero_values = []
        for t in range(self.num_timesteps):
            for l in range(self.num_layers):
                redundancy = self.redundancy_map[t, l].item()
                if redundancy > 0:
                    non_zero_values.append(redundancy)
        
        if len(non_zero_values) == 0:
            logger.warning("No non-zero redundancy values found, keeping default divider map")
            return
        
        # Calculate percentiles for redundancy distribution
        non_zero_values = np.array(non_zero_values)
        percentiles = [np.percentile(non_zero_values, p) for p in self.map_percentage]
        
        logger.info(f"Auto-setting divider map with percentiles: {percentiles}")
        logger.info(f"Corresponding to percentages: {self.map_percentage}")
        logger.info(f"Using dividers: {self.stride_divider}")
        
        # Reset divider map to base values
        self.divider_map = self._init_map()
        
        # Set divider based on redundancy value
        for t in range(self.num_timesteps):
            for l in range(self.num_layers):
                redundancy = self.redundancy_map[t, l].item()
                
                # Only process non-zero redundancy values
                if redundancy > 0:
                    # Map redundancy to appropriate divider
                    if redundancy < percentiles[0]:  # 0-20%
                        divider = self.stride_divider[0]  # Lowest divider (largest stride)
                    elif redundancy < percentiles[1]:  # 20-50%
                        divider = self.stride_divider[1]
                    elif redundancy < percentiles[2]:  # 50-80%
                        divider = self.stride_divider[2]
                    else:  # 80-100%
                        divider = self.stride_divider[3]  # Highest divider (smallest stride)
                    
                    # Ensure divider is valid for base_weight
                    if self.base_weight < divider:
                        divider = 1
                    elif self.base_weight % divider != 0:
                        # Find closest valid divider
                        for d in sorted(self.stride_divider):
                            if self.base_weight % d == 0 and d <= divider:
                                divider = d
                                break
                        else:
                            divider = 1
                            
                    # Set divider in map (ensure it's an integer)
                    self.divider_map[t, l] = int(divider)
        
        logger.info(f"Auto-set divider map complete")
        
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

    def visualize_maps(self, output_dir="./stride_visualizations", file_prefix="stride"):
        """
        Generate and save visualizations of redundancy and divider maps with improved aesthetics
        
        Args:
            output_dir: Directory to save visualization files
            file_prefix: Prefix for saved files
            
        Returns:
            Dictionary with paths to generated visualization files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        visualization_files = {}
        
        # Set better style
        plt.style.use('ggplot')
        
        # 1. Generate redundancy map visualization if available
        if self.redundancy_map is not None:
            redundancy_file = os.path.join(output_dir, f"{file_prefix}_{timestamp}_redundancy.png")
            
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
                
                visualization_files['redundancy_map'] = redundancy_file
                logger.info(f"Redundancy map visualization saved to {redundancy_file}")
        
        # 2. Generate divider map visualization with improved aesthetics
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
        
        # 3. Generate information text file (unchanged)
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
                    
                    # 0-20%
                    stride = self.base_weight // self.stride_divider[0] if self.base_weight % self.stride_divider[0] == 0 else "N/A"
                    f.write(f"0.0 - {percentiles[0]:.6f} | 0-{self.map_percentage[0]}% | {self.stride_divider[0]:17d} | {stride}\n")
                    
                    # 20-50%
                    stride = self.base_weight // self.stride_divider[1] if self.base_weight % self.stride_divider[1] == 0 else "N/A"
                    f.write(f"{percentiles[0]:.6f} - {percentiles[1]:.6f} | {self.map_percentage[0]}-{self.map_percentage[1]}% | {self.stride_divider[1]:17d} | {stride}\n")
                    
                    # 50-80%
                    stride = self.base_weight // self.stride_divider[2] if self.base_weight % self.stride_divider[2] == 0 else "N/A"
                    f.write(f"{percentiles[1]:.6f} - {percentiles[2]:.6f} | {self.map_percentage[1]}-{self.map_percentage[2]}% | {self.stride_divider[2]:17d} | {stride}\n")
                    
                    # 80-100%
                    stride = self.base_weight // self.stride_divider[3] if self.base_weight % self.stride_divider[3] == 0 else "N/A"
                    f.write(f"{percentiles[2]:.6f} - ∞ | {self.map_percentage[2]}-100% | {self.stride_divider[3]:17d} | {stride}\n")
        
        visualization_files['info'] = info_file
        logger.info(f"StrideMap information saved to {info_file}")
    
        return visualization_files

    def finished_one_round(self):
        """Mark completion of one processing round"""
        self.calc_round += 1
        logger.info(f"Finished StrideMap auto-setting round {self.calc_round}")
    
    def finish_auto_setting(self):
        """
        Disable auto setting and apply collected redundancy data to set divider map.
        All processes average their redundancy maps using all_reduce,
        then each process generates the same divider map based on the averaged redundancy.
        Visualization is only done by rank 0.
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
        
        # Now each process generates divider map based on the same redundancy data
        logger.info("Generating divider map from collected redundancy data")
        percentiles = self._auto_set_divider_map()
        self._refine_divider_map()
        
        # Only rank 0 generates visualizations and saves the map
        if torch.distributed.get_rank() == 0:
            # Generate visualizations
            self.visualize_maps()
            # Save the divider map to disk for future use
            save_path = save_divider_map(override=True)
            logger.info(f"Auto-setting complete for round {self.calc_round}" + 
                    (f" with percentiles: {percentiles}" if percentiles is not None else "") +
                    (f", map saved to {save_path}" if save_path else ""))
        
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
    base_weight_map = torch.ones_like(divider_map) * get_usp_group().world_size
    stride_values = base_weight_map // divider_map
    
    # Format the output as a readable text grid
    formatted_output = '\n'.join([' '.join([f"{int(num):2d}" for num in row]) for row in stride_values.cpu().numpy()])
    
    # Only rank 0 prints the map
    if torch.distributed.get_rank() == 0:
        if logger:
            logger.info(f"===== DiTango Stride Map =====\n{formatted_output}\n=================================")
        else:
            print(f"===== DiTango Stride Map =====\n{formatted_output}\n=================================", flush=True)

_stride_map: Optional[StrideMap] = None

def init_stride_map(args, load_map_path=None):
    """
    Initialize global stride map
    
    Args:
        args: Arguments containing num_inference_steps and num_layers
        load_map_path: Optional path to load a pre-computed divider map
        
    Returns:
        Initialized StrideMap instance
    """
    global _stride_map
    assert _stride_map is None, ("Stride map is already initialized")
    stride_map = StrideMap(args)
    _stride_map = stride_map
    
    # Try to load pre-computed divider map if path is provided or check default location
    if load_map_path is None:
        # Check default location
        default_path = os.path.join("./stride_maps", f"divider_map_{args.num_inference_steps}_{args.num_layers}.pt")
        if os.path.exists(default_path):
            load_map_path = default_path
    
    if load_map_path is not None and os.path.exists(load_map_path):
        # Only rank 0 prints the loading message
        if torch.distributed.get_rank() == 0:
            logger.info(f"Loading pre-computed divider map from {load_map_path}")
        
        try:
            # Load the divider map tensor
            loaded_map = torch.load(load_map_path, map_location='cpu')
            
            # Verify the loaded map has correct dimensions
            if (loaded_map.shape[0] == args.num_inference_steps and 
                loaded_map.shape[1] == args.num_layers):
                stride_map.divider_map = loaded_map
                
                # Only rank 0 prints success message
                if torch.distributed.get_rank() == 0:
                    logger.info(f"Successfully loaded divider map with shape {loaded_map.shape}")
                    print_stride_map()
            else:
                # If dimensions don't match, warn and use default
                if torch.distributed.get_rank() == 0:
                    logger.warning(f"Loaded map dimensions {loaded_map.shape} don't match "
                                 f"required dimensions ({args.num_inference_steps}, {args.num_layers}). "
                                 f"Using default map instead.")
        except Exception as e:
            # If loading fails, warn and use default
            if torch.distributed.get_rank() == 0:
                logger.warning(f"Failed to load divider map: {str(e)}. Using default map instead.")
    
    return stride_map

def save_divider_map(path=None, override=False):
    """
    Save current divider map to disk
    
    Args:
        path: Path to save the divider map. If None, uses default path.
        override: Whether to override existing file if it exists
        
    Returns:
        Path where map was saved, or None if saving failed
    """
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
        os.makedirs("./stride_maps", exist_ok=True)
        path = os.path.join("./stride_maps", 
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

prompt_list = [
  "A video of a cat playing with a ball",
  "A playful black Labrador, adorned in a vibrant pumpkin-themed Halloween costume, frolics in a sunlit autumn garden, surrounded by fallen leaves. The dog's costume features a bright orange body with a green leafy collar, perfectly complementing its shiny black fur. As it bounds joyfully across the lawn, the sunlight catches the costume's fabric, creating a delightful contrast with the dog's dark coat. The scene captures the essence of autumn festivities, with the dog's wagging tail and playful demeanor adding to the cheerful atmosphere. Nearby, carved pumpkins and scattered leaves enhance the festive setting.",
  "A charming boat glides gracefully along the serene Seine River, its sails catching a gentle breeze, while the iconic Eiffel Tower stands majestically in the background. The scene is rendered in rich, textured oil paints, capturing the warm hues of a late afternoon sun casting a golden glow over the water. The boat, with its elegant design and vibrant colors, contrasts beautifully with the soft, impressionistic strokes of the surrounding landscape. The Eiffel Tower, painted in delicate detail, rises above the Parisian skyline, its iron latticework shimmering in the light. The riverbanks are adorned with lush greenery and quaint buildings, their reflections dancing on the water's surface, creating a harmonious blend of nature and architecture. The overall composition exudes a sense of tranquility and timeless beauty, inviting viewers to immerse themselves in the idyllic Parisian scene.",
  # "A cat walks on the grass, realistic style.",
  # "A little girl is riding a bicycle at high speed. Focused, detailed, realistic.",
  # "Sun set over the sea"
]

def preprocess_for_stridemap(pipe):
    """
    Preprocess for generating stride map with auto-setting.
    Only rank 0 will set the map, then broadcast to all processes.
    
    Args:
        pipe: The generation pipeline to use for collecting redundancy data
    """
    config = get_config()
    height = 480
    width = 720
    frames = 48
    generator = torch.Generator().manual_seed(config.seed)
    
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
        video = pipe(
            height=height,
            width=width,
            num_frames=frames,
            prompt=prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=6,
            generator=generator,
        ).frames[0]
        
        # Mark round as finished
        get_stride_map().finished_one_round()
    
    # Finish auto setting - rank 0 will process data and broadcast results
    get_stride_map().finish_auto_setting()
    
    # Print final stride map (only on rank 0)
    if config.rank == 0:    
        print_stride_map()