import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union, Any
from ..logger import init_logger
from ..core.arguments import get_config
from ..core.parallel_state import get_usp_group
from ..utils import get_timestep

# 全局变量
REDUNDANCY_SENSOR = None
logger = init_logger(__name__)

def init_redundancy_sensor(
    model_name=None,
    isp_divider_thresholds=[0.1, 0.2, 0.3], 
):
    """
    Initialize the global redundancy sensor for adaptive caching.
    
    Args:
        model_name: Predefined model name ('cogvideox-5b', 'cogvideox-2b')
        coefficients: Custom coefficients (overrides model_name if provided)
        isp_divider_thresholds: List of thresholds for different computation fractions
                                    [threshold_for_1_8, threshold_for_2_8, threshold_for_4_8]
        history_window: Number of historical timesteps to store
        enable_adaptive_cache: Whether to enable adaptive caching optimization
        total_steps: Total number of inference steps (optional)
    
    Returns:
        The initialized redundancy sensor
    """
    global REDUNDANCY_SENSOR
    
    # Get model-specific coefficients if provided
    model_name = model_name.lower()
    if model_name in ['cogvideox-5b', 'cogvideox1.5-5b']:
        coefficients = [-1.53880483e+03, 8.43202495e+02, -1.34363087e+02, 7.97131516e+00, -5.23162339e-02]
    elif model_name == 'cogvideox-2b':
        coefficients = [-3.10658903e+01, 2.54732368e+01, -5.92380459e+00, 1.75769064e+00, -3.61568434e-03]
    else:
        logger.warning(f"Unknown model '{model_name}', using default coefficients")
    
    # Create the analyzer with the determined coefficients
    REDUNDANCY_SENSOR = AdaptiveCacheAnalyzer(
        coefficients=coefficients,
        isp_divider_thresholds=isp_divider_thresholds,
    )
    
    logger.info(f"Adaptive cache sensor initialized with model '{model_name}' and thresholds {isp_divider_thresholds}")
        
    return REDUNDANCY_SENSOR

def get_redundancy_sensor():
    """
    Get the global redundancy sensor.
    Initializes with default settings if not already initialized.
    
    Returns:
        The global redundancy sensor
    """
    global REDUNDANCY_SENSOR
    if REDUNDANCY_SENSOR is None:
        REDUNDANCY_SENSOR = AdaptiveCacheAnalyzer()
    return REDUNDANCY_SENSOR

def get_current_isp_divider():
    """
    Get the current ISP divider value.
    
    Returns:
        The current ISP divider value
    """
    return int(get_redundancy_sensor().current_isp_divider)

def print_sensor_history():
    """
    Print the ISP divider history.
    """
    sensor = get_redundancy_sensor()
    for i in range(len(sensor.timestep_history)):
        logger.info(f"T{sensor.timestep_history[i]} - ISP Divider: {sensor.divider_history[i]} ; Score: {sensor.score_history[i]}")

class AdaptiveCacheAnalyzer:
    """
    An adaptive caching and management system for diffusion models.
    Dynamically decides how much computation to perform based on redundancy analysis.
    """
    
    def __init__(
        self, 
        coefficients=None, 
        isp_divider_thresholds=[0.05, 0.1, 0.2],
    ):
        self.total_timesteps = get_config().num_inference_steps
        self.full_sp_size = get_usp_group().world_size
        self.warm_up = 3
        self.cool_down = 3
        
        # Default coefficients for CogVideoX1.5-5B 
        self.coefficients = coefficients or [-1.53880483e+03, 8.43202495e+02, -1.34363087e+02, 7.97131516e+00, -5.23162339e-02]
        self.rescale_func = np.poly1d(self.coefficients)

        self.accumulated_rel_l1_distance = 0.0
        self.current_isp_divider = 1
        
        self.isp_divider_thresholds = isp_divider_thresholds
        self.isp_divider_list = [8.0, 4.0, 2.0, 1.0]

        self.timestep_history = []
        self.divider_history = []
        self.score_history = []
        
        # Previous timestep's modulated input
        self.previous_emb = None

    
    def analyze_redundancy(self, emb):
        curr_timestep = get_timestep()
        if self.previous_emb is None:
            self.previous_emb = emb.detach().clone() if torch.is_tensor(emb) else emb
            
        if curr_timestep < self.warm_up or curr_timestep + 1 >= self.total_timesteps - self.cool_down:
            self.current_isp_divider = 1.0
            self.accumulated_rel_l1_distance = 0.0
            self.previous_emb = emb
            return
        
        # Calculate relative L1 distance
        if torch.is_tensor(emb) and torch.is_tensor(self.previous_emb):
            rel_l1 = ((emb - self.previous_emb).abs().mean() / 
                      self.previous_emb.abs().mean()).cpu().item()
        else:
            # Handle non-tensor case
            rel_l1 = abs(emb - self.previous_emb) / abs(self.previous_emb)
            
        # Apply non-linear transformation to get redundancy score
        percentage = max(1 - 2 / self.current_isp_divider, 0.5) # magic number
        self.accumulated_rel_l1_distance +=  max(self.rescale_func(rel_l1), 0.05) * percentage
        self.previous_emb = emb
        
        logger.info(f"T{curr_timestep} - Before DiT: prev_isp_divider = {self.current_isp_divider} ; acc_l1={self.accumulated_rel_l1_distance}")
        
        self.current_isp_divider = self._determine_isp_divider()

        self.timestep_history.append(curr_timestep)
        self.score_history.append(self.accumulated_rel_l1_distance)
        self.divider_history.append(self.current_isp_divider)
        
        if self.current_isp_divider == 1.0:
            self.accumulated_rel_l1_distance = 0.0

    
    def _determine_isp_divider(self):
        return 1.0
        if self.accumulated_rel_l1_distance < self.isp_divider_thresholds[0]:
            return self.isp_divider_list[0]
        elif self.accumulated_rel_l1_distance < self.isp_divider_thresholds[1]:
            return self.isp_divider_list[1]
        elif self.accumulated_rel_l1_distance < self.isp_divider_thresholds[2]:
            return self.isp_divider_list[2]
        else:
            return self.isp_divider_list[3]
        
        
    