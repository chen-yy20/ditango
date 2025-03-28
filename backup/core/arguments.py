# arguments.py
import os
import yaml
from pathlib import Path
import copy
import torch.distributed as dist

class DiTangoConfig:
    """Configuration class to manage DiTango system parameters"""
    def __init__(self, config_path=None):
        """
        Initialize configuration from file
        
        Args:
            config_path: Path to YAML config file
        """
        # Default configuration
        self.config = {
            # Basic parameters
            'model_name': 'cogvideox1.5-5b',
            'output_fn': 'output',
            'tag': 'test',
            
            # Generation parameters
            'num_layers': 42,
            'num_inference_steps': 50,
            'seed': 42,
            
            # Performance testing parameters
            'warmup': 0,
            'repeat': 2,
            'enable_timing': False,

            # Distributed parameters
            'gpus': None,
            'node': None,
            'do_cfg_parallel': False,
            
            # Baseline experiment parameters
            'use_ulysses': False,
            'use_distrifusion': False,
            
            # Cache and optimization parameters
            'use_easy_cache': False,
            'cache_threshold': None,
        }
        
        # Load from config file if provided
        if config_path is not None:
            self.load_from_file(config_path)
        
        # Process environment variables for distributed training
        self.process_env_vars()
        
    
    def load_from_file(self, config_path):
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Update config with file values (converting dashes to underscores)
                for key, value in file_config.items():
                    # Convert dash-style keys to underscore_style
                    updated_key = key.replace('-', '_')
                    self.config[updated_key] = value
        else:
            print(f"Warning: Config file {config_path} for DiTango not found. Using default values.")
    
    def process_env_vars(self):
        """Process environment variables for distributed settings"""
        self.config['world_size'] = int(os.getenv("WORLD_SIZE", "1"))
        self.config['rank'] = int(os.getenv("RANK", "0"))
        self.config['local_rank'] = int(os.getenv("LOCAL_RANK", "0"))
        self.config['tag'] = os.getenv("TAG", self.config['tag'])
    
    
    def __getattr__(self, name):
        """Allow attribute-style access to configuration values"""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'DiTangoConfig' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Allow attribute-style setting of configuration values"""
        if name == 'config':
            super().__setattr__(name, value)
        else:
            self.config[name] = value
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return copy.deepcopy(self.config)
    
    def save_to_file(self, file_path):
        """Save current configuration to YAML file"""
        with open(file_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# Global configuration object
DITANGO_CONFIG = None

def init_config(config_path):
    """
    Initialize global configuration object from config file
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        DiTangoConfig object
    """
    global DITANGO_CONFIG
    DITANGO_CONFIG = DiTangoConfig(config_path=config_path)
    return DITANGO_CONFIG

def get_config():
    """
    Get global configuration object
    
    Returns:
        DiTangoConfig object or None if not initialized
    """
    global DITANGO_CONFIG
    return DITANGO_CONFIG

def print_config():
    """
    Print all configuration parameters
    Only print on rank 0
    
    Args:
        config: DiTangoConfig object
    """
    config = get_config()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    if rank == 0:
        print("\n===== DiTango Configuration =====", flush=True)
        for key, value in config.to_dict().items():
            print(f"{key}: {value}", flush=True)
        print("\n=================================", flush=True)