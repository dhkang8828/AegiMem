"""
Configuration loader for CXL Memory RL Project
"""

import os
import yaml
from typing import Dict, Any


class Config:
    """Configuration class with dict-like access"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        self._config[key] = value
        setattr(self, key, value)

    def get(self, key, default=None):
        return self._config.get(key, default)

    def to_dict(self):
        """Convert config to dictionary"""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Config object
    """
    # Default config path
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        config_path = os.path.join(project_dir, "config", "default_config.yaml")

    # Load YAML file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def create_config_from_args(args) -> Config:
    """
    Create config from command line arguments

    Args:
        args: argparse arguments

    Returns:
        Config object
    """
    # Load base config from file if specified
    if hasattr(args, 'config') and args.config:
        config = load_config(args.config)
    else:
        config = load_config()  # Use default

    # Override with command line arguments
    if hasattr(args, 'episodes') and args.episodes:
        config.training.total_episodes = args.episodes

    if hasattr(args, 'safety_mode'):
        config.environment.safety_mode = args.safety_mode

    if hasattr(args, 'mbist_path') and args.mbist_path:
        config.environment.mbist_binary_path = args.mbist_path

    if hasattr(args, 'experiment_name') and args.experiment_name:
        config.output.experiment_name = args.experiment_name

    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.ppo.learning_rate = args.learning_rate

    if hasattr(args, 'batch_size') and args.batch_size:
        config.ppo.batch_size = args.batch_size

    return config


def save_config(config: Config, output_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Config object
        output_path: Path to save config
    """
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
