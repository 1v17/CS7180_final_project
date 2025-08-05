"""
Memory Configuration for Ebbinghaus Memory System

This module provides configuration settings for the Ebbinghaus memory system,
allowing easy switching between modes and adjustment of parameters.
"""

from typing import Dict, Any


class MemoryConfig:
    """Configuration manager for Ebbinghaus memory system."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        "memory_mode": "standard",  # "standard" or "ebbinghaus"
        "forgetting_curve": {
            "enabled": False,               # Controlled by memory_mode
            "initial_strength": 1.0,        # Base strength for new memories
            "min_retention_threshold": 0.1, # Minimum strength to keep memory
            "retrieval_boost": 0.5,         # Strength increase on retrieval
            "decay_rate": 0.5,             # Base decay rate
            "soft_delete": True,           # Archive vs delete weak memories
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
            }
        }
    }
    
    # Testing configuration (faster decay for testing)
    TESTING_CONFIG = {
        "memory_mode": "ebbinghaus",
        "forgetting_curve": {
            "enabled": True,
            "initial_strength": 1.0,
            "min_retention_threshold": 0.2,
            "retrieval_boost": 0.3,
            "decay_rate": 0.8,  # Faster decay for testing
            "soft_delete": True,
        },
        "llm": {
            "provider": "openai", 
            "config": {
                "model": "gpt-4o-mini",
            }
        }
    }
    
    # Production configuration (slower decay)
    PRODUCTION_CONFIG = {
        "memory_mode": "ebbinghaus",
        "forgetting_curve": {
            "enabled": True,
            "initial_strength": 1.0,
            "min_retention_threshold": 0.1,
            "retrieval_boost": 0.5,
            "decay_rate": 0.3,  # Slower decay for production
            "soft_delete": True,
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
            }
        }
    }
    
    @classmethod
    def get_config(cls, mode: str = "default") -> Dict[str, Any]:
        """
        Get configuration for specified mode.
        
        Args:
            mode (str): Configuration mode - "default", "testing", or "production"
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if mode == "testing":
            return cls.TESTING_CONFIG.copy()
        elif mode == "production":
            return cls.PRODUCTION_CONFIG.copy()
        else:
            return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def create_standard_config(cls) -> Dict[str, Any]:
        """Create configuration for standard (perfect) memory mode."""
        config = cls.DEFAULT_CONFIG.copy()
        config["memory_mode"] = "standard" 
        config["forgetting_curve"]["enabled"] = False
        return config
    
    @classmethod
    def create_ebbinghaus_config(cls,
                                decay_rate: float = 0.5,
                                min_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Create configuration for Ebbinghaus memory mode.
        
        Args:
            decay_rate (float): Rate of memory decay
            min_threshold (float): Minimum retention to keep memory
            
        Returns:
            Dict[str, Any]: Ebbinghaus configuration
        """
        config = cls.DEFAULT_CONFIG.copy()
        config["memory_mode"] = "ebbinghaus"
        config["forgetting_curve"]["enabled"] = True
        config["forgetting_curve"]["decay_rate"] = decay_rate
        config["forgetting_curve"]["min_retention_threshold"] = min_threshold
        return config
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required keys
            if "memory_mode" not in config:
                print("Error: memory_mode is required")
                return False
            
            if config["memory_mode"] not in ["standard", "ebbinghaus"]:
                print("Error: memory_mode must be 'standard' or 'ebbinghaus'")
                return False
            
            # Check forgetting curve parameters if in ebbinghaus mode
            if config["memory_mode"] == "ebbinghaus" and "forgetting_curve" in config:
                fc_config = config["forgetting_curve"]
                
                # Validate numeric parameters
                numeric_params = {
                    "initial_strength": (0.0, 1.0),
                    "min_retention_threshold": (0.0, 1.0),
                    "retrieval_boost": (0.0, 1.0),
                    "decay_rate": (0.0, 2.0)
                }
                
                for param, (min_val, max_val) in numeric_params.items():
                    if param in fc_config:
                        value = fc_config[param]
                        if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                            print(f"Error: {param} must be between {min_val} and {max_val}")
                            return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False
