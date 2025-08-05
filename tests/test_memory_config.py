"""
Test suite for MemoryConfig class

This test file covers the MemoryConfig class functionality including
configuration creation, validation, and different configuration modes.
"""

import unittest
import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your classes
from memory_config import MemoryConfig


class TestMemoryConfig(unittest.TestCase):
    """Test suite for MemoryConfig class"""
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        config = MemoryConfig.get_config("default")
        
        self.assertEqual(config["memory_mode"], "standard")
        self.assertIn("forgetting_curve", config)
        self.assertEqual(config["forgetting_curve"]["enabled"], False)
        self.assertIn("llm", config)
    
    def test_get_testing_config(self):
        """Test getting testing configuration"""
        config = MemoryConfig.get_config("testing")
        
        self.assertEqual(config["memory_mode"], "ebbinghaus")
        self.assertEqual(config["forgetting_curve"]["enabled"], True)
    
    def test_get_production_config(self):
        """Test getting production configuration"""
        config = MemoryConfig.get_config("production")
        
        self.assertEqual(config["memory_mode"], "ebbinghaus")
        self.assertEqual(config["forgetting_curve"]["enabled"], True)
    
    def test_create_standard_config(self):
        """Test creating standard configuration"""
        config = MemoryConfig.create_standard_config()
        
        self.assertEqual(config["memory_mode"], "standard")
        self.assertEqual(config["forgetting_curve"]["enabled"], False)
    
    def test_create_ebbinghaus_config(self):
        """Test creating ebbinghaus configuration"""
        config = MemoryConfig.create_ebbinghaus_config(
            decay_rate=0.3,
            min_threshold=0.2
        )
        
        self.assertEqual(config["memory_mode"], "ebbinghaus")
        self.assertEqual(config["forgetting_curve"]["enabled"], True)
        self.assertEqual(config["forgetting_curve"]["decay_rate"], 0.3)
        self.assertEqual(config["forgetting_curve"]["min_retention_threshold"], 0.2)
    
    def test_validate_config_valid(self):
        """Test config validation with valid config"""
        config = MemoryConfig.get_config("testing")
        self.assertEqual(MemoryConfig.validate_config(config), True)
    
    def test_validate_config_missing_mode(self):
        """Test config validation with missing memory_mode"""
        config = {"forgetting_curve": {}}
        self.assertEqual(MemoryConfig.validate_config(config), False)
    
    def test_validate_config_invalid_mode(self):
        """Test config validation with invalid memory_mode"""
        config = {"memory_mode": "invalid"}
        self.assertEqual(MemoryConfig.validate_config(config), False)
    
    def test_validate_config_invalid_parameters(self):
        """Test config validation with invalid parameters"""
        config = {
            "memory_mode": "ebbinghaus",
            "forgetting_curve": {
                "initial_strength": 2.0,  # Should be <= 1.0
                "min_retention_threshold": -0.1  # Should be >= 0.0
            }
        }
        self.assertEqual(MemoryConfig.validate_config(config), False)


if __name__ == "__main__":
    # Run tests with: python -m unittest test_memory_config.py -v
    unittest.main(verbosity=2)
