"""
Integration test suite for EbbinghausMemory system

This test file covers integration tests for the complete EbbinghausMemory
system, testing the interaction between different components and modes.
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your classes
from ebbinghaus_memory import EbbinghausMemory
from memory_config import MemoryConfig


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    @patch('mem0.Memory.add')
    @patch('mem0.Memory.search')
    def test_mode_switching_integration(self, mock_search, mock_add):
        """Test switching modes and verifying behavior changes"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            memory = EbbinghausMemory(memory_mode="standard")
            
            # Add memory in standard mode
            mock_add.return_value = "id1"
            memory.add("Test message 1", user_id="user1")
            
            # Check standard mode call
            call_args = mock_add.call_args
            self.assertIsNone(call_args[1].get('metadata'))
            
            # Switch to ebbinghaus mode
            memory.set_memory_mode("ebbinghaus")
            
            # Add memory in ebbinghaus mode
            mock_add.return_value = "id2"
            memory.add("Test message 2", user_id="user1")
            
            # Check ebbinghaus mode call
            call_args = mock_add.call_args
            metadata = call_args[1]['metadata']
            self.assertEqual(metadata["mode"], "ebbinghaus")
            self.assertIn("memory_strength", metadata)


if __name__ == "__main__":
    # Run tests with: python -m unittest test_integration.py -v
    unittest.main(verbosity=2)
