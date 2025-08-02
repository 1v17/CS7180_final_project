"""
Test suite for EbbinghausMemory core functionality

This test file covers the core EbbinghausMemory class functionality including
both standard and ebbinghaus memory modes, testing mode switching, memory 
strength tracking, and decay functionality.
"""

import unittest
import time
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your classes
from ebbinghaus_memory import EbbinghausMemory
from memory_config import MemoryConfig


class TestEbbinghausMemory(unittest.TestCase):
    """Test suite for EbbinghausMemory class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock the entire mem0.Memory initialization to avoid vector store issues
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('mem0.Memory.__init__', return_value=None):
                # Create mock instances
                self.standard_memory = EbbinghausMemory(memory_mode="standard")
                self.ebbinghaus_memory = EbbinghausMemory(memory_mode="ebbinghaus")
                
                # Set up mock attributes that would normally be set by mem0.Memory.__init__
                for memory in [self.standard_memory, self.ebbinghaus_memory]:
                    memory.vector_store = MagicMock()
                    memory.graph_store = MagicMock()
                    memory.llm = MagicMock()
                    memory.embedder = MagicMock()
                    memory.db = MagicMock()
    
    def test_initialization_standard_mode(self):
        """Test EbbinghausMemory initializes correctly in standard mode"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('mem0.Memory.__init__', return_value=None):
                memory = EbbinghausMemory(memory_mode="standard")
                self.assertEqual(memory.memory_mode, "standard")
                self.assertTrue(hasattr(memory, 'ebbinghaus_config'))
                self.assertTrue(hasattr(memory, 'fc_config'))
    
    def test_initialization_ebbinghaus_mode(self):
        """Test EbbinghausMemory initializes correctly in ebbinghaus mode"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('mem0.Memory.__init__', return_value=None):
                memory = EbbinghausMemory(memory_mode="ebbinghaus")
                self.assertEqual(memory.memory_mode, "ebbinghaus")
    
    def test_initialization_with_config(self):
        """Test EbbinghausMemory initializes with custom config"""
        custom_config = MemoryConfig.get_config("testing")
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('mem0.Memory.__init__', return_value=None):
                memory = EbbinghausMemory(config=custom_config, memory_mode="ebbinghaus")
                self.assertEqual(memory.memory_mode, "ebbinghaus")
                self.assertEqual(memory.ebbinghaus_config, custom_config)
    
    def test_invalid_memory_mode(self):
        """Test that invalid memory modes raise ValueError"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('mem0.Memory.__init__', return_value=None):
                with self.assertRaisesRegex(ValueError, "memory_mode must be 'standard' or 'ebbinghaus'"):
                    EbbinghausMemory(memory_mode="invalid")
    
    def test_set_memory_mode(self):
        """Test dynamic memory mode switching"""
        memory = self.standard_memory
        
        # Switch from standard to ebbinghaus
        memory.set_memory_mode("ebbinghaus")
        self.assertEqual(memory.memory_mode, "ebbinghaus")
        
        # Switch back to standard
        memory.set_memory_mode("standard")
        self.assertEqual(memory.memory_mode, "standard")
    
    def test_set_memory_mode_invalid(self):
        """Test that invalid mode switching raises ValueError"""
        memory = self.standard_memory
        with self.assertRaisesRegex(ValueError, "Mode must be 'standard' or 'ebbinghaus'"):
            memory.set_memory_mode("invalid")
    
    @patch('mem0.Memory.add')
    def test_add_standard_mode(self, mock_add):
        """Test adding memory in standard mode"""
        mock_add.return_value = "memory_id_123"
        
        memory = self.standard_memory
        result = memory.add("Test message", user_id="user1")
        
        # Should call parent add without strength metadata
        mock_add.assert_called_once_with(
            "Test message", 
            user_id="user1", 
            metadata=None
        )
        self.assertEqual(result, "memory_id_123")
    
    @patch('mem0.Memory.add')
    def test_add_ebbinghaus_mode(self, mock_add):
        """Test adding memory in ebbinghaus mode"""
        mock_add.return_value = "memory_id_456"
        
        memory = self.ebbinghaus_memory
        result = memory.add("Test message", user_id="user1")
        
        # Should call parent add with strength metadata
        mock_add.assert_called_once()
        call_args = mock_add.call_args
        
        self.assertEqual(call_args[0][0], "Test message")  # message
        self.assertEqual(call_args[1]['user_id'], "user1")
        
        # Check metadata contains ebbinghaus fields
        metadata = call_args[1]['metadata']
        self.assertIn("created_at", metadata)
        self.assertIn("last_accessed", metadata)
        self.assertIn("memory_strength", metadata)
        self.assertIn("access_count", metadata)
        self.assertEqual(metadata["mode"], "ebbinghaus")
        self.assertEqual(metadata["memory_strength"], 1.0)
        self.assertEqual(metadata["access_count"], 0)
        
        self.assertEqual(result, "memory_id_456")
    
    @patch('mem0.Memory.add')
    def test_add_ebbinghaus_mode_with_existing_metadata(self, mock_add):
        """Test adding memory in ebbinghaus mode with existing metadata"""
        mock_add.return_value = "memory_id_789"
        
        memory = self.ebbinghaus_memory
        existing_metadata = {"category": "test", "importance": "high"}
        
        result = memory.add(
            "Test message", 
            user_id="user1", 
            metadata=existing_metadata
        )
        
        # Check that existing metadata is preserved and ebbinghaus metadata is added
        call_args = mock_add.call_args
        metadata = call_args[1]['metadata']
        
        self.assertEqual(metadata["category"], "test")
        self.assertEqual(metadata["importance"], "high")
        self.assertEqual(metadata["mode"], "ebbinghaus")
        self.assertIn("memory_strength", metadata)
    
    def test_calculate_retention_standard_mode(self):
        """Test retention calculation in standard mode always returns 1.0"""
        memory = self.standard_memory
        
        # Any metadata should return perfect retention
        test_metadata = {
            "created_at": "2024-01-01T00:00:00+00:00",
            "memory_strength": 0.5
        }
        
        retention = memory.calculate_retention(test_metadata)
        self.assertEqual(retention, 1.0)
    
    def test_calculate_retention_ebbinghaus_mode(self):
        """Test retention calculation in ebbinghaus mode"""
        memory = self.ebbinghaus_memory
        
        # Test with recent memory (should have high retention)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_metadata = {
            "created_at": recent_time.isoformat(),
            "memory_strength": 1.0,
            "mode": "ebbinghaus"
        }
        
        retention = memory.calculate_retention(recent_metadata)
        self.assertGreater(retention, 0.9)
        self.assertLessEqual(retention, 1.0)  # Should be very high for recent memory
        
        # Test with old memory (should have lower retention)
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        old_metadata = {
            "created_at": old_time.isoformat(),
            "memory_strength": 1.0,
            "mode": "ebbinghaus"
        }
        
        retention = memory.calculate_retention(old_metadata)
        self.assertGreaterEqual(retention, 0.0)
        self.assertLess(retention, 0.9)  # Should be lower for old memory
    
    def test_calculate_retention_missing_metadata(self):
        """Test retention calculation with missing metadata"""
        memory = self.ebbinghaus_memory
        
        # Missing required fields should return 1.0 (fallback)
        incomplete_metadata = {"mode": "ebbinghaus"}
        retention = memory.calculate_retention(incomplete_metadata)
        self.assertEqual(retention, 1.0)
        
        # Non-ebbinghaus memory should return 1.0
        standard_metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "memory_strength": 1.0
        }
        retention = memory.calculate_retention(standard_metadata)
        self.assertEqual(retention, 1.0)
    
    def test_calculate_retention_with_different_strengths(self):
        """Test retention calculation with different memory strengths"""
        memory = self.ebbinghaus_memory
        
        base_time = datetime.now(timezone.utc) - timedelta(days=1)
        
        # Strong memory
        strong_metadata = {
            "created_at": base_time.isoformat(),
            "memory_strength": 1.0,
            "mode": "ebbinghaus"
        }
        strong_retention = memory.calculate_retention(strong_metadata)
        
        # Weak memory
        weak_metadata = {
            "created_at": base_time.isoformat(),
            "memory_strength": 0.2,
            "mode": "ebbinghaus"
        }
        weak_retention = memory.calculate_retention(weak_metadata)
        
        # Strong memory should have higher retention
        self.assertGreater(strong_retention, weak_retention)
    
    @patch('mem0.Memory.get')
    @patch('mem0.Memory.update')
    def test_update_memory_strength_standard_mode(self, mock_update, mock_get):
        """Test that update_memory_strength does nothing in standard mode"""
        memory = self.standard_memory
        
        memory.update_memory_strength("memory_id", boost=True)
        
        # Should not call get or update in standard mode
        mock_get.assert_not_called()
        mock_update.assert_not_called()
    
    @patch('mem0.Memory.get')
    def test_update_memory_strength_ebbinghaus_mode(self, mock_get):
        """Test memory strength update in ebbinghaus mode (now simplified)"""
        memory = self.ebbinghaus_memory
        
        # Mock memory details
        mock_memory = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "memory_strength": 0.7,
                "access_count": 2,
                "mode": "ebbinghaus"
            }
        }
        mock_get.return_value = mock_memory
        
        # This should not raise an error, but won't actually update due to API limitations
        memory.update_memory_strength("memory_id", boost=True)
        
        # Should call get to check memory details
        mock_get.assert_called_once_with("memory_id")
        
        # Note: The actual metadata update is skipped in the current implementation
        # due to Mem0 API limitations. Strength updates happen naturally during 
        # memory access operations instead.
    
    @patch('mem0.Memory.get')
    def test_update_memory_strength_error_handling(self, mock_get):
        """Test that update_memory_strength handles errors gracefully"""
        memory = self.ebbinghaus_memory
        
        # Mock get to raise an exception
        mock_get.side_effect = Exception("Memory not found")
        
        # Should not raise an error
        memory.update_memory_strength("nonexistent_memory_id", boost=True)
        
        mock_get.assert_called_once_with("nonexistent_memory_id")
    
    @patch('mem0.Memory.get')
    def test_update_memory_strength_non_ebbinghaus_memory(self, mock_get):
        """Test that non-ebbinghaus memories are skipped"""
        memory = self.ebbinghaus_memory
        
        mock_memory = {
            "metadata": {
                "some_field": "value"
                # No "mode" field
            }
        }
        mock_get.return_value = mock_memory
        
        # Should not raise an error and should handle gracefully
        memory.update_memory_strength("memory_id", boost=True)
        mock_get.assert_called_once()
    
    @patch('mem0.Memory.search')
    def test_search_standard_mode(self, mock_search):
        """Test search in standard mode passes through to parent"""
        memory = self.standard_memory
        mock_search.return_value = [{"id": "1", "content": "test"}]
        
        results = memory.search("test query", user_id="user1")
        
        mock_search.assert_called_once_with("test query", user_id="user1", limit=100)
        self.assertEqual(results, [{"id": "1", "content": "test"}])
    
    @patch('mem0.Memory.search')
    @patch.object(EbbinghausMemory, 'calculate_retention')
    @patch.object(EbbinghausMemory, 'update_memory_strength')
    def test_search_ebbinghaus_mode(self, mock_update_strength, mock_calc_retention, mock_search):
        """Test search in ebbinghaus mode with filtering"""
        memory = self.ebbinghaus_memory
        
        # Mock search results
        mock_search.return_value = [
            {
                "id": "1",
                "content": "strong memory",
                "metadata": {
                    "mode": "ebbinghaus",
                    "memory_strength": 0.8
                }
            },
            {
                "id": "2", 
                "content": "weak memory",
                "metadata": {
                    "mode": "ebbinghaus",
                    "memory_strength": 0.1
                }
            },
            {
                "id": "3",
                "content": "standard memory",
                "metadata": {
                    "some_field": "value"
                }
            }
        ]
        
        # Mock retention calculations
        mock_calc_retention.side_effect = [0.8, 0.05, 1.0]  # strong, weak, standard
        
        results = memory.search("test query", user_id="user1")
        
        # Should filter out weak memory (retention < 0.1)
        self.assertEqual(len(results), 2)
        
        # Results should be sorted by retention score in descending order
        # Memory 3 (standard) has retention 1.0, Memory 1 (ebbinghaus) has retention 0.8
        self.assertEqual(results[0]["id"], "3")  # Standard memory (highest retention)
        self.assertEqual(results[1]["id"], "1")  # Strong ebbinghaus memory
        
        # Should add retention scores for ebbinghaus memories only
        self.assertIn("retention_score", results[1])  # Memory 1 (ebbinghaus)
        self.assertEqual(results[1]["retention_score"], 0.8)
        self.assertNotIn("retention_score", results[0])  # Memory 3 (standard)
        
        # Should call update_memory_strength for retrieved ebbinghaus memories only
        self.assertEqual(mock_update_strength.call_count, 1)  # Only for memory 1 (ebbinghaus)
    
    def test_get_memory_statistics_standard_mode(self):
        """Test memory statistics in standard mode"""
        with patch.object(self.standard_memory, 'get_all') as mock_get_all:
            mock_get_all.return_value = [
                {"id": "1", "metadata": {}},
                {"id": "2", "metadata": {"category": "test"}},
                {"id": "3"}  # No metadata
            ]
            
            stats = self.standard_memory.get_memory_statistics(user_id="user1")
            
            self.assertEqual(stats["mode"], "standard")
            self.assertEqual(stats["total_memories"], 3)
            self.assertEqual(stats["standard_memories"], 3)
            self.assertEqual(stats["ebbinghaus_memories"], 0)
    
    def test_get_memory_statistics_ebbinghaus_mode(self):
        """Test memory statistics in ebbinghaus mode"""
        with patch.object(self.ebbinghaus_memory, 'get_all') as mock_get_all:
            with patch.object(self.ebbinghaus_memory, 'calculate_retention') as mock_calc_retention:
                
                mock_get_all.return_value = [
                    {
                        "id": "1", 
                        "metadata": {
                            "mode": "ebbinghaus", 
                            "memory_strength": 0.8
                        }
                    },
                    {
                        "id": "2", 
                        "metadata": {
                            "mode": "ebbinghaus", 
                            "memory_strength": 0.3,
                            "archived": True
                        }
                    },
                    {
                        "id": "3", 
                        "metadata": {"category": "standard"}
                    }
                ]
                
                # Mock retention calculations
                mock_calc_retention.side_effect = [0.8, 0.05]  # strong, weak
                
                stats = self.ebbinghaus_memory.get_memory_statistics(user_id="user1")
                
                self.assertEqual(stats["mode"], "ebbinghaus")
                self.assertEqual(stats["total_memories"], 3)
                self.assertEqual(stats["ebbinghaus_memories"], 2)
                self.assertEqual(stats["standard_memories"], 1)
                self.assertEqual(stats["average_strength"], 0.55)  # (0.8 + 0.3) / 2
                self.assertAlmostEqual(stats["average_retention"], 0.425, places=10)  # (0.8 + 0.05) / 2
                self.assertEqual(stats["weak_memories"], 1)  # retention < 0.1
                self.assertEqual(stats["archived_memories"], 1)

    def test_get_memory_statistics_no_user_id(self):
        """Test memory statistics when no user_id is provided"""
        stats = self.ebbinghaus_memory.get_memory_statistics()
        
        self.assertEqual(stats["mode"], "ebbinghaus")
        self.assertEqual(stats["total_memories"], "N/A (requires user_id)")
        self.assertEqual(stats["ebbinghaus_memories"], "N/A")
        self.assertEqual(stats["standard_memories"], "N/A")
        self.assertEqual(stats["strong_memories"], "N/A")
        self.assertEqual(stats["weak_memories"], "N/A")
        self.assertEqual(stats["archived_memories"], "N/A")
        self.assertIn("note", stats)

    @patch('mem0.Memory.get_all')
    @patch('mem0.Memory.delete')
    @patch.object(EbbinghausMemory, 'calculate_retention')
    def test_forget_weak_memories_ebbinghaus_mode(self, mock_calc_retention, mock_delete, mock_get_all):
        """Test forgetting weak memories in ebbinghaus mode"""
        memory = self.ebbinghaus_memory
        
        # Mock memories with different retention scores
        mock_get_all.return_value = [
            {
                "id": "strong_memory",
                "metadata": {
                    "mode": "ebbinghaus",
                    "memory_strength": 0.8
                }
            },
            {
                "id": "weak_memory", 
                "metadata": {
                    "mode": "ebbinghaus",
                    "memory_strength": 0.1
                }
            },
            {
                "id": "standard_memory",
                "metadata": {"category": "standard"}
            }
        ]
        
        # Mock retention calculations - weak memory should be below threshold
        mock_calc_retention.side_effect = [0.8, 0.05]  # strong, weak (standard skipped)
        
        result = memory.forget_weak_memories(user_id="user1")
        
        # Should delete the weak memory
        mock_delete.assert_called_once_with("weak_memory")
        
        # Check results
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["forgotten"], 1)
        self.assertEqual(result["archived"], 0)  # We don't do archiving anymore

    def test_forget_weak_memories_standard_mode(self):
        """Test that forgetting is skipped in standard mode"""
        memory = self.standard_memory
        
        result = memory.forget_weak_memories(user_id="user1")
        
        # Should return zero stats for standard mode
        self.assertEqual(result["processed"], 0)
        self.assertEqual(result["forgotten"], 0)
        self.assertEqual(result["archived"], 0)


if __name__ == "__main__":
    # Run tests with: python -m unittest test_ebbinghaus_memory.py -v
    # Updated tests reflect the current implementation after API limitation fixes
    unittest.main(verbosity=2)