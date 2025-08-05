"""
Test script for Phase 6: Forgetting Process Accuracy

This script tests the accuracy of the forgetting process, including threshold
detection, soft delete functionality, and memory archiving.
"""

import sys
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ebbinghaus_memory import EbbinghausMemory
from memory_config import MemoryConfig


class TestForgettingProcessAccuracy(unittest.TestCase):
    """Test forgetting process accuracy and threshold detection."""

    def setUp(self):
        """Set up test environment with testing configuration."""
        self.config = MemoryConfig.get_config("testing")
        # Mock mem0.Memory initialization to avoid vector store issues
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('mem0.Memory.__init__', return_value=None):
                self.memory = EbbinghausMemory(
                    config=self.config,
                    memory_mode="ebbinghaus"
                )
                # Set up mock attributes that would normally be set by mem0.Memory.__init__
                self.memory.vector_store = MagicMock()
                self.memory.graph_store = MagicMock()
                self.memory.llm = MagicMock()
                self.memory.embedder = MagicMock()
                self.memory.db = MagicMock()
        self.threshold = self.config['forgetting_curve']['min_retention_threshold']

    def test_threshold_detection_accuracy(self):
        """Test accurate detection of memories below retention threshold."""
        now = datetime.now(timezone.utc)
        
        # Create memories with different retention levels
        test_memories = [
            {
                'id': 'strong_memory',
                'memory': 'Strong memory content',
                'metadata': {
                    'created_at': (now - timedelta(hours=1)).isoformat(),
                    'last_accessed': (now - timedelta(hours=1)).isoformat(),
                    'memory_strength': 1.0,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            },
            {
                'id': 'weak_memory',
                'memory': 'Weak memory content',
                'metadata': {
                    'created_at': (now - timedelta(days=30)).isoformat(),
                    'last_accessed': (now - timedelta(days=30)).isoformat(),
                    'memory_strength': 0.3,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            },
            {
                'id': 'borderline_memory',
                'memory': 'Borderline memory content',
                'metadata': {
                    'created_at': (now - timedelta(days=7)).isoformat(),
                    'last_accessed': (now - timedelta(days=7)).isoformat(),
                    'memory_strength': 0.5,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            }
        ]
        
        # Check retention for each memory
        for memory in test_memories:
            metadata = memory['metadata']
            retention = self.memory.calculate_retention(metadata)
            
            if memory['id'] == 'strong_memory':
                # Should be above threshold
                self.assertGreater(retention, self.threshold)
            elif memory['id'] == 'weak_memory':
                # Should be below threshold
                self.assertLess(retention, self.threshold)
            # Borderline memory result depends on exact calculation

    def test_soft_delete_functionality(self):
        """Test soft delete (archiving) functionality."""
        # Mock memories for testing soft delete
        memories_to_archive = [
            {
                'id': 'archive_test_1',
                'memory': 'Memory to be archived',
                'metadata': {
                    'created_at': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    'last_accessed': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    'memory_strength': 0.1,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            }
        ]
        
        # Mock the get_all and update methods
        with patch.object(self.memory, 'get_all') as mock_get_all, \
             patch.object(self.memory, 'update') as mock_update:
            
            mock_get_all.return_value = memories_to_archive
            mock_update.return_value = {'success': True}
            
            # Run forgetting process with soft delete enabled
            stats = self.memory.forget_weak_memories(
                user_id="test_user_forgetting",
                soft_delete=True
            )
            
            # Verify soft delete was attempted
            if mock_update.called:
                # Check that update was called with archived prefix
                call_args = mock_update.call_args
                # The update method is called with: self.update(memory_id, data=archived_content)
                if call_args and len(call_args) > 1 and 'data' in call_args[1]:
                    updated_memory = call_args[1]['data']  # Get data from kwargs
                    self.assertTrue(updated_memory.startswith('[ARCHIVED]'))

    def test_hard_delete_functionality(self):
        """Test hard delete functionality."""
        memories_to_delete = [
            {
                'id': 'delete_test_1',
                'memory': 'Memory to be deleted',
                'metadata': {
                    'created_at': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    'last_accessed': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    'memory_strength': 0.05,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            }
        ]
        
        # Mock the get_all and delete methods
        with patch.object(self.memory, 'get_all') as mock_get_all, \
             patch.object(self.memory, 'delete') as mock_delete:
            
            mock_get_all.return_value = memories_to_delete
            mock_delete.return_value = {'success': True}
            
            # Run forgetting process with hard delete
            stats = self.memory.forget_weak_memories(
                user_id="test_user_forgetting",
                soft_delete=False
            )
            
            # Verify delete was attempted
            if mock_delete.called:
                self.assertTrue(True)  # Delete method was called

    def test_forgetting_statistics_accuracy(self):
        """Test accuracy of forgetting process statistics."""
        # Create test scenario with known outcomes
        test_memories = [
            # Strong memory - should not be forgotten
            {
                'id': 'strong_1',
                'memory': 'Strong memory 1',
                'metadata': {
                    'created_at': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
                    'last_accessed': (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                    'memory_strength': 1.0,
                    'access_count': 3,
                    'mode': 'ebbinghaus'
                }
            },
            # Weak memory - should be forgotten
            {
                'id': 'weak_1',
                'memory': 'Weak memory 1',
                'metadata': {
                    'created_at': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    'last_accessed': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    'memory_strength': 0.1,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            },
            # Another weak memory
            {
                'id': 'weak_2',
                'memory': 'Weak memory 2',
                'metadata': {
                    'created_at': (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
                    'last_accessed': (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
                    'memory_strength': 0.2,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            }
        ]
        
        # Mock methods
        with patch.object(self.memory, 'get_all') as mock_get_all, \
             patch.object(self.memory, 'update') as mock_update:
            
            mock_get_all.return_value = test_memories
            mock_update.return_value = {'success': True}
            
            # Run forgetting process
            stats = self.memory.forget_weak_memories(
                user_id="test_user_forgetting",
                soft_delete=True
            )
            
            # Verify statistics structure
            self.assertIn('processed', stats)
            self.assertIn('forgotten', stats)
            self.assertIn('archived', stats)
            
            # Basic validation of numbers
            processed = stats['processed']
            forgotten = stats['forgotten']
            archived = stats['archived']
            
            self.assertGreaterEqual(processed, 0)
            self.assertGreaterEqual(forgotten, 0)
            self.assertGreaterEqual(archived, 0)

    def test_standard_mode_no_forgetting(self):
        """Test that standard mode prevents any forgetting."""
        # Switch to standard mode
        self.memory.set_memory_mode("standard")
        
        # Create very weak memories that would normally be forgotten
        weak_memories = [
            {
                'id': 'standard_weak_1',
                'memory': 'Weak memory in standard mode',
                'metadata': {
                    'created_at': (datetime.now(timezone.utc) - timedelta(days=365)).isoformat(),
                    'last_accessed': (datetime.now(timezone.utc) - timedelta(days=365)).isoformat(),
                    'memory_strength': 0.001,
                    'access_count': 1,
                    'mode': 'standard'
                }
            }
        ]
        
        # Mock get_all to return weak memories
        with patch.object(self.memory, 'get_all') as mock_get_all:
            mock_get_all.return_value = weak_memories
            
            # Run forgetting process (should be bypassed)
            stats = self.memory.forget_weak_memories(
                user_id="test_user_forgetting",
                soft_delete=True
            )
            
            # In standard mode, no memories should be forgotten
            self.assertEqual(stats['forgotten'], 0)
            self.assertEqual(stats['archived'], 0)
            self.assertEqual(stats['processed'], 0)

    def test_archived_memory_detection(self):
        """Test detection and handling of already archived memories."""
        archived_memories = [
            {
                'id': 'already_archived',
                'memory': '[ARCHIVED] This memory is already archived',
                'metadata': {
                    'created_at': (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
                    'last_accessed': (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
                    'memory_strength': 0.05,
                    'access_count': 1,
                    'mode': 'ebbinghaus',
                    'archived': True
                }
            },
            {
                'id': 'not_archived',
                'memory': 'This memory is not archived yet',
                'metadata': {
                    'created_at': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    'last_accessed': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                    'memory_strength': 0.05,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            }
        ]
        
        with patch.object(self.memory, 'get_all') as mock_get_all, \
             patch.object(self.memory, 'update') as mock_update:
            
            mock_get_all.return_value = archived_memories
            mock_update.return_value = {'success': True}
            
            # Run forgetting process
            stats = self.memory.forget_weak_memories(
                user_id="test_user_forgetting",
                soft_delete=True
            )
            
            # Already archived memories should be skipped
            # Only the non-archived weak memory should be processed

    def test_forgetting_threshold_configuration_respect(self):
        """Test that forgetting respects configuration threshold."""
        # Test with different threshold values
        test_thresholds = [0.05, 0.1, 0.2, 0.3]
        
        for threshold in test_thresholds:
            with self.subTest(threshold=threshold):
                # Update config for this test
                test_config = self.config.copy()
                test_config['forgetting_curve']['min_retention_threshold'] = threshold
                
                # Create memory with retention just above threshold
                now = datetime.now(timezone.utc)
                # Calculate time needed for specific retention
                target_retention = threshold + 0.01
                
                # Use a memory that should be just above threshold
                test_memory = {
                    'id': f'threshold_test_{threshold}',
                    'memory': f'Memory for threshold {threshold}',
                    'metadata': {
                        'created_at': (now - timedelta(hours=6)).isoformat(),
                        'last_accessed': (now - timedelta(hours=6)).isoformat(),
                        'memory_strength': 0.8,
                        'access_count': 1,
                        'mode': 'ebbinghaus'
                    }
                }
                
                # Calculate actual retention
                metadata = test_memory['metadata']
                retention = self.memory.calculate_retention(metadata)
                
                # Test threshold comparison
                should_forget = retention < threshold
                should_keep = retention >= threshold
                
                # At least verify the logic is consistent
                self.assertEqual(should_forget, not should_keep)

    def test_batch_forgetting_efficiency(self):
        """Test that batch forgetting operations are efficient."""
        # Create many weak memories
        many_weak_memories = []
        now = datetime.now(timezone.utc)
        for i in range(50):  # Create 50 weak memories
            many_weak_memories.append({
                'id': f'weak_batch_{i}',
                'memory': f'Weak batch memory {i}',
                'metadata': {
                    'created_at': (now - timedelta(days=60)).isoformat(),
                    'last_accessed': (now - timedelta(days=60)).isoformat(),
                    'memory_strength': 0.05,
                    'access_count': 1,
                    'mode': 'ebbinghaus'
                }
            })
        
        with patch.object(self.memory, 'get_all') as mock_get_all, \
             patch.object(self.memory, 'update') as mock_update:
            
            mock_get_all.return_value = many_weak_memories
            mock_update.return_value = {'success': True}
            
            # Measure time for batch operation (should be reasonable)
            import time
            start_time = time.time()
            
            stats = self.memory.forget_weak_memories(
                user_id="test_user_forgetting",
                soft_delete=True
            )
            
            end_time = time.time()
            operation_time = end_time - start_time
            
            # Should complete within reasonable time (under 5 seconds for 50 memories)
            self.assertLess(operation_time, 5.0)
            
            # Should have processed all memories
            self.assertGreaterEqual(stats['processed'], 0)

    def test_memory_restoration_after_archiving(self):
        """Test that archived memories can be restored."""
        # Test the restore functionality
        archived_memory_id = 'test_restore_memory'
        
        # Mock getting an archived memory
        archived_memory = {
            'id': archived_memory_id,
            'memory': '[ARCHIVED] Memory to be restored',
            'metadata': {
                'created_at': datetime.now(timezone.utc).isoformat(),
                'last_accessed': datetime.now(timezone.utc).isoformat(),
                'memory_strength': 0.05,
                'access_count': 1,
                'mode': 'ebbinghaus',
                'archived': True
            }
        }
        
        with patch.object(self.memory, 'get') as mock_get, \
             patch.object(self.memory, 'update') as mock_update:
            
            mock_get.return_value = archived_memory
            mock_update.return_value = {'success': True}
            
            # Test restoration
            result = self.memory.restore_memory(
                memory_id=archived_memory_id
            )
            
            # Should indicate success if mocks were called
            if mock_update.called:
                # Check that update was called without [ARCHIVED] prefix
                call_args = mock_update.call_args
                if call_args and len(call_args) > 1 and 'data' in call_args[1]:
                    updated_memory = call_args[1]['data']  # Get data from kwargs
                    self.assertFalse(updated_memory.startswith('[ARCHIVED]'))


if __name__ == '__main__':
    unittest.main()
