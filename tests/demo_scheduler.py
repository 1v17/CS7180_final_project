"""
Simple test script for Phase 4: Memory Maintenance Scheduler

This script tests the scheduler functionality without complex database operations.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from memory_config import MemoryConfig
from memory_scheduler import MemoryMaintenanceScheduler

# Mock memory class for testing
class MockMemory:
    def __init__(self, mode="standard"):
        self.memory_mode = mode
        self.maintenance_count = 0
        self.mock_memories = [
            {"id": "mem1", "text": "Test memory 1"},
            {"id": "mem2", "text": "Test memory 2"},
            {"id": "mem3", "text": "Test memory 3"}
        ]
    
    def set_memory_mode(self, mode):
        self.memory_mode = mode
    
    def search(self, query="", limit=100):
        """Mock search method that returns all memories"""
        return self.mock_memories
    
    def update_memory_strength(self):
        self.maintenance_count += 1
        return {"updated": len(self.mock_memories), "processed": self.maintenance_count}
    
    def forget_weak_memories(self):
        return {"forgotten": 1, "archived": 0, "status": "success"}

def test_scheduler_modes():
    """Test scheduler behavior in different memory modes."""
    print("=== Simple Scheduler Test ===\n")
    
    # Test 1: Standard mode
    print("1. Testing STANDARD mode:")
    print("-" * 30)
    mock_memory = MockMemory("standard")
    scheduler = MemoryMaintenanceScheduler(mock_memory)
    
    started = scheduler.start()
    print(f"   Scheduler started: {started}")
    status = scheduler.get_status()
    print(f"   Status: running={status['is_running']}, mode={status['memory_mode']}")
    scheduler.stop()
    print()
    
    # Test 2: Ebbinghaus mode
    print("2. Testing EBBINGHAUS mode:")
    print("-" * 32)
    mock_memory = MockMemory("ebbinghaus")
    scheduler = MemoryMaintenanceScheduler(mock_memory)
    scheduler.set_maintenance_interval(2)  # 2 seconds for quick test
    
    started = scheduler.start()
    print(f"   Scheduler started: {started}")
    
    # Let it run for a few cycles
    print("   Running for 6 seconds...")
    time.sleep(6)
    
    status = scheduler.get_status()
    print(f"   Final status: running={status['is_running']}, count={status['maintenance_count']}")
    scheduler.stop()
    print()
    
    # Test 3: Manual maintenance
    print("3. Testing manual maintenance:")
    print("-" * 33)
    mock_memory = MockMemory("ebbinghaus")
    scheduler = MemoryMaintenanceScheduler(mock_memory)
    
    result = scheduler.force_maintenance()
    print(f"   Manual maintenance result: {result.get('status', 'unknown')}")
    print()

if __name__ == "__main__":
    test_scheduler_modes()
    print("=== Simple Test Complete ===")
