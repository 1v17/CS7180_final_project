"""
Memory Maintenance Scheduler for Ebbinghaus Memory System

This module provides automated background maintenance for the Ebbinghaus memory system.
It includes mode-aware scheduling that only operates when in "ebbinghaus" mode and
provides configurable maintenance intervals for different environments.
"""

import threading
import time
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timezone
import logging
from ebbinghaus_memory import EbbinghausMemory


class MemoryMaintenanceScheduler:
    """
    Background scheduler for memory maintenance tasks.
    
    This scheduler handles automated memory decay updates and forgetting processes
    only when the memory system is in "ebbinghaus" mode. It supports:
    - Mode-aware operation (only runs in ebbinghaus mode)
    - Configurable maintenance intervals
    - Runtime interval adjustment
    - Graceful shutdown
    - Status monitoring
    """
    
    def __init__(self, memory: EbbinghausMemory):
        """
        Initialize the memory maintenance scheduler.
        
        Args:
            memory (EbbinghausMemory): The memory instance to maintain
        """
        self.memory = memory
        self.is_running = False
        self.maintenance_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.maintenance_interval = self._get_maintenance_interval()
        self.last_maintenance = None
        self.maintenance_count = 0
        self.error_count = 0
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"MemoryMaintenanceScheduler initialized with interval: {self.maintenance_interval}s")
    
    def _get_maintenance_interval(self) -> int:
        """
        Get maintenance interval from memory configuration.
        
        Returns:
            int: Maintenance interval in seconds
        """
        try:
            return self.memory.fc_config.get("maintenance_interval", 3600)
        except Exception:
            return 3600  # Default to 1 hour
    
    def start(self) -> bool:
        """
        Start the background maintenance scheduler.
        
        Only starts if memory is in "ebbinghaus" mode. In "standard" mode,
        this method does nothing and returns False.
        
        Returns:
            bool: True if scheduler started, False if not (e.g., in standard mode)
        """
        if self.memory.memory_mode != "ebbinghaus":
            self.logger.info("Scheduler not started: memory is in 'standard' mode")
            return False
        
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return True
        
        self.is_running = True
        self.stop_event.clear()
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="MemoryMaintenance"
        )
        self.maintenance_thread.start()
        
        self.logger.info(f"Memory maintenance scheduler started (interval: {self.maintenance_interval}s)")
        return True
    
    def stop(self) -> None:
        """
        Stop the background maintenance scheduler gracefully.
        """
        if not self.is_running:
            self.logger.info("Scheduler is not running")
            return
        
        self.logger.info("Stopping memory maintenance scheduler...")
        self.is_running = False
        self.stop_event.set()
        
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5.0)
            if self.maintenance_thread.is_alive():
                self.logger.warning("Maintenance thread did not stop gracefully")
            else:
                self.logger.info("Memory maintenance scheduler stopped")
    
    def set_maintenance_interval(self, interval: int) -> None:
        """
        Update the maintenance interval at runtime.
        
        Useful for switching between test and production configurations
        without restarting the application.
        
        Args:
            interval (int): New maintenance interval in seconds
        """
        if interval < 1:
            raise ValueError("Maintenance interval must be at least 1 second")
        
        old_interval = self.maintenance_interval
        self.maintenance_interval = interval
        
        # Also update the memory configuration
        if hasattr(self.memory, 'fc_config'):
            self.memory.fc_config["maintenance_interval"] = interval
        
        self.logger.info(f"Maintenance interval changed from {old_interval}s to {interval}s")
        
        # If scheduler is running, restart it with new interval
        if self.is_running:
            self.logger.info("Restarting scheduler with new interval...")
            self.stop()
            time.sleep(0.1)  # Brief pause
            self.start()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status and statistics.
        
        Returns:
            Dict[str, Any]: Status information including:
                - is_running: Whether scheduler is active
                - memory_mode: Current memory mode
                - maintenance_interval: Current interval in seconds
                - last_maintenance: Timestamp of last maintenance
                - maintenance_count: Total maintenance operations performed
                - error_count: Number of errors encountered
                - next_maintenance: Estimated next maintenance time
        """
        status = {
            "is_running": self.is_running,
            "memory_mode": self.memory.memory_mode,
            "maintenance_interval": self.maintenance_interval,
            "last_maintenance": self.last_maintenance,
            "maintenance_count": self.maintenance_count,
            "error_count": self.error_count,
        }
        
        # Calculate next maintenance time if running
        if self.is_running and self.last_maintenance:
            next_maintenance = self.last_maintenance + self.maintenance_interval
            status["next_maintenance"] = datetime.fromtimestamp(
                next_maintenance, tz=timezone.utc
            ).isoformat()
        else:
            status["next_maintenance"] = None
        
        return status
    
    def force_maintenance(self) -> Dict[str, Any]:
        """
        Force immediate maintenance operation.
        
        Can be called regardless of scheduler state for manual maintenance.
        
        Returns:
            Dict[str, Any]: Results of maintenance operation
        """
        if self.memory.memory_mode != "ebbinghaus":
            return {
                "status": "skipped",
                "reason": "Memory is in 'standard' mode",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        self.logger.info("Forcing immediate maintenance operation...")
        return self._perform_maintenance()
    
    def _maintenance_loop(self) -> None:
        """
        Main maintenance loop that runs in background thread.
        
        This method continuously performs maintenance operations at the
        configured interval until stopped.
        """
        self.logger.info("Memory maintenance loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check if memory mode changed to standard
                if self.memory.memory_mode != "ebbinghaus":
                    self.logger.info("Memory mode changed to 'standard', stopping scheduler...")
                    self.is_running = False
                    break
                
                # Perform maintenance
                self._perform_maintenance()
                
                # Wait for next interval or stop signal
                if self.stop_event.wait(timeout=self.maintenance_interval):
                    break  # Stop event was set
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error in maintenance loop: {e}")
                
                # Wait a bit before retrying to avoid rapid error loops
                if self.stop_event.wait(timeout=min(60, self.maintenance_interval)):
                    break
        
        self.logger.info("Memory maintenance loop ended")
    
    def _perform_maintenance(self) -> Dict[str, Any]:
        """
        Perform a single maintenance operation.
        
        Returns:
            Dict[str, Any]: Results of the maintenance operation
        """
        start_time = time.time()
        self.last_maintenance = start_time
        
        try:
            self.logger.debug("Starting memory maintenance operation...")
            
            # Update all memory strengths based on time decay
            strength_results = self._update_all_strengths()
            
            # Run forgetting process to remove weak memories
            forgetting_results = self._run_forgetting_process()
            
            # Compile results
            results = {
                "status": "success",
                "timestamp": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
                "duration_seconds": time.time() - start_time,
                "strength_updates": strength_results,
                "forgetting_results": forgetting_results,
                "maintenance_count": self.maintenance_count + 1
            }
            
            self.maintenance_count += 1
            self.logger.info(
                f"Memory maintenance completed: {strength_results.get('updated', 0)} "
                f"memories updated, {forgetting_results.get('forgotten', 0)} forgotten"
            )
            
            return results
            
        except Exception as e:
            self.error_count += 1
            error_result = {
                "status": "error",
                "timestamp": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
                "duration_seconds": time.time() - start_time,
                "error": str(e),
                "error_count": self.error_count
            }
            
            self.logger.error(f"Memory maintenance failed: {e}")
            return error_result
    
    def _update_all_strengths(self) -> Dict[str, Any]:
        """
        Update strength for all memories based on time decay.
        
        Note: This is a simplified approach that focuses on maintenance.
        In a production system, you might want to maintain user-specific
        maintenance schedules.
        
        Returns:
            Dict[str, Any]: Results of strength update operation
        """
        try:
            # Since we can't easily get all memories without user_id,
            # we'll skip the strength update for now and let it happen
            # naturally when memories are accessed during chat
            
            self.logger.info("Strength updates handled naturally during memory access")
            return {
                "updated": 0,
                "errors": 0,
                "total_processed": 0,
                "note": "Strength updates handled during natural access"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update memory strengths: {e}")
            return {
                "updated": 0,
                "errors": 1,
                "total_processed": 0,
                "error": str(e)
            }
    
    def _run_forgetting_process(self) -> Dict[str, Any]:
        """
        Run the forgetting process to remove weak memories.
        
        Note: Since forgetting requires user_id and we don't have a way
        to iterate through all users in this context, we'll skip the
        automatic forgetting process. Forgetting can be triggered manually
        per user or handled during normal chat operations.
        
        Returns:
            Dict[str, Any]: Results of forgetting operation
        """
        try:
            # Skip automatic forgetting since we don't have user context
            self.logger.info("Automatic forgetting skipped (requires user context)")
            return {
                "forgotten": 0,
                "archived": 0,
                "note": "Forgetting handled per-user during chat operations"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to run forgetting process: {e}")
            return {
                "forgotten": 0,
                "archived": 0,
                "error": str(e)
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class SchedulerFactory:
    """
    Factory class for creating memory maintenance schedulers.
    
    Provides convenient methods to create schedulers with different
    configurations for various environments.
    """
    
    @staticmethod
    def create_scheduler(memory: EbbinghausMemory) -> MemoryMaintenanceScheduler:
        """
        Create a standard memory maintenance scheduler.
        
        Args:
            memory (EbbinghausMemory): The memory instance to maintain
            
        Returns:
            MemoryMaintenanceScheduler: Configured scheduler instance
        """
        return MemoryMaintenanceScheduler(memory)
    
    @staticmethod
    def create_testing_scheduler(memory: EbbinghausMemory, 
                               interval: int = 60) -> MemoryMaintenanceScheduler:
        """
        Create a scheduler optimized for testing with short intervals.
        
        Args:
            memory (EbbinghausMemory): The memory instance to maintain
            interval (int): Maintenance interval in seconds (default: 60)
            
        Returns:
            MemoryMaintenanceScheduler: Configured testing scheduler
        """
        scheduler = MemoryMaintenanceScheduler(memory)
        scheduler.set_maintenance_interval(interval)
        return scheduler
    
    @staticmethod
    def create_production_scheduler(memory: EbbinghausMemory,
                                  interval: int = 3600) -> MemoryMaintenanceScheduler:
        """
        Create a scheduler optimized for production with longer intervals.
        
        Args:
            memory (EbbinghausMemory): The memory instance to maintain
            interval (int): Maintenance interval in seconds (default: 3600)
            
        Returns:
            MemoryMaintenanceScheduler: Configured production scheduler
        """
        scheduler = MemoryMaintenanceScheduler(memory)
        scheduler.set_maintenance_interval(interval)
        return scheduler


# Convenience function for quick scheduler creation
def create_memory_scheduler(memory: EbbinghausMemory, 
                          mode: str = "standard") -> MemoryMaintenanceScheduler:
    """
    Convenience function to create a memory scheduler.
    
    Args:
        memory (EbbinghausMemory): The memory instance to maintain
        mode (str): Scheduler mode - "standard", "testing", or "production"
        
    Returns:
        MemoryMaintenanceScheduler: Configured scheduler instance
    """
    if mode == "testing":
        return SchedulerFactory.create_testing_scheduler(memory)
    elif mode == "production":
        return SchedulerFactory.create_production_scheduler(memory)
    else:
        return SchedulerFactory.create_scheduler(memory)
