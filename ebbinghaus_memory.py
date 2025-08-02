"""
Ebbinghaus Memory Extension for Mem0

This module extends Mem0's Memory class to support memory decay based on the 
Ebbinghaus forgetting curve. It provides two modes:
- "standard": Traditional perfect memory (default Mem0 behavior)
- "ebbinghaus": Memory with strength tracking and decay over time

Author: CS7180 Final Project
Date: August 2025
"""

import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from mem0 import Memory
from memory_config import MemoryConfig  # Add this import


class EbbinghausMemory(Memory):
    """
    Extended Memory class that implements Ebbinghaus forgetting curve.
    
    This class extends Mem0's Memory to add support for memory strength tracking
    and time-based decay. It can operate in two modes:
    - "standard": Uses default Mem0 behavior (perfect memory)
    - "ebbinghaus": Applies forgetting curve with strength metadata
    """
    
    def __init__(self, config: Optional[Dict] = None, memory_mode: str = "standard"):
        """
        Initialize EbbinghausMemory with mode support.
        
        Args:
            config (Dict, optional): Memory configuration for Mem0
            memory_mode (str): Memory mode - "standard" or "ebbinghaus"
        """
        super().__init__(config)
        
        # Validate memory mode
        if memory_mode not in ["standard", "ebbinghaus"]:
            raise ValueError("memory_mode must be 'standard' or 'ebbinghaus'")
        
        self.memory_mode = memory_mode
        
        # Use MemoryConfig for configuration management
        if config:
            self.config = MemoryConfig.from_dict(config)
        else:
            # Use default configuration
            self.config = MemoryConfig()
        
        print(f"EbbinghausMemory initialized in '{self.memory_mode}' mode")
    
    def set_memory_mode(self, mode: str) -> None:
        """
        Switch memory mode dynamically.
        
        Args:
            mode (str): New memory mode - "standard" or "ebbinghaus"
        """
        if mode not in ["standard", "ebbinghaus"]:
            raise ValueError("Mode must be 'standard' or 'ebbinghaus'")
        
        old_mode = self.memory_mode
        self.memory_mode = mode
        print(f"Memory mode changed from '{old_mode}' to '{mode}'")
    
    def add(self, message: str, user_id: str = None, metadata: Dict = None, **kwargs) -> str:
        """
        Override add method to conditionally include strength metadata.
        
        Args:
            message (str): The message/memory to store
            user_id (str, optional): User identifier
            metadata (Dict, optional): Additional metadata
            **kwargs: Additional arguments passed to parent class
            
        Returns:
            str: Memory ID
        """
        # In standard mode, use default Mem0 behavior
        if self.memory_mode == "standard":
            return super().add(message, user_id=user_id, metadata=metadata, **kwargs)
        
        # In ebbinghaus mode, add strength metadata
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Prepare ebbinghaus metadata
        ebbinghaus_metadata = {
            "created_at": current_time,
            "last_accessed": current_time,
            "memory_strength": self.config["initial_strength"],
            "access_count": 0,
            "mode": "ebbinghaus"
        }
        
        # Merge with any existing metadata
        if metadata:
            ebbinghaus_metadata.update(metadata)
        
        return super().add(message, user_id=user_id, metadata=ebbinghaus_metadata, **kwargs)
    
    def calculate_retention(self, memory_metadata: Dict) -> float:
        """
        Calculate memory retention based on Ebbinghaus formula.
        
        Args:
            memory_metadata (Dict): Memory metadata containing timestamps and strength
            
        Returns:
            float: Retention value (0.0 to 1.0)
        """
        # In standard mode, always return perfect retention
        if self.memory_mode == "standard":
            return 1.0
        
        # Check if memory has required metadata for Ebbinghaus calculation
        if not all(key in memory_metadata for key in ["created_at", "memory_strength"]):
            # If memory doesn't have Ebbinghaus metadata, treat as standard
            return 1.0
        
        try:
            # Parse timestamps
            created_at = datetime.fromisoformat(memory_metadata["created_at"].replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            
            # Calculate time elapsed in hours
            time_elapsed = (current_time - created_at).total_seconds() / 3600
            
            # Get memory strength
            strength = memory_metadata.get("memory_strength", self.config["initial_strength"])
            
            # Apply Ebbinghaus forgetting curve: R(t) = e^(-t/(24*S))
            # Where R(t) is retention, t is time in hours, S is strength factor
            retention = math.exp(-time_elapsed / (24 * strength))
            
            return max(retention, 0.0)  # Ensure non-negative
            
        except (ValueError, TypeError) as e:
            print(f"Error calculating retention: {e}")
            return 1.0  # Default to perfect retention on error
    
    def update_memory_strength(self, memory_id: str, boost: bool = True) -> None:
        """
        Update memory strength based on time and retrieval.
        
        Args:
            memory_id (str): ID of the memory to update
            boost (bool): Whether to apply retrieval boost
        """
        # In standard mode, skip strength updates
        if self.memory_mode == "standard":
            return
        
        try:
            # Get memory details
            memory_details = self.get(memory_id)
            if not memory_details or "metadata" not in memory_details:
                return
            
            metadata = memory_details["metadata"]
            
            # Skip if not an Ebbinghaus memory
            if metadata.get("mode") != "ebbinghaus":
                return
            
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Update last accessed time
            metadata["last_accessed"] = current_time
            
            # Increment access count
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            
            # Apply retrieval boost if requested
            if boost:
                current_strength = metadata.get("memory_strength", self.config["initial_strength"])
                boost_amount = self.config["retrieval_boost"]
                new_strength = min(current_strength + boost_amount, 1.0)  # Cap at 1.0
                metadata["memory_strength"] = new_strength
            
            # Update the memory with new metadata
            self.update(memory_id, metadata=metadata)
            
        except Exception as e:
            print(f"Error updating memory strength: {e}")
    
    def search(self, query: str, user_id: str = None, limit: int = 100, **kwargs) -> List[Dict]:
        """
        Override search method to be mode-aware.
        
        Args:
            query (str): Search query
            user_id (str, optional): User identifier
            limit (int): Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List[Dict]: Search results
        """
        # Get search results from parent class
        results = super().search(query, user_id=user_id, limit=limit, **kwargs)
        
        # In standard mode, return results as-is
        if self.memory_mode == "standard":
            return results
        
        # In ebbinghaus mode, filter by strength and update accessed memories
        if isinstance(results, dict) and "results" in results:
            memory_list = results["results"]
        else:
            memory_list = results if isinstance(results, list) else []
        
        filtered_results = []
        
        for memory in memory_list:
            if "metadata" not in memory:
                # Memory without metadata - include as-is
                filtered_results.append(memory)
                continue
            
            metadata = memory["metadata"]
            
            # Skip non-Ebbinghaus memories in Ebbinghaus mode
            if metadata.get("mode") != "ebbinghaus":
                filtered_results.append(memory)
                continue
            
            # Calculate current retention
            retention = self.calculate_retention(metadata)
            
            # Filter by minimum retention threshold
            if retention >= self.config["min_retention_threshold"]:
                # Add retention score to memory for debugging/monitoring
                memory["retention_score"] = retention
                filtered_results.append(memory)
                
                # Update memory strength due to retrieval
                if "id" in memory:
                    self.update_memory_strength(memory["id"], boost=True)
        
        # Sort by retention score if in Ebbinghaus mode
        if self.memory_mode == "ebbinghaus":
            filtered_results.sort(key=lambda x: x.get("retention_score", 1.0), reverse=True)
        
        # Return in same format as input
        if isinstance(results, dict) and "results" in results:
            return {"results": filtered_results}
        else:
            return filtered_results
    
    def forget_weak_memories(self, user_id: str = None, soft_delete: bool = True) -> Dict[str, int]:
        """
        Remove or archive memories below retention threshold.
        
        Args:
            user_id (str, optional): User identifier
            soft_delete (bool): If True, archive instead of delete
            
        Returns:
            Dict[str, int]: Statistics about forgotten memories
        """
        # In standard mode, skip forgetting entirely
        if self.memory_mode == "standard":
            return {"processed": 0, "forgotten": 0, "archived": 0}
        
        stats = {"processed": 0, "forgotten": 0, "archived": 0}
        
        try:
            # Get all memories for user
            all_memories = self.get_all(user_id=user_id)
            
            for memory in all_memories:
                stats["processed"] += 1
                
                if "metadata" not in memory:
                    continue
                
                metadata = memory["metadata"]
                
                # Skip non-Ebbinghaus memories
                if metadata.get("mode") != "ebbinghaus":
                    continue
                
                # Calculate retention
                retention = self.calculate_retention(metadata)
                
                # Check if memory should be forgotten
                if retention < self.config["min_retention_threshold"]:
                    memory_id = memory.get("id")
                    if memory_id:
                        if soft_delete:
                            # Archive by updating metadata
                            metadata["archived"] = True
                            metadata["archived_at"] = datetime.now(timezone.utc).isoformat()
                            self.update(memory_id, metadata=metadata)
                            stats["archived"] += 1
                        else:
                            # Hard delete
                            self.delete(memory_id)
                            stats["forgotten"] += 1
            
        except Exception as e:
            print(f"Error in forget_weak_memories: {e}")
        
        return stats
    
    def get_memory_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get statistics about memory usage and strength distribution.
        
        Args:
            user_id (str, optional): User identifier
            
        Returns:
            Dict[str, Any]: Memory statistics
        """
        stats = {
            "mode": self.memory_mode,
            "total_memories": 0,
            "ebbinghaus_memories": 0,
            "standard_memories": 0,
            "average_strength": 0.0,
            "average_retention": 0.0,
            "weak_memories": 0,
            "archived_memories": 0
        }
        
        try:
            # Get all memories
            all_memories = self.get_all(user_id=user_id)
            stats["total_memories"] = len(all_memories)
            
            if self.memory_mode == "standard":
                stats["standard_memories"] = stats["total_memories"]
                return stats
            
            # Analyze Ebbinghaus memories
            strengths = []
            retentions = []
            
            for memory in all_memories:
                if "metadata" not in memory:
                    stats["standard_memories"] += 1
                    continue
                
                metadata = memory["metadata"]
                
                if metadata.get("mode") == "ebbinghaus":
                    stats["ebbinghaus_memories"] += 1
                    
                    # Track strength
                    strength = metadata.get("memory_strength", 1.0)
                    strengths.append(strength)
                    
                    # Calculate retention
                    retention = self.calculate_retention(metadata)
                    retentions.append(retention)
                    
                    # Count weak memories
                    if retention < self.config["min_retention_threshold"]:
                        stats["weak_memories"] += 1
                    
                    # Count archived memories
                    if metadata.get("archived", False):
                        stats["archived_memories"] += 1
                else:
                    stats["standard_memories"] += 1
            
            # Calculate averages
            if strengths:
                stats["average_strength"] = sum(strengths) / len(strengths)
            if retentions:
                stats["average_retention"] = sum(retentions) / len(retentions)
                
        except Exception as e:
            print(f"Error getting memory statistics: {e}")
        
        return stats
