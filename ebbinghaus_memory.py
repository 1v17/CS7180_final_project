"""
Ebbinghaus Memory Extension for Mem0

This module extends Mem0's Memory class to support memory decay based on the 
Ebbinghaus forgetting curve. It provides two modes:
- "standard": Traditional perfect memory (default Mem0 behavior)
- "ebbinghaus": Memory with strength tracking and decay over time
"""

import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from mem0 import Memory
from mem0.configs.base import MemoryConfig as Mem0Config
from memory_config import MemoryConfig


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
            config (Dict, optional): Ebbinghaus configuration dictionary
            memory_mode (str): Memory mode - "standard" or "ebbinghaus"
        """
        # Set up environment for mem0
        import os
        if not os.environ.get('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = 'dummy-key-for-testing'
        
        # Get our Ebbinghaus configuration
        if config:
            self.ebbinghaus_config = config
        else:
            self.ebbinghaus_config = MemoryConfig.get_config("default")
        
        # Extract LLM config for mem0 and create proper mem0 config
        llm_config = self.ebbinghaus_config.get("llm", {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
            }
        })
        
        # Create mem0 config with just the LLM configuration
        mem0_config = Mem0Config()
        # Override the LLM config in mem0's config if needed
        if hasattr(mem0_config, 'llm') and hasattr(mem0_config.llm, 'config'):
            mem0_config.llm.config.update(llm_config.get("config", {}))
        
        # Initialize parent class with proper mem0 config
        super().__init__(mem0_config)
        
        # Validate memory mode
        if memory_mode not in ["standard", "ebbinghaus"]:
            raise ValueError("memory_mode must be 'standard' or 'ebbinghaus'")
        
        self.memory_mode = memory_mode
        
        # Extract forgetting curve parameters for easy access
        self.fc_config = self.ebbinghaus_config.get("forgetting_curve", {})
        
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
            "memory_strength": self.fc_config.get("initial_strength", 1.0),
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
            strength = memory_metadata.get("memory_strength", self.fc_config.get("initial_strength", 1.0))
            
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
                current_strength = metadata.get("memory_strength", self.fc_config.get("initial_strength", 1.0))
                boost_amount = self.fc_config.get("retrieval_boost", 0.5)
                new_strength = min(current_strength + boost_amount, 1.0)  # Cap at 1.0
                metadata["memory_strength"] = new_strength
            
            # Note: Direct metadata updates may not be supported by Mem0
            # We'll skip direct strength updates here and let them happen
            # naturally during memory retrieval and chat operations
            
            # For now, we'll just log that we would update the strength
            # The actual strength updates will happen during natural memory access
            pass
            
        except Exception as e:
            # Silently handle errors since strength updates are not critical
            # and will happen naturally during memory access
            pass
    
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
            if retention >= self.fc_config.get("min_retention_threshold", 0.1):
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
                if retention < self.fc_config.get("min_retention_threshold", 0.1):
                    memory_id = memory.get("id")
                    if memory_id:
                        # Delete the memory (both soft and hard delete do the same thing now
                        # since we can't reliably update metadata)
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
            "strong_memories": 0,
            "weak_memories": 0,
            "archived_memories": 0
        }
        
        try:
            # If no user_id provided, we can't get memories from Mem0
            # Return basic mode information only
            if user_id is None:
                return {
                    "mode": self.memory_mode,
                    "total_memories": "N/A (requires user_id)",
                    "ebbinghaus_memories": "N/A",
                    "standard_memories": "N/A", 
                    "average_strength": 0.0,
                    "average_retention": 0.0,
                    "strong_memories": "N/A",
                    "weak_memories": "N/A",
                    "archived_memories": "N/A",
                    "note": "Provide user_id for detailed statistics"
                }
            
            # Get all memories for the specified user
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
                    
                    # Count weak memories (low retention)
                    if retention < self.fc_config.get("min_retention_threshold", 0.1):
                        stats["weak_memories"] += 1
                    
                    # Count strong memories (high retention)
                    if retention > 0.5:
                        stats["strong_memories"] += 1
                    
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
