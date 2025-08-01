# Ebbinghaus Forgetting Curve Implementation for Mem0 Chatbot

## Project Overview

Enhance the existing chatbot with human-like memory that naturally forgets information over time following the Ebbinghaus forgetting curve. This will make the chatbot's memory more efficient and realistic by:

- Automatically decaying memory strength over time
- Strengthening memories when retrieved (testing effect)
- Removing or archiving weak memories
- Implementing spaced repetition for important information

## Current Architecture

- **ChatBot Class**: Main chatbot using local LLM + Mem0 for memory
- **Memory System**: Mem0 with OpenAI/GPT-3.5-turbo for memory operations
- **Storage**: Vector database for semantic search

## Implementation Notes for Copilot

- **Start Simple**: Begin with basic decay formula, then add complexity
- **Preserve Compatibility**: Ensure existing chatbot functionality remains intact
- **Use Metadata**: Leverage Mem0's metadata system for strength tracking
- **Async Operations**: Use async for background tasks to avoid blocking
- **Configurable**: Make all parameters adjustable via configuration
- **Error Handling**: Gracefully handle memories without strength metadata
- **Performance**: Batch operations where possible

## Key Classes and Methods to Implement

### EbbinghausMemory (extends Memory)

- `__init__()`: Initialize with decay parameters
- `add_with_strength()`: Add memory with strength metadata
- `calculate_retention()`: Apply Ebbinghaus formula
- `update_memory_strength()`: Update strength based on time/retrieval
- `search_with_strength()`: Search considering memory strength
- `forget_weak_memories()`: Remove/archive weak memories

### SpacedRepetitionScheduler

- `get_next_review()`: Calculate optimal review time
- `get_memories_for_review()`: Find memories due for review

### MemoryMaintenanceScheduler

- `start()`: Start background tasks
- `update_all_strengths()`: Batch update all memories
- `run_forgetting_process()`: Execute forgetting

## Configuration Parameters

```python
{
    "forgetting_curve": {
        "enabled": True,
        "decay_rate": 0.5,              # Base decay rate
        "min_retention_threshold": 0.1,  # Minimum strength to keep
        "retrieval_boost": 0.3,         # Strength increase on retrieval
        "soft_delete": True,            # Archive vs delete
        "importance_weights": {         # Decay multipliers by importance
            "critical": 0.1,
            "high": 0.3,
            "normal": 0.5,
            "low": 0.8
        }
    }
}
```


## Implementation Plan (Step-by-Step)

### Phase 1: Extend Memory with Strength Metadata

**Goal**: Add memory strength tracking to existing memories

1. Create `ebbinghaus_memory.py` that extends Mem0's Memory class
2. Override `add()` method to include strength metadata:
   - `created_at`: timestamp
   - `last_accessed`: timestamp
   - `initial_strength`: float (0.0-1.0)
   - `current_strength`: float (0.0-1.0)
   - `access_count`: integer
   - `importance`: string (critical/high/normal/low)

3. Implement `calculate_retention()` method using Ebbinghaus formula:
   ```
   retention = initial_strength * e^(-time_elapsed_hours / (24 * strength_factor))
   ```

### Phase 2: Memory Strength Updates

**Goal**: Update memory strength based on time decay and retrieval

1. Add `update_memory_strength()` method that:
   - Calculates time elapsed since last access
   - Applies decay formula
   - Boosts strength if memory was retrieved
   - Updates metadata

2. Modify `search()` method to:
   - Update strength of retrieved memories
   - Filter out memories below threshold
   - Sort by combined relevance and strength scores

### Phase 3: Automatic Forgetting Process

**Goal**: Remove or archive weak memories

1. Implement `forget_weak_memories()` method:
   - Scan all memories and update strengths
   - Archive memories below threshold (soft delete)
   - Option for hard delete

2. Add configuration options:
   - `forgetting_threshold`: minimum strength to keep memory
   - `soft_delete`: archive vs permanent deletion
   - `decay_rate`: how fast memories decay

### Phase 4: Background Maintenance

**Goal**: Automate memory maintenance

1. Create `memory_scheduler.py` with:
   - Daily forgetting process
   - Hourly strength updates
   - Memory statistics tracking

2. Use threading or async for background tasks

### Phase 5: Spaced Repetition System

**Goal**: Identify memories that need reinforcement

1. Implement `SpacedRepetitionScheduler` class:
   - Calculate optimal review intervals
   - Adjust based on memory strength
   - Generate review reminders

2. Add `get_memories_for_review()` method

### Phase 6: Integration with ChatBot

**Goal**: Use enhanced memory in the chatbot

1. Update `chatbot.py` to use EbbinghausMemory
2. Add importance classification for different types of information
3. Implement context-aware memory retrieval

### Phase 7: Testing and Optimization

**Goal**: Ensure system works correctly

1. Create test cases for:
   - Memory decay over time
   - Retrieval strengthening
   - Forgetting process
   - Spaced repetition scheduling

2. Add monitoring and analytics

## File Structure

```
project/
├── chatbot.py              # Existing chatbot (to be updated)
├── ebbinghaus_memory.py    # Extended memory class with forgetting
├── memory_scheduler.py     # Background maintenance tasks
├── spaced_repetition.py    # Spaced repetition logic
├── memory_config.py        # Configuration settings
├── tests/
│   ├── test_decay.py      # Test memory decay
│   ├── test_retrieval.py  # Test retrieval strengthening
│   └── test_forgetting.py # Test forgetting process
└── README.md              # This file
```

