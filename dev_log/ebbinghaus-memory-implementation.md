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

- `__init__()`: Initialize with decay parameters and memory mode
- `set_memory_mode()`: Switch between standard and ebbinghaus modes
- `add_with_strength()`: Add memory with strength metadata (only in ebbinghaus mode)
- `calculate_retention()`: Apply Ebbinghaus formula (bypass in standard mode)
- `update_memory_strength()`: Update strength based on time/retrieval (bypass in standard mode)
- `search_with_strength()`: Search considering memory strength (use standard search in standard mode)
- `forget_weak_memories()`: Remove/archive weak memories (bypass in standard mode)

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
    "memory_mode": "standard",  # "standard" or "ebbinghaus"
    "forgetting_curve": {
        "enabled": False,               # Controlled by memory_mode
        "decay_rate": 0.5,              # Base decay rate
        "min_retention_threshold": 0.1,  # Minimum strength to keep
        "retrieval_boost": 0.3,         # Strength increase on retrieval
        "soft_delete": True,            # Archive vs delete
        "maintenance_interval": 3600,   # Seconds between strength updates (3600=1hr for production, 60=1min for testing)
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

### Phase 1: Extend Memory with Mode Support

**Goal**: Add memory strength tracking and mode switching to existing memories

1. Create `ebbinghaus_memory.py` that extends Mem0's Memory class
2. Add `memory_mode` parameter to `__init__()` method
3. Override `add()` method to conditionally include strength metadata:
   - In "standard" mode: use standard Mem0 behavior
   - In "ebbinghaus" mode: add strength metadata:
     - `created_at`: timestamp
     - `last_accessed`: timestamp
     - `initial_strength`: float (0.0-1.0)
     - `current_strength`: float (0.0-1.0)
     - `access_count`: integer
     - `importance`: string (critical/high/normal/low)

4. Implement `set_memory_mode()` method to switch modes dynamically
5. Implement `calculate_retention()` method with mode checking:
   ```
   if memory_mode == "standard":
       return 1.0  # Always standard retention
   else:
       return initial_strength * e^(-time_elapsed_hours / (24 * strength_factor))
   ```

### Phase 2: Mode-Aware Memory Operations

**Goal**: Update memory operations to respect the current mode

1. Add `update_memory_strength()` method with mode checking:
   - In "standard" mode: skip strength updates
   - In "ebbinghaus" mode: apply decay and retrieval boosts

2. Modify `search()` method to be mode-aware:
   - In "standard" mode: use standard Mem0 search
   - In "ebbinghaus" mode: filter by strength and update accessed memories

### Phase 3: Conditional Forgetting Process

**Goal**: Only apply forgetting in ebbinghaus mode

1. Implement `forget_weak_memories()` method with mode checking:
   - In "standard" mode: skip forgetting entirely
   - In "ebbinghaus" mode: apply forgetting logic

2. Add mode validation in configuration

### Phase 4: Mode-Aware Background Maintenance

**Goal**: Automate memory maintenance only when needed

1. Create `memory_scheduler.py` with mode awareness and configurable intervals:
   - Only start background tasks in "ebbinghaus" mode
   - Disable all scheduling in "standard" mode
   - Use `maintenance_interval` from config for update frequency
   - Allow dynamic mode switching without restart
   - Support different intervals for testing (short) vs production (long)

2. Add `set_maintenance_interval()` method for runtime adjustment:
   - Useful for switching between test/production without restart
   - Automatically restart scheduler with new interval

### Phase 5: Spaced Repetition System

**Goal**: Identify memories that need reinforcement

1. Implement `SpacedRepetitionScheduler` class:
   - Calculate optimal review intervals
   - Adjust based on memory strength
   - Generate review reminders

2. Add `get_memories_for_review()` method

### Phase 6: Integration with ChatBot and Mode Control

**Goal**: Allow chatbot to switch memory modes

1. Update `chatbot.py` to:
   - Initialize with desired memory mode
   - Add `set_memory_mode()` method for runtime switching
   - Display current memory mode in status/info commands

2. Add user commands for mode switching:
   - `/memory_mode standard` - Switch to standard memory
   - `/memory_mode ebbinghaus` - Switch to forgetting mode
   - `/memory_status` - Show current mode and memory statistics

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
└── README.md
```

