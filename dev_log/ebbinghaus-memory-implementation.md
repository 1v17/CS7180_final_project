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

## Current Implementation Status

**Last Updated**: August 2, 2025  
**Overall Progress**: 3/7 Phases Complete (43%)

### ✅ Completed Phases
- **Phase 1**: Extend Memory with Mode Support ✅ 
- **Phase 2**: Mode-Aware Memory Operations ✅
- **Phase 3**: Conditional Forgetting Process ✅

### 🚧 Remaining Phases
- **Phase 4**: Mode-Aware Background Maintenance 🚧
- **Phase 5**: Spaced Repetition System 🚧
- **Phase 6**: Integration with ChatBot and Mode Control 🚧
- **Phase 7**: Testing and Optimization 🚧

### Key Achievements So Far

1. **Dual-Mode Architecture**: Successfully implemented backward-compatible system that can operate in both standard (perfect memory) and Ebbinghaus (forgetting) modes
2. **Runtime Mode Switching**: Users can switch between memory modes without restart
3. **Intelligent Metadata Management**: Conditional metadata addition based on mode
4. **Complete Ebbinghaus Implementation**: Full forgetting curve formula with configurable parameters
5. **Advanced Memory Operations**: Mode-aware search, strength updates, and forgetting processes
6. **Comprehensive Configuration**: Flexible config system with testing and production presets
7. **Robust Error Handling**: Graceful degradation and comprehensive validation

### Next Phase Priority

**Phase 4** (Background Maintenance) is the next logical step as it will automate the memory decay process and provide the foundation for the remaining phases.

## Key Classes and Methods Status

### EbbinghausMemory (extends Memory) ✅ IMPLEMENTED

✅ **Completed Methods:**
- `__init__()`: Initialize with decay parameters and memory mode
- `set_memory_mode()`: Switch between standard and ebbinghaus modes  
- `add()`: Add memory with conditional strength metadata (replaces `add_with_strength()`)
- `calculate_retention()`: Apply Ebbinghaus formula (bypass in standard mode)
- `update_memory_strength()`: Update strength based on time/retrieval (bypass in standard mode)
- `search()`: Search considering memory strength (use standard search in standard mode)
- `forget_weak_memories()`: Remove/archive weak memories (bypass in standard mode)

✅ **Additional Methods Implemented:**
- `get_memory_statistics()`: Comprehensive memory analytics and health metrics

### MemoryConfig ✅ IMPLEMENTED

✅ **Completed Features:**
- Centralized configuration management
- Multiple preset configurations (default, testing, production)
- Configuration validation and error handling
- Mode-specific parameter management

### SpacedRepetitionScheduler 🚧 PLANNED

🚧 **Planned Methods:**
- `get_next_review()`: Calculate optimal review time
- `get_memories_for_review()`: Find memories due for review

### MemoryMaintenanceScheduler 🚧 PLANNED

🚧 **Planned Methods:**
- `start()`: Start background tasks
- `update_all_strengths()`: Batch update all memories
- `run_forgetting_process()`: Execute forgetting

## Configuration Parameters

```python
{
    "memory_mode": "standard",  # "standard" or "ebbinghaus"
    "forgetting_curve": {
        "enabled": False,               # Controlled by memory_mode
        "initial_strength": 1,              # Base strength
        "min_retention_threshold": 0.1,  # Minimum strength to keep
        "retrieval_boost": 0.5,         # Strength increase on retrieval
        "soft_delete": True,            # Archive vs delete
        "maintenance_interval": 60,   # Seconds between strength updates (3600=1hr for production, 60=1min for testing)
    }
}
```


## Implementation Plan (Step-by-Step)

### Phase 1: Extend Memory with Mode Support ✅ COMPLETE

**Status**: ✅ Complete (August 2, 2025)  
**Goal**: Add memory strength tracking and mode switching to existing memories  
**Files**: `ebbinghaus_memory.py`, `memory_config.py`

✅ **Completed Tasks:**
1. ✅ Created `ebbinghaus_memory.py` that extends Mem0's Memory class
2. ✅ Added `memory_mode` parameter to `__init__()` method with validation
3. ✅ Overrode `add()` method to conditionally include strength metadata:
   - In "standard" mode: uses standard Mem0 behavior
   - In "ebbinghaus" mode: adds comprehensive strength metadata:
     - `created_at`: timestamp
     - `last_accessed`: timestamp
     - `memory_strength`: float (0.0-1.0)
     - `access_count`: integer
     - `mode`: "ebbinghaus" identifier

4. ✅ Implemented `set_memory_mode()` method to switch modes dynamically
5. ✅ Implemented `calculate_retention()` method with mode checking:
   ```python
   if self.memory_mode == "standard":
       return 1.0  # Perfect retention in standard mode
   else:
       return e^(-time_elapsed_hours / (24 * strength))
   ```

**Additional Features Implemented:**
- ✅ Comprehensive configuration system with `MemoryConfig` class
- ✅ Three preset configurations (default, testing, production)
- ✅ Full backward compatibility with existing Mem0 code
- ✅ Robust error handling and validation
- ✅ Advanced memory analytics with `get_memory_statistics()` method

### Phase 2: Mode-Aware Memory Operations ✅ COMPLETE

**Status**: ✅ Complete (August 2, 2025)  
**Goal**: Update memory operations to respect the current mode

✅ **Completed Tasks:**
1. ✅ Added `update_memory_strength()` method with mode checking:
   - In "standard" mode: skips strength updates
   - In "ebbinghaus" mode: applies decay and retrieval boosts
   - Updates `last_accessed` timestamp and `access_count`

2. ✅ Modified `search()` method to be mode-aware:
   - In "standard" mode: uses standard Mem0 search
   - In "ebbinghaus" mode: filters by retention threshold and updates accessed memories
   - Automatically strengthens memories when retrieved
   - Sorts results by retention score while maintaining format compatibility

**Additional Features Implemented:**
- ✅ Intelligent search result filtering based on retention scores
- ✅ Automatic memory strengthening during search operations
- ✅ Comprehensive retention-based result ranking

### Phase 3: Conditional Forgetting Process ✅ COMPLETE

**Status**: ✅ Complete (August 2, 2025)  
**Goal**: Only apply forgetting in ebbinghaus mode

✅ **Completed Tasks:**
1. ✅ Implemented `forget_weak_memories()` method with mode checking:
   - In "standard" mode: skips forgetting entirely
   - In "ebbinghaus" mode: applies sophisticated forgetting logic
   - Supports both soft delete (archiving) and hard delete
   - Returns detailed statistics about forgetting operations
   - Batch processes multiple memories efficiently

2. ✅ Added comprehensive mode validation in configuration:
   - Validates memory modes during initialization
   - Ensures configuration consistency
   - Provides clear error messages for invalid modes

**Additional Features Implemented:**
- ✅ Configurable forgetting thresholds
- ✅ Memory archiving system (soft delete option)
- ✅ Detailed forgetting operation statistics
- ✅ Batch processing for efficient memory cleanup

### Phase 4: Mode-Aware Background Maintenance 🚧 PLANNED

**Status**: 🚧 Not Started  
**Goal**: Automate memory maintenance only when needed

**Planned Tasks:**
1. Create `memory_scheduler.py` with mode awareness and configurable intervals:
   - Only start background tasks in "ebbinghaus" mode
   - Disable all scheduling in "standard" mode
   - Use `maintenance_interval` from config for update frequency
   - Allow dynamic mode switching without restart
   - Support different intervals for testing (short) vs production (long)

2. Add `set_maintenance_interval()` method for runtime adjustment:
   - Useful for switching between test/production without restart
   - Automatically restart scheduler with new interval

**Notes**: Foundation is ready from Phase 1-3 implementation. The `MemoryConfig` class already includes `maintenance_interval` parameter for this phase.

### Phase 5: Spaced Repetition System 🚧 PLANNED

**Status**: 🚧 Not Started  
**Goal**: Identify memories that need reinforcement

**Planned Tasks:**
1. Implement `SpacedRepetitionScheduler` class:
   - Adjust based on memory strength
   - Generate review reminders

2. Add `get_memories_for_review()` method

### Phase 6: Integration with ChatBot and Mode Control 🚧 PLANNED

**Status**: 🚧 Not Started  
**Goal**: Allow chatbot to switch memory modes

**Planned Tasks:**
1. Update `chatbot.py` to:
   - Initialize with desired memory mode
   - Add `set_memory_mode()` method for runtime switching
   - Display current memory mode in status/info commands

2. Add user commands for mode switching:
   - `/memory_mode standard` - Switch to standard memory
   - `/memory_mode ebbinghaus` - Switch to forgetting mode
   - `/memory_status` - Show current mode and memory statistics

**Notes**: The `EbbinghausMemory` class already provides `get_memory_statistics()` method for detailed memory analytics that can be used in the status command.

### Phase 7: Testing and Optimization 🚧 PLANNED

**Status**: 🚧 Not Started  
**Goal**: Ensure system works correctly

**Planned Tasks:**
1. Create test cases for:
   - Memory decay over time
   - Retrieval strengthening
   - Forgetting process
   - Spaced repetition scheduling

2. Add monitoring and analytics

**Notes**: Test files structure is already created (`tests/` directory with placeholder files). The implemented `EbbinghausMemory` class includes comprehensive error handling and analytics methods to support testing.

## File Structure Status

```
project/
├── chatbot.py              # Existing chatbot (to be updated in Phase 6)
├── ebbinghaus_memory.py    # ✅ Extended memory class with forgetting (COMPLETE)
├── memory_config.py        # ✅ Configuration settings (COMPLETE)
├── memory_scheduler.py     # 🚧 Background maintenance tasks (PLANNED - Phase 4)
├── spaced_repetition.py    # 🚧 Spaced repetition logic (PLANNED - Phase 5)
├── tests/                  # 🚧 Test infrastructure exists, tests to be written (Phase 7)
│   ├── __init__.py         # ✅ Created
│   ├── test_chatbot.py     # ✅ Created (empty)
│   ├── test_ebbinghaus_memory.py  # ✅ Created (empty)
│   ├── test_integration.py # ✅ Created (empty)
│   └── test_memory_config.py # ✅ Created (empty)
├── utils/
│   └── download_model.py   # ✅ Utility for model management
├── models/                 # ✅ Local model storage
├── resources/              # ✅ Dataset and resources
├── dev_log/                # ✅ Development documentation
│   ├── ebbinghaus-memory-implementation.md  # ✅ This file
│   └── phase1-implementation.md  # ✅ Detailed Phase 1 completion report
└── README.md               # ✅ Project documentation
```

### Implementation Files Status
- ✅ **ebbinghaus_memory.py**: Core implementation complete with all planned features
- ✅ **memory_config.py**: Configuration system complete with presets and validation
- 🚧 **memory_scheduler.py**: Not created yet (Phase 4)
- 🚧 **spaced_repetition.py**: Not created yet (Phase 5)
- 🚧 **Test implementations**: Files exist but tests need to be written (Phase 7)

