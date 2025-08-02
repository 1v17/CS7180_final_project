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
**Overall Progress**: 4/6 Phases Complete (67%)

### ✅ Completed Phases

- **Phase 1**: Extend Memory with Mode Support ✅
- **Phase 2**: Mode-Aware Memory Operations ✅
- **Phase 3**: Conditional Forgetting Process ✅
- **Phase 4**: Mode-Aware Background Maintenance ✅

### 🚧 Remaining Phases

- **Phase 5**: Integration with ChatBot and Mode Control 🚧 _(formerly Phase 6)_
- **Phase 6**: Testing and Optimization 🚧 _(formerly Phase 7)_

### Key Achievements So Far

1. **Dual-Mode Architecture**: Successfully implemented backward-compatible system that can operate in both standard (perfect memory) and Ebbinghaus (forgetting) modes
2. **Runtime Mode Switching**: Users can switch between memory modes without restart
3. **Intelligent Metadata Management**: Conditional metadata addition based on mode
4. **Complete Ebbinghaus Implementation**: Full forgetting curve formula with configurable parameters
5. **Advanced Memory Operations**: Mode-aware search, strength updates, and forgetting processes
6. **Comprehensive Configuration**: Flexible config system with testing and production presets
7. **Robust Error Handling**: Graceful degradation and comprehensive validation
8. **Automated Background Maintenance**: Intelligent scheduling system that only operates when needed

### Next Phase Priority

**Phase 5** (ChatBot Integration) is the next logical step as the core memory system and automated maintenance are now complete.

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

### MemoryMaintenanceScheduler ✅ IMPLEMENTED

✅ **Completed Methods:**

- `start()`: Start background tasks (only in ebbinghaus mode)
- `stop()`: Graceful shutdown with timeout handling
- `update_all_strengths()`: Batch update all memories
- `run_forgetting_process()`: Execute forgetting
- `set_maintenance_interval()`: Runtime configuration changes
- `get_status()`: Real-time status and statistics monitoring
- `force_maintenance()`: Manual maintenance trigger

✅ **Additional Features:**

- Mode-aware operation (only runs in ebbinghaus mode)
- Threading-based background operation
- Factory pattern for different scheduler configurations
- Comprehensive error handling and recovery
- Runtime configuration adjustments

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

### Phase 4: Mode-Aware Background Maintenance ✅ COMPLETE

**Status**: ✅ Complete (August 2, 2025)  
**Goal**: Automate memory maintenance only when needed  
**Files**: `memory_scheduler.py`, `demo_scheduler.py`

✅ **Completed Tasks:**

1. ✅ Created `MemoryMaintenanceScheduler` class with mode awareness:

   - Only starts background tasks in "ebbinghaus" mode
   - Disables all scheduling in "standard" mode
   - Uses `maintenance_interval` from config for update frequency
   - Supports dynamic mode switching without restart
   - Provides different intervals for testing vs production

2. ✅ Implemented comprehensive scheduling features:

   - Background threading for non-blocking operation
   - Graceful shutdown with timeout handling
   - Real-time status monitoring and statistics
   - Error resilience and recovery mechanisms
   - Runtime configuration adjustments

3. ✅ Added factory pattern for scheduler creation:
   - `create_scheduler()`: Standard configuration
   - `create_testing_scheduler()`: Fast intervals (60s) for development
   - `create_production_scheduler()`: Optimized intervals (3600s) for production

**Additional Features Implemented:**

- ✅ Intelligent maintenance loop with mode validation
- ✅ Batch processing of memory strength updates
- ✅ Comprehensive error handling and logging
- ✅ Performance metrics and operation tracking

### Phase 5: Integration with ChatBot and Mode Control 🚧 PLANNED

**Status**: 🚧 Not Started _(formerly Phase 6)_
**Goal**: Allow chatbot to switch memory modes

**Planned Tasks:**

1. Update `chatbot.py` to:

   - Initialize with desired memory mode
   - Add `set_memory_mode()` method for runtime switching
   - Display current memory mode in status/info commands
   - Integrate with `MemoryMaintenanceScheduler`

2. Add user commands for mode switching:
   - `/memory_mode standard` - Switch to standard memory
   - `/memory_mode ebbinghaus` - Switch to forgetting mode
   - `/memory_status` - Show current mode and memory statistics
   - `/memory_maintenance` - Force maintenance or view scheduler status

**Notes**: The `EbbinghausMemory` class and `MemoryMaintenanceScheduler` already provide comprehensive methods for integration and status reporting.

### Phase 6: Testing and Optimization 🚧 PLANNED

**Status**: 🚧 Not Started _(formerly Phase 7)_
**Goal**: Ensure system works correctly

**Planned Tasks:**

1. Create test cases for:

   - Memory decay over time
   - Retrieval strengthening
   - Forgetting process
   - Mode switching functionality
   - Background maintenance scheduling
   - Scheduler behavior across mode changes

2. Add monitoring and analytics

**Notes**: Test files structure is already created (`tests/` directory with placeholder files). The implemented classes include comprehensive error handling and analytics methods to support testing. The `demo_scheduler.py` provides testing patterns for scheduler functionality.

## File Structure Status

```
project/
├── chatbot.py              # Existing chatbot (to be updated in Phase 5)
├── ebbinghaus_memory.py    # ✅ Extended memory class with forgetting (COMPLETE)
├── memory_config.py        # ✅ Configuration settings (COMPLETE)
├── memory_scheduler.py     # ✅ Background maintenance tasks (COMPLETE - Phase 4)
├── tests/                  # 🚧 Test infrastructure exists, tests to be written (Phase 6)
│   ├── __init__.py         # ✅ Created
│   ├── test_chatbot.py     # ✅ Created (empty)
|   ├── demo_scheduler.py       # ✅ Scheduler testing and examples (COMPLETE)
│   ├── test_ebbinghaus_memory.py  # ✅ Created (empty)
│   ├── test_integration.py # ✅ Created (empty)
│   └── test_memory_config.py # ✅ Created (empty)
├── utils/
│   └── download_model.py   # ✅ Utility for model management
├── models/                 # ✅ Local model storage
├── resources/              # ✅ Dataset and resources
├── dev_log/                # ✅ Development documentation
│   ├── ebbinghaus-memory-implementation.md  # ✅ This file
│   ├── phase1to3-implementation.md  # ✅ Detailed Phase 1 to 3 completion report
│   └── phase4-implementation.md  # ✅ Detailed Phase 4 completion report
└── README.md               # ✅ Project documentation
```

### Implementation Files Status

- ✅ **ebbinghaus_memory.py**: Core implementation complete with all planned features
- ✅ **memory_config.py**: Configuration system complete with presets and validation
- ✅ **memory_scheduler.py**: Background maintenance scheduler complete with mode awareness
- 🚧 **Test implementations**: Files exist but tests need to be written (Phase 6)
