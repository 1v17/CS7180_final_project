# Ebbinghaus Memory Chatbot

An intelligent chatbot that implements human-like memory patterns using the Ebbinghaus forgetting curve. This project extends traditional AI memory systems to include realistic memory decay, strengthening through retrieval, and soft archiving of forgotten memories.

## Overview

This chatbot combines a local Large Language Model (LLM) with an innovative memory system that mimics human memory patterns. Unlike traditional AI systems with perfect memory, this chatbot:

- **Naturally forgets** information over time following the Ebbinghaus forgetting curve
- **Strengthens memories** when they are retrieved or discussed again
- **Archives weak memories** rather than permanently deleting them
- **Supports dual modes** - traditional perfect memory and realistic Ebbinghaus memory

## Features

### Memory Modes
- **Standard Mode**: Traditional perfect memory (default Mem0 behavior)
- **Ebbinghaus Mode**: Realistic memory with decay, strengthening, and forgetting

### Key Capabilities
- Interactive command-line chatbot interface
- Configurable memory decay parameters
- Memory statistics and health monitoring
- Soft delete system with memory restoration

### Model Support
- Uses local LLM models
- Easy model switching - just change the model path
- Supports any Hugging Face transformers-compatible model

## Requirements

**Python**: 3.10.18 (tested version)

## Installation & Setup

### Step 1: Install Miniconda

If you don't have Miniconda installed:

1. Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Install following the instructions for your operating system
3. Restart your terminal/command prompt

### Step 2: Create Conda Environment

```bash
# Create a new conda environment with Python 3.10.18
conda create -n ebbinghaus-chatbot python=3.10.18

# Activate the environment
conda activate ebbinghaus-chatbot
```

### Step 3: Clone and Setup Project

```bash
# Clone the repository
git clone <your-repository-url>
cd CS7180_final_project

# Install required packages
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Copy the example environment file and configure your API keys:

```bash
# Copy the example file
cp .env.example .env  # On Windows: copy .env.example .env
```

Edit the `.env` file and add your API keys:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here # optional
OPENAI_API_KEY=your_openai_api_key_here
```

**API Key Requirements**:
- **OpenAI API Key** (Required): Used by Mem0 for memory operations. Get one from [OpenAI's website](https://platform.openai.com/api-keys)
- **OpenRouter API Key** (Optional): Alternative API provider. Get one from [OpenRouter's website](https://openrouter.ai/keys)

> **Note**: At minimum, you need an OpenAI API key for the memory system to function properly. The OpenRouter API key is also specified but not currently used in this implementation - it's available because the Mem0 package uses an LLM to manage memory operations and supports OpenRouter models as an alternative to OpenAI.

### Step 5: Download the Default Model

```bash
# Download the default model (Microsoft DialoGPT-medium)
python utils/download_model.py
```

This will create the model files in `./models/test_local_model/`.

## Usage

### Basic Usage

```bash
# Activate your conda environment
conda activate ebbinghaus-chatbot

# Run the chatbot
python main.py
```

### Interactive Configuration

When you run the chatbot, you'll be prompted to choose:

1. **Memory Mode**:
   - Standard: Traditional perfect memory
   - Ebbinghaus: Realistic memory with forgetting

2. **Configuration** (for Ebbinghaus mode):
   - Testing: Fast decay (1-minute intervals, for development)
   - Production: Slower decay (1-hour intervals, for real use)
   - Standard: Default moderate decay settings

### Available Commands

During chat, you can use these commands:

- `/help` - Show available commands
- `/memory_status` - Display current memory statistics
- `/quit` or `quit` - Exit the chatbot

### Example Session

```
=== Ebbinghaus Memory ChatBot Configuration ===

Choose memory mode:
  1. Standard - Traditional perfect memory (default Mem0 behavior)
  2. Ebbinghaus - Memory with forgetting curve and decay over time

Enter your choice (1 or 2): 2

Choose scheduler configuration for Ebbinghaus mode:
  1. Testing - Fast decay, 1-minute maintenance interval (for development)
  2. Production - Slower decay, 1-hour maintenance interval (for real use)
  3. Standard - Default settings with moderate decay

Enter your choice (1, 2, or 3): 3

Selected: Ebbinghaus memory mode with default configuration

ChatBot ready! Type 'quit' to exit.

You: Hello, my name is Alice and I love hiking.
Bot: Nice to meet you, Alice! Hiking is a wonderful hobby...
```

## Using Different Models

To use a different model, please follow the steps:

1. Place your model files in a new folder under `./models/`
2. Ensure the model is compatible with Hugging Face transformers
3. **Important**: The folder name must look like a valid Hugging Face repository ID (e.g., `microsoft-DialoGPT-medium`, `my-custom-model`, etc.) because the Hugging Face library validates local paths as if they were Hub repository IDs, even when `local_files_only=True` is specified.
4. Update the model path in `main.py`:

```python
chatbot = ChatBot(
    model_path="./models/mymodel", 
    memory_mode=memory_mode, 
    config_mode=config_mode
)
```

## Evaluation System

This project includes a comprehensive evaluation system that compares the Ebbinghaus memory implementation against standard Mem0 memory using the LOCOMO dataset. The evaluation system provides statistical analysis and performance comparisons between different memory modes.

### Prerequisites for Evaluation

1. **Larger Model**: For reliable evaluation results, use a larger model like: `./models/TinyLlama-1.1B-Chat-v1.0`
   
   Note: small models like `microsoft/DialoGPT-medium` may generate empty responses. Follow the instructions for **Using Different Models** to add model into the /models folder.

2. **LOCOMO Dataset**: Sample dataset is included at `resources/dataset/locomo10_sample.json`, which contains part of the 10 conversations in the complete dataset. The full dataset of 10 conversations is `resources/dataset/locomo10.json`.

### Running Evaluations

#### Quick Test Mode
For a fast evaluation with minimal data:
```bash
python evaluation\run_locomo_evaluation.py --quick-test
```

#### Individual Memory Mode Evaluation
Due to database lock conflicts, memory modes should be evaluated separately:

```bash
# Evaluate standard memory mode
python evaluation\run_locomo_evaluation.py --model-path ./models/TinyLlama-1.1B-Chat-v1.0 --memory-modes standard

# Evaluate Ebbinghaus memory mode  
python evaluation\run_locomo_evaluation.py --model-path ./models/TinyLlama-1.1B-Chat-v1.0 --memory-modes ebbinghaus
```

#### Combine and Analyze Results
After running separate evaluations, combine the results for statistical analysis:

```bash
# Replace with your actual result file names
python evaluation\combine_results.py evaluation\evaluation_output\locomo_evaluation_results_TIMESTAMP1.json evaluation\evaluation_output\locomo_evaluation_results_TIMESTAMP2.json
```

#### Advanced Options
```bash
# Custom model and configuration
python evaluation\run_locomo_evaluation.py --model-path ./models/Llama-3.1-8B-Instruct --max-conversations 20 --memory-modes standard

# Use different dataset (if available)
python evaluation\run_locomo_evaluation.py --dataset-path ./resources/dataset/custom_dataset.json
```

### Evaluation Metrics

The evaluation system uses multiple metrics to assess performance:

1. **F1 Score**: Measures precision and recall of generated answers
2. **BLEU-1 Score**: Evaluates n-gram overlap with ground truth
3. **LLM Judge Score**: GPT-4o-mini rates answers on a 1-100 scale

## Configuration

### Memory Modes

**Standard Mode**:
- Perfect memory retention
- No forgetting or decay
- Compatible with existing Mem0 behavior

**Ebbinghaus Mode**:
- Time-based memory decay
- Retrieval strengthening
- Configurable forgetting parameters

### Configuration Presets

**Testing Configuration**:
```python
{
    "memory_mode": "ebbinghaus",
    "forgetting_curve": {
        "initial_strength": 1.0,
        "min_retention_threshold": 0.2,
        "retrieval_boost": 0.3,
        "decay_rate": 0.8,  # Fast decay
    }
}
```

**Production Configuration**:
```python
{
    "memory_mode": "ebbinghaus", 
    "forgetting_curve": {
        "initial_strength": 1.0,
        "min_retention_threshold": 0.1,
        "retrieval_boost": 0.5,
        "decay_rate": 0.3,  # Slower decay
    }
}
```

## Technical Details

### Ebbinghaus Forgetting Curve

The memory system implements a simplified exponential decay formula:

```
R = e^(-t/S)
```

Where:
- `R` = Retrievability (measure of how easy it is to retrieve information from memory)
- `S` = Stability of memory (determines how fast R falls over time without recall)
- `t` = Time elapsed
- `e` = Euler's number (natural logarithm base)

### Memory Strength System

- **Initial Strength**: 1.0 for new memories
- **Retrieval Boost**: +0.5 strength when accessed
- **Decay Rate**: Configurable exponential decay
- **Threshold**: Memories below retention threshold are archived

### Soft Delete Architecture

Instead of permanently deleting weak memories:
- Memories are prefixed with `[ARCHIVED]`
- Archived memories can be restored
- Provides memory statistics and analytics
- Maintains conversation continuity

## Acknowledgments

- **Mem0**: For the foundational memory system
- **Hugging Face**: For transformer models and tokenizers
- **OpenAI**: For GPT models used in memory operations
- **Hermann Ebbinghaus**: For the foundational research on memory and forgetting

## References

- [Ebbinghaus Forgetting Curve](https://en.wikipedia.org/wiki/Forgetting_curve)
- [Mem0 Documentation](https://mem0.ai/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

**CS7180 Final Project** - Northeastern University  
*Implementing Human-like Memory in AI Systems*