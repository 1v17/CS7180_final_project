# LOCOMO Evaluation Implementation Plan (Using Your Existing ChatBot)

## Overview
This plan creates an evaluation system to compare your Ebbinghaus forgetting curve memory implementation against standard Mem0 memory using the LOCOMO dataset. The evaluation will leverage your existing `chatbot.py` which already has local Llama model integration and Ebbinghaus memory support.

## Files to Create

### File 1: `evaluation_config.py`
**Purpose**: Configuration settings and utility functions for the evaluation

**Key Components**:
- `EvaluationConfig` class with dataset paths and local model settings
- `MetricsCalculator` class with F1, BLEU-1 calculation methods
- `LLMJudge` class using GPT-4o-mini for evaluation
- OPENAI_API_KEY is in the .env file 

**Key Settings**:
```python
local_model_path: str = "./models/Meta-Llama-3-8B-Instruct"  # Your model path
use_existing_chatbot: bool = True
temperature: float = 0.0
max_conversations: int = 3  # For testing
```

**Prmopts**
#### prompt for LLM-as-a-judge
```
prompt = f"""
Your task is to evaluate an answer to a question by comparing it with the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {predicted}

Please evaluate the generated answer on a scale of 1-100 based on:
1. Factual accuracy compared to the ground truth
2. Completeness of the answer
3. Relevance to the question

Consider the following:
- If the generated answer contains the same key information as the ground truth, it should score highly
- Minor differences in wording or format should not significantly impact the score
- If the answer is completely wrong or irrelevant, it should score very low
- If the answer is partially correct, give partial credit

Respond with only a number between 1 and 100.
"""
```

#### prompt for LLM
```
prompt = f"""
Based on the following conversation memories, please answer the question.

Memories:
{memory_context}

Question: {question}

Instructions:
- Provide a concise, factual answer based only on the information in the memories
- If the memories don't contain enough information to answer the question, say "Information not available"
- Focus on the most relevant details
- Keep the answer brief and to the point

Answer:"""
```

### File 2: `locomo_dataset_loader.py`
**Purpose**: Load and standardize LOCOMO dataset format

**Key Components**:
- `LOCOMODatasetLoader` class
- `load_conversations()` method to read JSON files from `resources/dataset/`, please use the `resources/dataset/locomo10_sample.json` dataset
- `_standardize_conversation()` method to normalize different LOCOMO formats
- Handle various JSON structures (conversations, messages, questions)

**Expected Input**: JSON files in `resources/dataset/` directory
**Expected Output**: Standardized conversation objects with messages and questions

### File 3: `memory_evaluator.py`
**Purpose**: Core evaluation logic comparing memory systems using your existing ChatBot

**Key Components**:
- `MemoryEvaluator` class
- `initialize_chatbots()` - Creates ChatBot instances for both standard and Ebbinghaus modes
- `populate_memory_system()` - Adds conversation messages through ChatBot interface
- `evaluate_question()` - Tests questions using ChatBot.chat() method
- `run_evaluation()` - Runs complete evaluation on dataset
- Integration with your existing `ChatBot` class

**ChatBot Integration**:
```python
# Standard memory chatbot
standard_chatbot = ChatBot(
    model_path="./models/Llama-3.1-8B-Instruct",
    memory_mode="standard",
    config_mode="testing"
)

# Ebbinghaus memory chatbot  
ebbinghaus_chatbot = ChatBot(
    model_path="./models/Llama-3.1-8B-Instruct",
    memory_mode="ebbinghaus", 
    config_mode="testing"
)
```

### File 4: `results_analyzer.py`
**Purpose**: Analyze results and generate reports

**Key Components**:
- `EvaluationAnalyzer` class
- `generate_summary_report()` - Creates comprehensive analysis
- `_calculate_metrics_by_memory_mode()` - Compare standard vs Ebbinghaus
- `_test_statistical_significance()` - T-tests for significance
- `save_report()` - Export results to JSON

### File 5: `run_locomo_evaluation.py`
**Purpose**: Main script that ties everything together

**Key Components**:
- Command-line interface with argparse
- Integration with your existing files:
  - `from chatbot import ChatBot`
  - `from ebbinghaus_memory import EbbinghausMemory`
  - `from memory_config import MemoryConfig`
- ChatBot initialization and management
- Main execution flow with proper cleanup
- Console output with summary results

## Step-by-Step Implementation

### Step 1: Create Configuration System
1. Create `evaluation_config.py`
2. Implement `EvaluationConfig` dataclass with settings that match your ChatBot
3. Add `MetricsCalculator` with F1 and BLEU-1 methods
4. Add `LLMJudge` class that uses GPT-4o-mini for evaluation
5. Configure for your existing model path and settings

**Key Methods for ChatBot Integration**:
```python
def judge_answer_with_chatbot(chatbot: ChatBot, question: str, answer: str, ground_truth: str) -> float:
    # import the judge_prompt from `evaluation_config.py`
    
    response = chatbot.chat(judge_prompt, user_id="judge", max_new_tokens=10)
    # Parse numeric score from response
```

### Step 2: Create Dataset Loader
1. Create `locomo_dataset_loader.py`
2. Implement `LOCOMODatasetLoader` class
3. Add `load_conversations()` to read JSON files from `resources/dataset/`
4. Add `_standardize_conversation()` to handle different LOCOMO formats
5. Test with sample data to ensure loading works

### Step 3: Create Memory Evaluator Using Your ChatBot
1. Create `memory_evaluator.py`
2. Import your existing `ChatBot` class
3. Implement `MemoryEvaluator` class
4. Add methods that leverage your ChatBot:
   - `initialize_chatbots()` - Create standard and Ebbinghaus ChatBot instances
   - `populate_chatbot_memory()` - Use `chatbot.chat()` to add conversation messages
   - `evaluate_question_with_chatbot()` - Use `chatbot.chat()` to generate answers
   - `run_evaluation()` - Main evaluation loop using ChatBot instances

**Key Integration Points**:
```python
# Populate memory by simulating conversation
for message in conversation_messages:
    if message['speaker'] == 'User1':
        chatbot.chat(message['content'], user_id=f"locomo_{conv_id}_User1")
    # Don't need assistant responses, just user inputs

# Generate answer to evaluation question
answer = chatbot.chat(question, user_id=f"locomo_{conv_id}_User1", max_new_tokens=100)
```

### Step 4: Create Results Analyzer
1. Create `results_analyzer.py`
2. Implement `EvaluationAnalyzer` class
3. Add statistical analysis methods
4. Add report generation with performance comparisons
5. Include significance testing (optional scipy dependency)

### Step 5: Create Main Runner Script
1. Create `run_locomo_evaluation.py`
2. Import your existing `ChatBot` class
3. Add command-line argument parsing (no API key needed)
4. Implement main execution flow:
   - Initialize ChatBot instances for both memory modes
   - Load dataset
   - Initialize evaluator
   - Run evaluation using ChatBot instances
   - Analyze results
   - Print summary
   - Cleanup ChatBot instances
5. Add error handling and logging

## Integration Points with Your Existing Code

### Direct ChatBot Usage
```python
from chatbot import ChatBot

# Your ChatBot already handles:
# - Local Llama model loading from configurable path
# - Ebbinghaus memory integration
# - Memory mode switching (standard vs ebbinghaus)
# - Conversation management with user_ids
```

### Memory Operations Through ChatBot
```python
# Adding memories (through conversation simulation)
response = chatbot.chat(message, user_id=user_id)

# Memory is automatically updated in chatbot.chat()
# Your existing forgetting process is already integrated

# Retrieving memories (for analysis)
memories = chatbot.get_memories(query, user_id=user_id)
```

## Expected Directory Structure
```
project/
├── models/
│   └── Llama-3.1-8B-Instruct/     # Your existing model
├── resources/dataset/                # LOCOMO JSON files
├── evaluation/
│   ├── evaluation_config.py             # NEW - Configuration
│   ├── locomo_dataset_loader.py         # NEW - Dataset loading
│   ├── memory_evaluator.py              # NEW - Core evaluation using ChatBot
│   ├── results_analyzer.py              # NEW - Results analysis
│   ├── run_locomo_evaluation.py         # NEW - Main script
|   └── evaluation_output/               # Generated results
├── chatbot.py                       # YOUR EXISTING FILE
├── ebbinghaus_memory.py             # YOUR EXISTING FILE
└── memory_config.py                 # YOUR EXISTING FILE

```

## Expected Output
```
LOCOMO EVALUATION SUMMARY (Using Your ChatBot)
============================================

Model Information:
  Model Path: ./models/Meta-Llama-3-8B-Instruct
  Answer Generation: ChatBot with Local Llama
  Judge Model: ChatBot with Local Llama
  Device: Auto (from your ChatBot config)

Dataset Statistics:
  Total Questions: 120
  Total Conversations: 3

Performance Comparison:

Standard Memory (via ChatBot):
  F1 Score: 0.4523 ± 0.2103
  LLM Judge: 67.2 ± 18.4
  Search Latency: 0.145s
  Generation Latency: 2.1s

Ebbinghaus Memory (via ChatBot):
  F1 Score: 0.4987 ± 0.2287
  LLM Judge: 71.8 ± 19.7
  Search Latency: 0.187s
  Generation Latency: 2.3s

Performance Differences (Ebbinghaus - Standard):
  F1 Score: +0.0464 (+10.3%)
  LLM Judge: +4.6 (+6.8%)
```

## Key Design Decisions

1. **Leverage Existing ChatBot**: Use your proven ChatBot class instead of creating new LLM client
2. **Memory Mode Comparison**: Create two ChatBot instances (standard vs ebbinghaus) for direct comparison
3. **Conversation Simulation**: Use `chatbot.chat()` calls to populate memory naturally
4. **Answer Generation**: Use `chatbot.chat()` for generating answers to evaluation questions
5. **Judging Integration**: Use separate ChatBot instance for consistent answer evaluation

## Implementation Order
1. Start with `evaluation_config.py` (foundation, integrate with your ChatBot)
2. Then `locomo_dataset_loader.py` (test data loading)
3. Then `memory_evaluator.py` (core logic using your ChatBot)
4. Then `results_analyzer.py` (analysis)
5. Finally `run_locomo_evaluation.py` (integration)

This approach maximizes reuse of your existing, working code while adding the evaluation framework on top. Your ChatBot becomes the engine that drives both memory population and answer generation for the evaluation.
