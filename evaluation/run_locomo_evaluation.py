"""
LOCOMO Evaluation Runner - Main execution script for comprehensive memory evaluation.

This is File 5 of the LOCOMO evaluation implementation - the main runner script
that orchestrates the complete evaluation pipeline comparing Ebbinghaus vs standard memory.

Usage:
    python run_locomo_evaluation.py [options]

Options:
    --model-path: Path to the local model (default: ./models/Llama-3.1-8B-Instruct)
    --dataset: Path to LOCOMO dataset (default: ./resources/dataset/locomo10_sample.json)  
    --output-dir: Output directory for results (default: ./evaluation/evaluation_output)
    --max-conversations: Max conversations to evaluate (default: 3, use -1 for all)
    --memory-modes: Comma-separated memory modes (default: standard,ebbinghaus)
    --quick-test: Run with 1 conversation and test model for quick validation
    --no-llm-judge: Disable LLM judge evaluation (faster)
    --answer-tokens: Max tokens for answer generation (default: 150)

Note: Analysis is now handled separately using combine_results.py after running evaluations.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up to project root
sys.path.append(str(project_root))

from evaluation.evaluation_config import EvaluationConfig
from evaluation.memory_evaluator import MemoryEvaluator

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"locomo_evaluation_{timestamp}.log"
    
    # Configure logging to both file and console with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive LOCOMO evaluation comparing Ebbinghaus vs standard memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full evaluation with default settings
    python run_locomo_evaluation.py
    
    # Quick test with small model
    python run_locomo_evaluation.py --quick-test
    
    # Evaluate specific memory modes only
    python run_locomo_evaluation.py --memory-modes standard --max-conversations 5
    
    # Custom configuration
    python run_locomo_evaluation.py \\
        --model-path ./models/mymodel \\
        --dataset ./my_dataset.json \\
        --max-conversations 10 \\
        --answer-tokens 200
        
    # Analyze results from multiple runs
    python combine_results.py standard_results.json ebbinghaus_results.json
        """
    )
    
    # Model settings
    parser.add_argument('--model-path', 
                       default='./models/Llama-3.1-8B-Instruct',
                       help='Path to the local model directory')
    
    # Dataset settings  
    parser.add_argument('--dataset',
                       default='./resources/dataset/locomo10_sample.json',
                       help='Path to LOCOMO dataset JSON file')
    
    parser.add_argument('--output-dir',
                       default='./evaluation/evaluation_output',
                       help='Output directory for results and logs')
    
    # Evaluation settings
    parser.add_argument('--max-conversations', type=int, default=3,
                       help='Maximum conversations to evaluate (-1 for all)')
    
    parser.add_argument('--memory-modes',
                       default='standard,ebbinghaus',
                       help='Comma-separated memory modes to evaluate')
    
    parser.add_argument('--answer-tokens', type=int, default=150,
                       help='Maximum tokens for answer generation')
    
    # Analysis settings - REMOVED: Analysis now handled by combine_results.py
    # parser.add_argument('--analyze-only',
    #                    help='Only run analysis on existing JSON results file')
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 1 conversation and test model')
    
    # Performance settings
    parser.add_argument('--no-llm-judge', action='store_true',
                       help='Disable LLM judge evaluation (faster)')
    
    return parser.parse_args()

def create_evaluation_config(args) -> EvaluationConfig:
    """Create evaluation configuration from arguments."""
    
    # Parse memory modes
    memory_modes = [mode.strip() for mode in args.memory_modes.split(',')]
    
    # Quick test mode overrides
    if args.quick_test:
        model_path = './models/TinyLlama-1.1B-Chat-v1.0'
        max_conversations = 1
        memory_modes = ['standard']  # Only test one mode
        print("[QUICK TEST MODE]:")
        print(f"   Using test model: {model_path}")
        print(f"   Max conversations: {max_conversations}")  
        print(f"   Memory modes: {memory_modes}")
        print()
    else:
        model_path = args.model_path
        max_conversations = args.max_conversations
    
    config = EvaluationConfig(
        local_model_path=model_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_conversations=max_conversations,
        answer_max_tokens=args.answer_tokens,
        memory_modes=memory_modes,
        config_modes={mode: "testing" for mode in memory_modes},
        use_llm_judge=not args.no_llm_judge,
        use_traditional_metrics=True
    )
    
    return config

def run_evaluation(config: EvaluationConfig, logger: logging.Logger):
    """Run the complete evaluation pipeline."""
    logger.info("[START] Starting LOCOMO Evaluation Pipeline")
    logger.info("="*60)
    
    # Configuration summary
    logger.info("[CONFIG] CONFIGURATION SUMMARY:")
    logger.info(f"   Model: {config.local_model_path}")
    logger.info(f"   Dataset: {config.dataset_path}")
    logger.info(f"   Output: {config.output_dir}")
    logger.info(f"   Max conversations: {config.max_conversations}")
    logger.info(f"   Memory modes: {config.memory_modes}")
    logger.info(f"   Answer tokens: {config.answer_max_tokens}")
    logger.info(f"   LLM judge: {config.use_llm_judge}")
    logger.info("")
    
    try:
        # Initialize evaluator
        logger.info("[INIT] Initializing Memory Evaluator...")
        evaluator = MemoryEvaluator(config)
        
        # Run evaluation
        logger.info("[RUN] Running evaluation...")
        evaluation_start = datetime.now()
        
        results = evaluator.run_evaluation()
        
        evaluation_duration = datetime.now() - evaluation_start
        logger.info(f"[SUCCESS] Evaluation completed in {evaluation_duration}")
        
        # Save evaluation results to JSON file
        logger.info("[SAVE] Saving evaluation results...")
        results_file = evaluator.save_results(results)
        logger.info(f"💾 Evaluation results saved to: {results_file}")
        
        # Print evaluation summary
        print_evaluation_summary(results, logger)
        
        return results
        
    except Exception as e:
        logger.error(f"[ERROR] Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def print_evaluation_summary(results, logger: logging.Logger):
    """Print evaluation results summary."""
    logger.info("")
    logger.info("="*60)
    logger.info("[SUMMARY] EVALUATION RESULTS SUMMARY")
    logger.info("="*60)
    
    # Handle dictionary format: {memory_mode: [ConversationEvaluationSummary, ...]}
    for memory_mode, mode_conversations in results.items():
        logger.info(f"\n[{memory_mode.upper()}] MEMORY MODE:")
        
        if not mode_conversations:
            logger.warning(f"   No results found for {memory_mode} mode")
            continue
        
        # Aggregate statistics across all conversations for this mode
        total_questions = sum(conv.total_questions for conv in mode_conversations)
        total_successful = sum(conv.successful_evaluations for conv in mode_conversations)
        
        if total_questions > 0:
            success_rate = (total_successful / total_questions) * 100
            avg_f1 = sum(conv.avg_f1_score * conv.total_questions for conv in mode_conversations) / total_questions
            avg_bleu = sum(conv.avg_bleu_1_score * conv.total_questions for conv in mode_conversations) / total_questions
            avg_judge = sum(conv.avg_llm_judge_score * conv.total_questions for conv in mode_conversations) / total_questions
            avg_gen_time = sum(conv.avg_generation_time * conv.total_questions for conv in mode_conversations) / total_questions
            avg_search_time = sum(conv.avg_memory_search_time * conv.total_questions for conv in mode_conversations) / total_questions
        else:
            success_rate = avg_f1 = avg_bleu = avg_judge = avg_gen_time = avg_search_time = 0.0
        
        logger.info(f"   [SUCCESS] Success rate: {success_rate:.1f}%")
        logger.info(f"   [QUESTIONS] Questions evaluated: {total_questions}")
        logger.info(f"   [CONVERSATIONS] Conversations: {len(mode_conversations)}")
        logger.info(f"   [F1] Average F1 Score: {avg_f1:.3f}")
        logger.info(f"   [BLEU] Average BLEU-1: {avg_bleu:.3f}")
        logger.info(f"   [LLM] Average LLM Judge: {avg_judge:.1f}")
        logger.info(f"   [TIME] Average generation time: {avg_gen_time:.2f}s")
        logger.info(f"   [SEARCH] Average search time: {avg_search_time:.3f}s")
        
        # Check for issues
        total_empty = sum(
            sum(1 for result in conv.results if result.generated_answer == "") 
            for conv in mode_conversations
        )
        if total_empty > 0:
            logger.warning(f"   [WARNING] Empty responses: {total_empty}/{total_questions}")
    
    logger.info("")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logging
    logger = setup_logging(args.output_dir)
    
    logger.info("[SYSTEM] LOCOMO Evaluation System - Main Runner")
    logger.info(f"[START] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    try:
        # Create configuration
        config = create_evaluation_config(args)
        
        # Run evaluation
        logger.info("[PIPELINE] Running evaluation pipeline")
        results = run_evaluation(config, logger)
        
        if results is None:
            logger.error("[ERROR] Evaluation pipeline failed")
            return 1
        
        logger.info("[SUCCESS] Evaluation completed successfully!")
        logger.info("ℹ️  Use combine_results.py to analyze results from multiple runs")
        
        logger.info(f"[COMPLETE] Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
        
    except KeyboardInterrupt:
        logger.info("[INTERRUPT] Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"[FATAL] Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
