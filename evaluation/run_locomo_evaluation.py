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
    --analyze-only: Only run analysis on existing results (provide JSON path)
    --quick-test: Run with 1 conversation and test model for quick validation
    --no-llm-judge: Disable LLM judge evaluation (faster)
    --answer-tokens: Max tokens for answer generation (default: 150)
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
from evaluation.results_analyzer import EvaluationAnalyzer

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
    
    # Only analyze existing results
    python run_locomo_evaluation.py --analyze-only ./evaluation/evaluation_output/results.json
    
    # Custom configuration
    python run_locomo_evaluation.py \\
        --model-path ./models/mymodel \\
        --dataset ./my_dataset.json \\
        --max-conversations 10 \\
        --answer-tokens 200
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
    
    # Analysis settings
    parser.add_argument('--analyze-only',
                       help='Only run analysis on existing JSON results file')
    
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
        
        # Print evaluation summary
        print_evaluation_summary(results, logger)
        
        return results
        
    except Exception as e:
        logger.error(f"[ERROR] Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_analysis(results_or_path, dataset_name: str, logger: logging.Logger):
    """Run comprehensive analysis on evaluation results."""
    logger.info("[ANALYSIS] Starting Results Analysis...")
    
    try:
        # Initialize analyzer
        analyzer = EvaluationAnalyzer()
        
        # If path provided, load results
        if isinstance(results_or_path, str):
            logger.info(f"[LOAD] Loading results from: {results_or_path}")
            import json
            with open(results_or_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            # Convert to expected format (implementation depends on your data structure)
            results_data = raw_data  # Simplified for now
        else:
            # Use provided results directly
            results_data = results_or_path
            
        # Run analysis
        analysis_start = datetime.now()
        report = analyzer.analyze_results(results_data, dataset_name)
        analysis_duration = datetime.now() - analysis_start
        
        logger.info(f"[SUCCESS] Analysis completed in {analysis_duration}")
        
        # Print and save report
        print_analysis_summary(report, logger)
        report_path = analyzer.save_report(report)
        
        logger.info(f"[SAVE] Analysis report saved to: {report_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"[ERROR] Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def print_evaluation_summary(results, logger: logging.Logger):
    """Print evaluation results summary."""
    logger.info("")
    logger.info("="*60)
    logger.info("[SUMMARY] EVALUATION RESULTS SUMMARY")
    logger.info("="*60)
    
    for mode_results in results:
        mode = mode_results.memory_mode.upper()
        success_rate = (mode_results.successful_evaluations / mode_results.total_questions) * 100
        
        logger.info(f"[{mode}] MEMORY MODE:")
        logger.info(f"   [SUCCESS] Success rate: {success_rate:.1f}%")
        logger.info(f"   [QUESTIONS] Questions evaluated: {mode_results.total_questions}")
        logger.info(f"   [F1] Average F1 Score: {mode_results.avg_f1_score:.3f}")
        logger.info(f"   [BLEU] Average BLEU-1: {mode_results.avg_bleu_1_score:.3f}")
        logger.info(f"   [LLM] Average LLM Judge: {mode_results.avg_llm_judge_score:.1f}")
        logger.info(f"   [TIME] Average generation time: {mode_results.avg_generation_time:.2f}s")
        logger.info(f"   [SEARCH] Average search time: {mode_results.avg_memory_search_time:.3f}s")
        
        # Check for issues
        empty_count = sum(1 for result in mode_results.results if result.generated_answer == "")
        if empty_count > 0:
            logger.warning(f"   [WARNING] Empty responses: {empty_count}/{len(mode_results.results)}")
        
        logger.info("")

def print_analysis_summary(report, logger: logging.Logger):
    """Print analysis results summary.""" 
    logger.info("")
    logger.info("="*60)
    logger.info("[ANALYSIS] COMPREHENSIVE ANALYSIS RESULTS")
    logger.info("="*60)
    
    # Use the existing analyzer's print method
    from evaluation.results_analyzer import EvaluationAnalyzer
    analyzer = EvaluationAnalyzer()
    analyzer.print_summary_report(report)

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
        # Analysis-only mode
        if args.analyze_only:
            logger.info("[MODE] Running analysis-only mode")
            dataset_name = Path(args.dataset).name
            report = run_analysis(args.analyze_only, dataset_name, logger)
            
            if report:
                logger.info("[SUCCESS] Analysis-only mode completed successfully!")
            else:
                logger.error("[ERROR] Analysis failed")
                return 1
        
        # Full evaluation mode
        else:
            # Create configuration
            config = create_evaluation_config(args)
            
            # Run evaluation
            logger.info("[PIPELINE] Running full evaluation pipeline")
            results = run_evaluation(config, logger)
            
            if results is None:
                logger.error("[ERROR] Evaluation pipeline failed")
                return 1
            
            # Run analysis on results
            dataset_name = Path(config.dataset_path).name
            logger.info("[TRANSITION] Proceeding to analysis phase...")
            report = run_analysis(results, dataset_name, logger)
            
            if report:
                logger.info("[SUCCESS] Complete evaluation pipeline finished successfully!")
            else:
                logger.warning("[WARNING] Evaluation completed but analysis failed")
                return 0  # Evaluation succeeded even if analysis failed
        
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
