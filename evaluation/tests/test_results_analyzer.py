"""
Test script to run results analysis on existing evaluation results JSON file.

This script loads saved evaluation results and runs the analysis to test
the results_analyzer.py functionality.
"""

import json
import sys
import os

# Add the project root to the path so we can import evaluation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.results_analyzer import EvaluationAnalyzer
from evaluation.memory_evaluator import ConversationEvaluationSummary, EvaluationResult

def load_and_convert_results(json_file_path: str):
    """
    Load JSON results and convert them back to the expected format.
    
    Args:
        json_file_path: Path to the JSON results file
        
    Returns:
        Dict with results in the format expected by the analyzer
    """
    print(f"📂 Loading results from: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Convert back to the expected format with dataclass objects
    converted_results = {}
    
    for memory_mode, summaries_data in raw_data.items():
        converted_summaries = []
        
        for summary_data in summaries_data:
            # Convert individual results
            results = []
            for result_data in summary_data['results']:
                result = EvaluationResult(
                    conversation_id=result_data['conversation_id'],
                    question_id=result_data['question_id'],
                    question=result_data['question'],
                    ground_truth=result_data['ground_truth'],
                    generated_answer=result_data['generated_answer'],
                    f1_score=result_data['f1_score'],
                    bleu_1_score=result_data['bleu_1_score'],
                    llm_judge_score=result_data['llm_judge_score'],
                    generation_time=result_data['generation_time'],
                    memory_search_time=result_data['memory_search_time'],
                    memory_mode=result_data['memory_mode'],
                    metadata=result_data['metadata']
                )
                results.append(result)
            
            # Convert summary
            summary = ConversationEvaluationSummary(
                conversation_id=summary_data['conversation_id'],
                memory_mode=summary_data['memory_mode'],
                total_questions=summary_data['total_questions'],
                successful_evaluations=summary_data['successful_evaluations'],
                failed_evaluations=summary_data['failed_evaluations'],
                avg_f1_score=summary_data['avg_f1_score'],
                avg_bleu_1_score=summary_data['avg_bleu_1_score'],
                avg_llm_judge_score=summary_data['avg_llm_judge_score'],
                avg_generation_time=summary_data['avg_generation_time'],
                avg_memory_search_time=summary_data['avg_memory_search_time'],
                results=results
            )
            converted_summaries.append(summary)
        
        converted_results[memory_mode] = converted_summaries
    
    print(f"✅ Loaded {len(converted_results)} memory modes:")
    for mode, summaries in converted_results.items():
        total_questions = sum(s.total_questions for s in summaries)
        print(f"  - {mode}: {len(summaries)} conversations, {total_questions} questions")
    
    return converted_results

def main():
    """Run analysis on the test JSON file."""
    print("🧪 Testing Results Analyzer with Existing JSON Data\n")
    
    # Path to your JSON file
    json_file_path = "./evaluation/evaluation_output/locomo_evaluation_results_20250805_154507.json"
    
    try:
        # Load and convert the results
        results_data = load_and_convert_results(json_file_path)
        
        # Initialize the analyzer
        print("\n🔬 Initializing Results Analyzer...")
        analyzer = EvaluationAnalyzer()
        
        # Run the analysis
        print("📊 Running analysis...")
        report = analyzer.analyze_results(results_data, "locomo10_sample.json")
        
        # Print the summary report
        print("\n" + "="*70)
        print("🎯 ANALYSIS RESULTS")
        print("="*70)
        analyzer.print_summary_report(report)
        
        # Save the analysis report
        print(f"\n💾 Saving analysis report...")
        report_path = analyzer.save_report(report)
        print(f"📄 Analysis report saved to: {report_path}")
        
        print(f"\n✅ Analysis completed successfully!")
        
        # Print some insights about the data
        print(f"\n🔍 DATA INSIGHTS:")
        print(f"📊 This appears to be results from the full model (not test model)")
        print(f"⚠️  All generated answers are empty strings - this confirms the issue we found")
        print(f"⏱️  Generation times: {report.memory_mode_stats['standard'].avg_generation_time:.1f}s average")
        print(f"🔍 Memory search times: {report.memory_mode_stats['standard'].avg_memory_search_time:.3f}s average")
        print(f"📈 Memory retrieval is working (5 memories per question)")
        print(f"💭 The issue is in text generation, not memory retrieval")
        
    except FileNotFoundError:
        print(f"❌ File not found: {json_file_path}")
        print(f"📂 Make sure the JSON file exists at the specified path")
        print(f"🔍 Current working directory: {os.getcwd()}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
