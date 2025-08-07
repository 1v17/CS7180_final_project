#!/usr/bin/env python3
"""
Results Combiner - Combine separate memory mode evaluation results.

This script combines results from separate standard and ebbinghaus evaluation runs
into a single comprehensive analysis.

Usage:
    python combine_results.py standard_results.json ebbinghaus_results.json [output_name]
"""

import sys
import json
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation.results_analyzer import EvaluationAnalyzer


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_results_file(filepath: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load results from a JSON file."""
    try:
        logger.info(f"Loading results from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine the memory mode from the data
        memory_modes = list(data.keys())
        logger.info(f"Found memory modes: {memory_modes}")
        
        return data
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        raise


def combine_results(standard_data: Dict[str, Any], ebbinghaus_data: Dict[str, Any], 
                   logger: logging.Logger) -> Dict[str, Any]:
    """Combine standard and ebbinghaus results into a single dataset."""
    logger.info("Combining results from both memory modes...")
    
    # Create combined results dictionary
    combined_results = {}
    
    # Add standard results
    if 'standard' in standard_data:
        combined_results['standard'] = standard_data['standard']
        logger.info(f"Added standard mode results: {len(standard_data['standard'])} conversations")
    else:
        logger.warning("No standard mode data found in first file")
    
    # Add ebbinghaus results
    if 'ebbinghaus' in ebbinghaus_data:
        combined_results['ebbinghaus'] = ebbinghaus_data['ebbinghaus']
        logger.info(f"Added ebbinghaus mode results: {len(ebbinghaus_data['ebbinghaus'])} conversations")
    else:
        logger.warning("No ebbinghaus mode data found in second file")
    
    # Validate that we have data for both modes
    if len(combined_results) != 2:
        logger.error(f"Expected 2 memory modes, got {len(combined_results)}: {list(combined_results.keys())}")
        raise ValueError("Could not find both standard and ebbinghaus results")
    
    # Validate that both modes have the same number of conversations
    standard_count = len(combined_results['standard']) if 'standard' in combined_results else 0
    ebbinghaus_count = len(combined_results['ebbinghaus']) if 'ebbinghaus' in combined_results else 0
    
    if standard_count != ebbinghaus_count:
        logger.warning(f"Conversation count mismatch: standard={standard_count}, ebbinghaus={ebbinghaus_count}")
    
    logger.info("✅ Successfully combined results from both memory modes")
    return combined_results


def save_combined_results(combined_data: Dict[str, Any], output_path: str, 
                         logger: logging.Logger) -> str:
    """Save combined results to a new JSON file."""
    logger.info(f"Saving combined results to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Combined results saved successfully")
    return output_path


def run_combined_analysis(combined_data: Dict[str, Any], dataset_name: str, 
                         output_dir: str, logger: logging.Logger):
    """Run comprehensive analysis on the combined results."""
    logger.info("🔬 Running comprehensive analysis on combined results...")
    
    try:
        # Initialize analyzer
        analyzer = EvaluationAnalyzer()
        
        # Run analysis
        analysis_start = datetime.now()
        report = analyzer.analyze_results(combined_data, dataset_name)
        analysis_duration = datetime.now() - analysis_start
        
        logger.info(f"✅ Analysis completed in {analysis_duration}")
        
        # Print summary
        analyzer.print_summary_report(report)
        
        # Save analysis report
        report_path = analyzer.save_report(report, output_dir)
        logger.info(f"📊 Analysis report saved to: {report_path}")
        
        # Export category statistics to CSV
        logger.info("📈 Exporting category statistics to CSV...")
        csv_path = analyzer.export_category_stats_to_csv(combined_data, output_dir)
        logger.info(f"📊 CSV statistics saved to: {csv_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Combine separate memory mode evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Combine two result files
    python combine_results.py standard_results.json ebbinghaus_results.json
    
    # Combine with custom output name
    python combine_results.py results1.json results2.json --output combined_comparison
    
    # Skip analysis (just combine files)
    python combine_results.py results1.json results2.json --no-analysis
        """
    )
    
    parser.add_argument('standard_file', help='JSON file with standard mode results')
    parser.add_argument('ebbinghaus_file', help='JSON file with ebbinghaus mode results')
    parser.add_argument('--output', '-o', help='Output filename prefix (default: combined_results)')
    parser.add_argument('--output-dir', default='./evaluation/evaluation_output', 
                       help='Output directory (default: ./evaluation/evaluation_output)')
    parser.add_argument('--no-analysis', action='store_true', 
                       help='Skip analysis, only combine files')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("🔗 LOCOMO Results Combiner")
    logger.info(f"📁 Standard results: {args.standard_file}")
    logger.info(f"📁 Ebbinghaus results: {args.ebbinghaus_file}")
    logger.info("")
    
    try:
        # Load both result files
        standard_data = load_results_file(args.standard_file, logger)
        ebbinghaus_data = load_results_file(args.ebbinghaus_file, logger)
        
        # Combine results
        combined_data = combine_results(standard_data, ebbinghaus_data, logger)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate output filename
        if args.output:
            output_name = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"combined_results_{timestamp}"
        
        output_path = os.path.join(args.output_dir, f"{output_name}.json")
        
        # Save combined results
        save_combined_results(combined_data, output_path, logger)
        
        # Run analysis unless disabled
        if not args.no_analysis:
            dataset_name = "locomo10_sample.json"  # Default dataset name
            report = run_combined_analysis(combined_data, dataset_name, args.output_dir, logger)
            
            if report:
                logger.info("🎉 Combined analysis completed successfully!")
            else:
                logger.warning("⚠️ Results combined but analysis failed")
                return 1
        else:
            logger.info("⏭️ Skipping analysis as requested")
        
        logger.info("✅ Results combination completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
