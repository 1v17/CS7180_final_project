"""
Results Analyzer for LOCOMO Evaluation

This module analyzes evaluation results and generates comprehensive reports
comparing standard vs Ebbinghaus memory performance.
"""

import json
import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
from collections import defaultdict

# Try to import scipy for statistical tests, but make it optional
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class MemoryModeStats:
    """Statistics for a single memory mode."""
    memory_mode: str
    total_questions: int
    successful_evaluations: int
    failed_evaluations: int
    success_rate: float
    
    # Metric averages and standard deviations
    avg_f1_score: float
    std_f1_score: float
    avg_bleu_1_score: float
    std_bleu_1_score: float
    avg_llm_judge_score: float
    std_llm_judge_score: float
    
    # Performance averages and standard deviations
    avg_generation_time: float
    std_generation_time: float
    avg_memory_search_time: float
    std_memory_search_time: float
    
    # Additional statistics
    median_f1_score: float
    median_bleu_1_score: float
    median_llm_judge_score: float
    
    # Per-conversation breakdown
    conversation_count: int
    avg_questions_per_conversation: float


@dataclass
class ComparisonResult:
    """Comparison between two memory modes."""
    baseline_mode: str
    comparison_mode: str
    
    # Differences (comparison - baseline)
    f1_score_diff: float
    f1_score_percent_change: float
    bleu_1_score_diff: float
    bleu_1_percent_change: float
    llm_judge_score_diff: float
    llm_judge_percent_change: float
    
    generation_time_diff: float
    generation_time_percent_change: float
    memory_search_time_diff: float
    memory_search_time_percent_change: float
    
    # Statistical significance (if scipy available)
    f1_p_value: Optional[float] = None
    bleu_1_p_value: Optional[float] = None
    llm_judge_p_value: Optional[float] = None
    generation_time_p_value: Optional[float] = None
    memory_search_time_p_value: Optional[float] = None
    
    # Effect sizes
    f1_effect_size: Optional[float] = None
    bleu_1_effect_size: Optional[float] = None
    llm_judge_effect_size: Optional[float] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    dataset_info: Dict[str, Any]
    memory_mode_stats: Dict[str, MemoryModeStats]
    comparisons: List[ComparisonResult]
    summary: Dict[str, Any]
    recommendations: List[str]


class EvaluationAnalyzer:
    """Analyzes evaluation results and generates comprehensive reports."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.logger = logging.getLogger(__name__)
        
    def analyze_results(self, results_data: Dict[str, List[Any]], 
                       dataset_filename: str = None) -> EvaluationReport:
        """
        Analyze evaluation results and generate comprehensive report.
        
        Args:
            results_data: Raw evaluation results by memory mode
            dataset_filename: Name of the dataset file used
            
        Returns:
            EvaluationReport: Comprehensive analysis report
        """
        self.logger.info("🔬 Analyzing evaluation results...")
        
        # Calculate statistics for each memory mode
        memory_mode_stats = {}
        for mode, summaries in results_data.items():
            stats = self._calculate_memory_mode_stats(mode, summaries)
            memory_mode_stats[mode] = stats
            self.logger.info(f"📊 {mode.upper()} mode: {stats.total_questions} questions, "
                           f"F1={stats.avg_f1_score:.3f}, BLEU={stats.avg_bleu_1_score:.3f}, "
                           f"Judge={stats.avg_llm_judge_score:.1f}")
        
        # Generate comparisons between memory modes
        comparisons = self._generate_comparisons(memory_mode_stats, results_data)
        
        # Generate dataset information
        dataset_info = self._generate_dataset_info(results_data, dataset_filename)
        
        # Generate summary and recommendations
        summary = self._generate_summary(memory_mode_stats, comparisons)
        recommendations = self._generate_recommendations(memory_mode_stats, comparisons)
        
        report = EvaluationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset_info=dataset_info,
            memory_mode_stats=memory_mode_stats,
            comparisons=comparisons,
            summary=summary,
            recommendations=recommendations
        )
        
        self.logger.info("✅ Analysis completed successfully!")
        return report
    
    def _calculate_memory_mode_stats(self, mode: str, summaries: List[Any]) -> MemoryModeStats:
        """Calculate comprehensive statistics for a memory mode."""
        
        # Collect all individual results
        all_results = []
        for summary in summaries:
            all_results.extend(summary.results)
        
        if not all_results:
            # Return default stats if no results
            return MemoryModeStats(
                memory_mode=mode,
                total_questions=0,
                successful_evaluations=0,
                failed_evaluations=0,
                success_rate=0.0,
                avg_f1_score=0.0, std_f1_score=0.0,
                avg_bleu_1_score=0.0, std_bleu_1_score=0.0,
                avg_llm_judge_score=0.0, std_llm_judge_score=0.0,
                avg_generation_time=0.0, std_generation_time=0.0,
                avg_memory_search_time=0.0, std_memory_search_time=0.0,
                median_f1_score=0.0, median_bleu_1_score=0.0, median_llm_judge_score=0.0,
                conversation_count=0, avg_questions_per_conversation=0.0
            )
        
        # Extract metrics
        f1_scores = [r.f1_score for r in all_results]
        bleu_1_scores = [r.bleu_1_score for r in all_results]
        llm_judge_scores = [r.llm_judge_score for r in all_results]
        generation_times = [r.generation_time for r in all_results]
        memory_search_times = [r.memory_search_time for r in all_results]
        
        # Calculate basic statistics
        total_questions = sum(s.total_questions for s in summaries)
        successful_evaluations = sum(s.successful_evaluations for s in summaries)
        failed_evaluations = sum(s.failed_evaluations for s in summaries)
        success_rate = successful_evaluations / total_questions if total_questions > 0 else 0.0
        
        # Calculate averages and standard deviations
        def safe_mean_std(values):
            if not values:
                return 0.0, 0.0
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            return mean_val, std_val
        
        def safe_median(values):
            return statistics.median(values) if values else 0.0
        
        avg_f1, std_f1 = safe_mean_std(f1_scores)
        avg_bleu, std_bleu = safe_mean_std(bleu_1_scores)
        avg_judge, std_judge = safe_mean_std(llm_judge_scores)
        avg_gen_time, std_gen_time = safe_mean_std(generation_times)
        avg_search_time, std_search_time = safe_mean_std(memory_search_times)
        
        return MemoryModeStats(
            memory_mode=mode,
            total_questions=total_questions,
            successful_evaluations=successful_evaluations,
            failed_evaluations=failed_evaluations,
            success_rate=success_rate,
            avg_f1_score=avg_f1,
            std_f1_score=std_f1,
            avg_bleu_1_score=avg_bleu,
            std_bleu_1_score=std_bleu,
            avg_llm_judge_score=avg_judge,
            std_llm_judge_score=std_judge,
            avg_generation_time=avg_gen_time,
            std_generation_time=std_gen_time,
            avg_memory_search_time=avg_search_time,
            std_memory_search_time=std_search_time,
            median_f1_score=safe_median(f1_scores),
            median_bleu_1_score=safe_median(bleu_1_scores),
            median_llm_judge_score=safe_median(llm_judge_scores),
            conversation_count=len(summaries),
            avg_questions_per_conversation=total_questions / len(summaries) if summaries else 0.0
        )
    
    def _generate_comparisons(self, memory_mode_stats: Dict[str, MemoryModeStats], 
                            results_data: Dict[str, List[Any]]) -> List[ComparisonResult]:
        """Generate pairwise comparisons between memory modes."""
        comparisons = []
        
        modes = list(memory_mode_stats.keys())
        
        # Generate all pairwise comparisons
        for i, baseline_mode in enumerate(modes):
            for comparison_mode in modes[i+1:]:
                comparison = self._compare_memory_modes(
                    baseline_mode, comparison_mode, 
                    memory_mode_stats, results_data
                )
                comparisons.append(comparison)
        
        return comparisons
    
    def _compare_memory_modes(self, baseline_mode: str, comparison_mode: str,
                            memory_mode_stats: Dict[str, MemoryModeStats],
                            results_data: Dict[str, List[Any]]) -> ComparisonResult:
        """Compare two memory modes."""
        
        baseline_stats = memory_mode_stats[baseline_mode]
        comparison_stats = memory_mode_stats[comparison_mode]
        
        def safe_percent_change(new_val, old_val):
            if old_val == 0:
                return 0.0 if new_val == 0 else float('inf')
            return ((new_val - old_val) / old_val) * 100
        
        # Calculate differences
        f1_diff = comparison_stats.avg_f1_score - baseline_stats.avg_f1_score
        f1_percent = safe_percent_change(comparison_stats.avg_f1_score, baseline_stats.avg_f1_score)
        
        bleu_diff = comparison_stats.avg_bleu_1_score - baseline_stats.avg_bleu_1_score
        bleu_percent = safe_percent_change(comparison_stats.avg_bleu_1_score, baseline_stats.avg_bleu_1_score)
        
        judge_diff = comparison_stats.avg_llm_judge_score - baseline_stats.avg_llm_judge_score
        judge_percent = safe_percent_change(comparison_stats.avg_llm_judge_score, baseline_stats.avg_llm_judge_score)
        
        gen_time_diff = comparison_stats.avg_generation_time - baseline_stats.avg_generation_time
        gen_time_percent = safe_percent_change(comparison_stats.avg_generation_time, baseline_stats.avg_generation_time)
        
        search_time_diff = comparison_stats.avg_memory_search_time - baseline_stats.avg_memory_search_time
        search_time_percent = safe_percent_change(comparison_stats.avg_memory_search_time, baseline_stats.avg_memory_search_time)
        
        comparison = ComparisonResult(
            baseline_mode=baseline_mode,
            comparison_mode=comparison_mode,
            f1_score_diff=f1_diff,
            f1_score_percent_change=f1_percent,
            bleu_1_score_diff=bleu_diff,
            bleu_1_percent_change=bleu_percent,
            llm_judge_score_diff=judge_diff,
            llm_judge_percent_change=judge_percent,
            generation_time_diff=gen_time_diff,
            generation_time_percent_change=gen_time_percent,
            memory_search_time_diff=search_time_diff,
            memory_search_time_percent_change=search_time_percent
        )
        
        # Add statistical significance tests if scipy is available
        if SCIPY_AVAILABLE:
            self._add_statistical_tests(comparison, baseline_mode, comparison_mode, results_data)
        
        return comparison
    
    def _add_statistical_tests(self, comparison: ComparisonResult, 
                             baseline_mode: str, comparison_mode: str,
                             results_data: Dict[str, List[Any]]):
        """Add statistical significance tests to comparison."""
        
        # Extract individual results for both modes
        baseline_results = []
        comparison_results = []
        
        for summary in results_data[baseline_mode]:
            baseline_results.extend(summary.results)
        
        for summary in results_data[comparison_mode]:
            comparison_results.extend(summary.results)
        
        if not baseline_results or not comparison_results:
            return
        
        # Extract metric values
        baseline_f1 = [r.f1_score for r in baseline_results]
        comparison_f1 = [r.f1_score for r in comparison_results]
        
        baseline_bleu = [r.bleu_1_score for r in baseline_results]
        comparison_bleu = [r.bleu_1_score for r in comparison_results]
        
        baseline_judge = [r.llm_judge_score for r in baseline_results]
        comparison_judge = [r.llm_judge_score for r in comparison_results]
        
        baseline_gen_time = [r.generation_time for r in baseline_results]
        comparison_gen_time = [r.generation_time for r in comparison_results]
        
        baseline_search_time = [r.memory_search_time for r in baseline_results]
        comparison_search_time = [r.memory_search_time for r in comparison_results]
        
        # Perform t-tests
        try:
            _, comparison.f1_p_value = stats.ttest_ind(comparison_f1, baseline_f1)
            _, comparison.bleu_1_p_value = stats.ttest_ind(comparison_bleu, baseline_bleu)
            _, comparison.llm_judge_p_value = stats.ttest_ind(comparison_judge, baseline_judge)
            _, comparison.generation_time_p_value = stats.ttest_ind(comparison_gen_time, baseline_gen_time)
            _, comparison.memory_search_time_p_value = stats.ttest_ind(comparison_search_time, baseline_search_time)
            
            # Calculate Cohen's d (effect size)
            comparison.f1_effect_size = self._cohens_d(comparison_f1, baseline_f1)
            comparison.bleu_1_effect_size = self._cohens_d(comparison_bleu, baseline_bleu)
            comparison.llm_judge_effect_size = self._cohens_d(comparison_judge, baseline_judge)
            
        except Exception as e:
            self.logger.warning(f"Statistical tests failed: {e}")
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not group1 or not group2:
            return 0.0
        
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        
        if len(group1) == 1 and len(group2) == 1:
            return 0.0
        
        var1 = statistics.variance(group1) if len(group1) > 1 else 0.0
        var2 = statistics.variance(group2) if len(group2) > 1 else 0.0
        
        pooled_std = ((var1 + var2) / 2) ** 0.5
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _generate_dataset_info(self, results_data: Dict[str, List[Any]], 
                             dataset_filename: str = None) -> Dict[str, Any]:
        """Generate dataset information summary."""
        
        total_conversations = 0
        total_questions = 0
        
        for mode, summaries in results_data.items():
            mode_conversations = len(summaries)
            mode_questions = sum(s.total_questions for s in summaries)
            
            total_conversations = max(total_conversations, mode_conversations)
            total_questions = max(total_questions, mode_questions)
        
        return {
            "dataset_filename": dataset_filename or "Unknown",
            "total_conversations": total_conversations,
            "total_questions": total_questions,
            "average_questions_per_conversation": total_questions / total_conversations if total_conversations > 0 else 0.0,
            "memory_modes_evaluated": list(results_data.keys()),
            "scipy_available": SCIPY_AVAILABLE
        }
    
    def _generate_summary(self, memory_mode_stats: Dict[str, MemoryModeStats],
                        comparisons: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate executive summary."""
        
        summary = {
            "memory_modes_count": len(memory_mode_stats),
            "comparisons_count": len(comparisons),
            "best_performing_mode": {},
            "performance_improvements": [],
            "statistical_significance": {}
        }
        
        # Find best performing mode for each metric
        if memory_mode_stats:
            best_f1_mode = max(memory_mode_stats.items(), key=lambda x: x[1].avg_f1_score)
            best_bleu_mode = max(memory_mode_stats.items(), key=lambda x: x[1].avg_bleu_1_score)
            best_judge_mode = max(memory_mode_stats.items(), key=lambda x: x[1].avg_llm_judge_score)
            
            summary["best_performing_mode"] = {
                "f1_score": {"mode": best_f1_mode[0], "score": best_f1_mode[1].avg_f1_score},
                "bleu_1_score": {"mode": best_bleu_mode[0], "score": best_bleu_mode[1].avg_bleu_1_score},
                "llm_judge_score": {"mode": best_judge_mode[0], "score": best_judge_mode[1].avg_llm_judge_score}
            }
        
        # Summarize significant improvements
        for comparison in comparisons:
            if comparison.f1_score_percent_change > 5:  # 5% improvement threshold
                summary["performance_improvements"].append({
                    "comparison": f"{comparison.comparison_mode} vs {comparison.baseline_mode}",
                    "metric": "F1 Score",
                    "improvement": f"+{comparison.f1_score_percent_change:.1f}%",
                    "p_value": comparison.f1_p_value
                })
            
            if comparison.llm_judge_percent_change > 5:
                summary["performance_improvements"].append({
                    "comparison": f"{comparison.comparison_mode} vs {comparison.baseline_mode}",
                    "metric": "LLM Judge Score",
                    "improvement": f"+{comparison.llm_judge_percent_change:.1f}%",
                    "p_value": comparison.llm_judge_p_value
                })
        
        # Statistical significance summary
        if SCIPY_AVAILABLE and comparisons:
            significant_results = []
            for comparison in comparisons:
                if comparison.f1_p_value and comparison.f1_p_value < 0.05:
                    significant_results.append(f"F1: {comparison.comparison_mode} vs {comparison.baseline_mode}")
                if comparison.llm_judge_p_value and comparison.llm_judge_p_value < 0.05:
                    significant_results.append(f"Judge: {comparison.comparison_mode} vs {comparison.baseline_mode}")
            
            summary["statistical_significance"] = {
                "significant_differences": significant_results,
                "alpha_level": 0.05
            }
        
        return summary
    
    def _generate_recommendations(self, memory_mode_stats: Dict[str, MemoryModeStats],
                                comparisons: List[ComparisonResult]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        if not memory_mode_stats:
            recommendations.append("No evaluation results to analyze.")
            return recommendations
        
        # Performance-based recommendations
        if len(memory_mode_stats) > 1:
            best_f1_mode = max(memory_mode_stats.items(), key=lambda x: x[1].avg_f1_score)
            best_judge_mode = max(memory_mode_stats.items(), key=lambda x: x[1].avg_llm_judge_score)
            
            if best_f1_mode[0] == best_judge_mode[0]:
                recommendations.append(f"🏆 **{best_f1_mode[0].upper()} memory consistently outperforms** "
                                     f"across both traditional (F1) and LLM-based evaluation metrics.")
            else:
                recommendations.append(f"📊 **Mixed results**: {best_f1_mode[0].upper()} performs best on F1 score "
                                     f"({best_f1_mode[1].avg_f1_score:.3f}), while {best_judge_mode[0].upper()} "
                                     f"performs best on LLM judge score ({best_judge_mode[1].avg_llm_judge_score:.1f}).")
        
        # Statistical significance recommendations
        if SCIPY_AVAILABLE and comparisons:
            significant_improvements = []
            for comparison in comparisons:
                if comparison.f1_p_value and comparison.f1_p_value < 0.05 and comparison.f1_score_diff > 0:
                    significant_improvements.append(f"{comparison.comparison_mode} over {comparison.baseline_mode}")
            
            if significant_improvements:
                recommendations.append(f"📈 **Statistically significant improvements** found: "
                                     f"{', '.join(significant_improvements)}.")
            else:
                recommendations.append("⚠️  **No statistically significant differences** detected between memory modes "
                                     "(p > 0.05). Consider increasing sample size or investigating other factors.")
        
        # Performance vs efficiency trade-offs
        if len(memory_mode_stats) > 1:
            fastest_mode = min(memory_mode_stats.items(), key=lambda x: x[1].avg_generation_time)
            slowest_mode = max(memory_mode_stats.items(), key=lambda x: x[1].avg_generation_time)
            
            speed_diff = slowest_mode[1].avg_generation_time - fastest_mode[1].avg_generation_time
            if speed_diff > 1.0:  # More than 1 second difference
                recommendations.append(f"⚡ **Performance trade-off**: {fastest_mode[0].upper()} is "
                                     f"{speed_diff:.1f}s faster per question than {slowest_mode[0].upper()}. "
                                     f"Consider speed requirements for production deployment.")
        
        # Data quality recommendations
        total_questions = sum(stats.total_questions for stats in memory_mode_stats.values())
        if total_questions < 50:
            recommendations.append("🔬 **Small sample size**: Consider evaluating on more questions "
                                 f"(current: {total_questions}) for more robust statistical conclusions.")
        
        # Success rate recommendations
        low_success_modes = [mode for mode, stats in memory_mode_stats.items() if stats.success_rate < 0.9]
        if low_success_modes:
            recommendations.append(f"🚨 **Low success rates** detected in {', '.join(low_success_modes)} "
                                 f"modes. Investigate evaluation failures and error handling.")
        
        # Future work recommendations
        recommendations.append("🔮 **Future work**: Consider evaluating on larger datasets, "
                             "different question types, and longer conversation histories for comprehensive analysis.")
        
        return recommendations
    
    def save_report(self, report: EvaluationReport, output_dir: str = "./evaluation/evaluation_output") -> str:
        """
        Save evaluation report to JSON file.
        
        Args:
            report: Evaluation report to save
            output_dir: Output directory
            
        Returns:
            str: Path to saved report file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert dataclasses to dictionaries for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "dataset_info": report.dataset_info,
            "memory_mode_stats": {
                mode: {
                    "memory_mode": stats.memory_mode,
                    "total_questions": stats.total_questions,
                    "successful_evaluations": stats.successful_evaluations,
                    "failed_evaluations": stats.failed_evaluations,
                    "success_rate": stats.success_rate,
                    "avg_f1_score": stats.avg_f1_score,
                    "std_f1_score": stats.std_f1_score,
                    "avg_bleu_1_score": stats.avg_bleu_1_score,
                    "std_bleu_1_score": stats.std_bleu_1_score,
                    "avg_llm_judge_score": stats.avg_llm_judge_score,
                    "std_llm_judge_score": stats.std_llm_judge_score,
                    "avg_generation_time": stats.avg_generation_time,
                    "std_generation_time": stats.std_generation_time,
                    "avg_memory_search_time": stats.avg_memory_search_time,
                    "std_memory_search_time": stats.std_memory_search_time,
                    "median_f1_score": stats.median_f1_score,
                    "median_bleu_1_score": stats.median_bleu_1_score,
                    "median_llm_judge_score": stats.median_llm_judge_score,
                    "conversation_count": stats.conversation_count,
                    "avg_questions_per_conversation": stats.avg_questions_per_conversation
                }
                for mode, stats in report.memory_mode_stats.items()
            },
            "comparisons": [
                {
                    "baseline_mode": comp.baseline_mode,
                    "comparison_mode": comp.comparison_mode,
                    "f1_score_diff": comp.f1_score_diff,
                    "f1_score_percent_change": comp.f1_score_percent_change,
                    "bleu_1_score_diff": comp.bleu_1_score_diff,
                    "bleu_1_percent_change": comp.bleu_1_percent_change,
                    "llm_judge_score_diff": comp.llm_judge_score_diff,
                    "llm_judge_percent_change": comp.llm_judge_percent_change,
                    "generation_time_diff": comp.generation_time_diff,
                    "generation_time_percent_change": comp.generation_time_percent_change,
                    "memory_search_time_diff": comp.memory_search_time_diff,
                    "memory_search_time_percent_change": comp.memory_search_time_percent_change,
                    "f1_p_value": comp.f1_p_value,
                    "bleu_1_p_value": comp.bleu_1_p_value,
                    "llm_judge_p_value": comp.llm_judge_p_value,
                    "generation_time_p_value": comp.generation_time_p_value,
                    "memory_search_time_p_value": comp.memory_search_time_p_value,
                    "f1_effect_size": comp.f1_effect_size,
                    "bleu_1_effect_size": comp.bleu_1_effect_size,
                    "llm_judge_effect_size": comp.llm_judge_effect_size
                }
                for comp in report.comparisons
            ],
            "summary": report.summary,
            "recommendations": report.recommendations
        }
        
        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_analysis_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 Analysis report saved to: {filepath}")
        return filepath
    
    def print_summary_report(self, report: EvaluationReport):
        """Print a formatted summary report to console."""
        
        print("\n" + "=" * 70)
        print("🔬 LOCOMO EVALUATION ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\n📅 Generated: {report.timestamp}")
        print(f"📂 Dataset: {report.dataset_info['dataset_filename']}")
        print(f"💬 Conversations: {report.dataset_info['total_conversations']}")
        print(f"❓ Questions: {report.dataset_info['total_questions']}")
        print(f"📊 Memory Modes: {', '.join(report.dataset_info['memory_modes_evaluated'])}")
        
        print(f"\n📈 PERFORMANCE BY MEMORY MODE")
        print("-" * 50)
        
        for mode, stats in report.memory_mode_stats.items():
            print(f"\n🧠 {mode.upper()} Memory:")
            print(f"  Success Rate: {stats.success_rate:.1%} ({stats.successful_evaluations}/{stats.total_questions})")
            print(f"  F1 Score: {stats.avg_f1_score:.3f} ± {stats.std_f1_score:.3f}")
            print(f"  BLEU-1 Score: {stats.avg_bleu_1_score:.3f} ± {stats.std_bleu_1_score:.3f}")
            print(f"  LLM Judge Score: {stats.avg_llm_judge_score:.1f} ± {stats.std_llm_judge_score:.1f}")
            print(f"  Generation Time: {stats.avg_generation_time:.2f}s ± {stats.std_generation_time:.2f}s")
            print(f"  Memory Search Time: {stats.avg_memory_search_time:.3f}s ± {stats.std_memory_search_time:.3f}s")
        
        if report.comparisons:
            print(f"\n⚖️  PERFORMANCE COMPARISONS")
            print("-" * 50)
            
            for comp in report.comparisons:
                print(f"\n📊 {comp.comparison_mode.upper()} vs {comp.baseline_mode.upper()}:")
                print(f"  F1 Score: {comp.f1_score_diff:+.3f} ({comp.f1_score_percent_change:+.1f}%)"
                      + (f" [p={comp.f1_p_value:.3f}]" if comp.f1_p_value else ""))
                print(f"  BLEU-1 Score: {comp.bleu_1_score_diff:+.3f} ({comp.bleu_1_percent_change:+.1f}%)"
                      + (f" [p={comp.bleu_1_p_value:.3f}]" if comp.bleu_1_p_value else ""))
                print(f"  LLM Judge Score: {comp.llm_judge_score_diff:+.1f} ({comp.llm_judge_percent_change:+.1f}%)"
                      + (f" [p={comp.llm_judge_p_value:.3f}]" if comp.llm_judge_p_value else ""))
                print(f"  Generation Time: {comp.generation_time_diff:+.2f}s ({comp.generation_time_percent_change:+.1f}%)")
        
        if report.summary.get("best_performing_mode"):
            print(f"\n🏆 BEST PERFORMING MODES")
            print("-" * 50)
            best = report.summary["best_performing_mode"]
            print(f"  F1 Score: {best['f1_score']['mode'].upper()} ({best['f1_score']['score']:.3f})")
            print(f"  BLEU-1 Score: {best['bleu_1_score']['mode'].upper()} ({best['bleu_1_score']['score']:.3f})")
            print(f"  LLM Judge Score: {best['llm_judge_score']['mode'].upper()} ({best['llm_judge_score']['score']:.1f})")
        
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 50)
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        if not SCIPY_AVAILABLE:
            print(f"\n⚠️  Note: Install scipy for statistical significance testing")
        
        print("\n" + "=" * 70)
