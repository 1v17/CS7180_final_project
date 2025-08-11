#!/usr/bin/env python3
"""
Simple Performance Visualization Script

A simplified version focusing on key performance comparisons between
Ebbinghaus and Standard memory modes across different question categories.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Global variables
CSV_FILE_PATH = 'evaluation/evaluation_output/category_performance_stats_20250806_215913.csv'
JSON_FILE_PATH = 'evaluation/evaluation_output/evaluation_analysis_report_20250806_215913.json'
OUTPUT_DIR = 'resources'
COLORS = ['skyblue', 'coral']  # Sky blue for Standard, Coral for Ebbinghaus
CATEGORY_MAPPING = {
    1: "Single Hop",
    2: "Multi-Hop", 
    3: "Open Domain",
    4: "Temporal"
}

def load_json_data():
    """Load JSON data for overall performance comparison."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(script_dir, JSON_FILE_PATH)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_overall_performance_comparison(output_dir):
    """Create comparison plots for F1, BLEU-1, and LLM Judge scores."""
    data = load_json_data()
    
    # Extract overall metrics for both modes from memory_mode_stats
    standard_data = data['memory_mode_stats']['standard']
    ebbinghaus_data = data['memory_mode_stats']['ebbinghaus']
    
    metrics = {
        'F1 Score': {
            'Standard': standard_data['avg_f1_score'],
            'Ebbinghaus': ebbinghaus_data['avg_f1_score']
        },
        'BLEU-1 Score': {
            'Standard': standard_data['avg_bleu_1_score'],
            'Ebbinghaus': ebbinghaus_data['avg_bleu_1_score']
        },
        'LLM Judge Score': {
            'Standard': standard_data['avg_llm_judge_score'],
            'Ebbinghaus': ebbinghaus_data['avg_llm_judge_score']
        }
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        
        modes = list(values.keys())
        scores = list(values.values())
        
        bars = ax.bar(modes, scores, color=COLORS, alpha=0.8, width=0.6)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits for better visualization
        if metric_name == 'LLM Judge Score':
            ax.set_ylim(0, max(scores) * 1.2)
        else:
            ax.set_ylim(0, max(scores) * 1.3)
    
    plt.suptitle('Overall Performance Comparison: Standard vs Ebbinghaus Memory Modes', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'overall_performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Overall performance comparison plot generated")

def create_timing_comparison(output_dir):
    """Create comparison plots for Generation Time and Memory Search Time."""
    data = load_json_data()
    
    # Extract timing metrics for both modes from memory_mode_stats
    standard_data = data['memory_mode_stats']['standard']
    ebbinghaus_data = data['memory_mode_stats']['ebbinghaus']
    
    timing_metrics = {
        'Generation Time (s)': {
            'Standard': standard_data['avg_generation_time'],
            'Ebbinghaus': ebbinghaus_data['avg_generation_time']
        },
        'Memory Search Time (s)': {
            'Standard': standard_data['avg_memory_search_time'],
            'Ebbinghaus': ebbinghaus_data['avg_memory_search_time']
        }
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (metric_name, values) in enumerate(timing_metrics.items()):
        ax = axes[idx]
        
        modes = list(values.keys())
        times = list(values.values())
        
        bars = ax.bar(modes, times, color=COLORS, alpha=0.8, width=0.6)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.3f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits for better visualization
        ax.set_ylim(0, max(times) * 1.3)
    
    plt.suptitle('Timing Performance Comparison: Standard vs Ebbinghaus Memory Modes', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'timing_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Timing comparison plot generated")

def load_and_prepare_data():
    """Load CSV data and add category names."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(script_dir, CSV_FILE_PATH)
    
    df = pd.read_csv(csv_path)
    df['category_name'] = df['category'].map(CATEGORY_MAPPING)
    return df

def setup_output_directory():
    """Create output directory if it doesn't exist."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def create_main_comparison_plot(df, output_dir):
    """Create separate comparison plots for each metric."""
    metrics = [
        ('avg_f1_score', 'F1 Score'),
        ('avg_bleu_1_score', 'BLEU-1 Score'), 
        ('avg_llm_judge_score', 'LLM Judge Score')
    ]

    colors = COLORS  # Sky blue for Standard, Coral for Ebbinghaus

    for metric, title in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create grouped bar chart
        categories = df['category_name'].unique()
        x = np.arange(len(categories))
        width = 0.35
        
        standard_values = []
        ebbinghaus_values = []
        
        for category in categories:
            standard_val = df[(df['category_name'] == category) & 
                             (df['memory_mode'] == 'standard')][metric].iloc[0]
            ebbinghaus_val = df[(df['category_name'] == category) & 
                               (df['memory_mode'] == 'ebbinghaus')][metric].iloc[0]
            standard_values.append(standard_val)
            ebbinghaus_values.append(ebbinghaus_val)
        
        bars1 = ax.bar(x - width/2, standard_values, width, 
                      label='Standard', color=colors[0], alpha=0.8)
        bars2 = ax.bar(x + width/2, ebbinghaus_values, width, 
                      label='Ebbinghaus', color=colors[1], alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Question Category')
        ax.set_ylabel(title)
        ax.set_title(f'Memory System Performance Comparison - {title}')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Create filename based on metric
        filename = f"{metric.replace('avg_', '').replace('_score', '').replace('_', '')}_comparison.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {title} comparison plot generated")

def create_improvement_analysis(df, output_dir):
    """Create improvement analysis showing percentage gains/losses."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance Improvement Analysis\n(Ebbinghaus vs Standard)', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('avg_f1_score', 'F1 Score'),
        ('avg_bleu_1_score', 'BLEU-1 Score'),
        ('avg_llm_judge_score', 'LLM Judge Score')
    ]
    
    categories = df['category_name'].unique()
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        
        improvements = []
        for category in categories:
            ebbinghaus_val = df[(df['category_name'] == category) & 
                               (df['memory_mode'] == 'ebbinghaus')][metric].iloc[0]
            standard_val = df[(df['category_name'] == category) & 
                             (df['memory_mode'] == 'standard')][metric].iloc[0]
            
            improvement = ((ebbinghaus_val - standard_val) / standard_val) * 100
            improvements.append(improvement)
        
        # Color bars based on positive/negative improvement
        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax.bar(categories, improvements, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:+.1f}%', ha='center', 
                   va='bottom' if val > 0 else 'top', fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Question Category')
        ax.set_ylabel('Improvement (%)')
        ax.set_title(f'{title} Improvement')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'simple_improvement_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Improvement analysis generated")

def main():
    """Main function to run the simple visualization."""
    try:
        # Setup output directory
        output_dir = setup_output_directory()
        
        print(f"Generating visualizations...")
        
        # Create new overall performance comparisons
        create_overall_performance_comparison(output_dir)
        create_timing_comparison(output_dir)
        
        # print("Loading performance data...")
        # df = load_and_prepare_data()
        
        # Comment/uncomment the functions below to control which graphs to generate
        # create_main_comparison_plot(df, output_dir)        # Main comparison plot
        # create_improvement_analysis(df, output_dir)        # Improvement analysis
    
    except FileNotFoundError:
        print(f"Warning: CSV file not found. Skipping category-based analysis.")
        
        print(f"✓ All graphs saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
