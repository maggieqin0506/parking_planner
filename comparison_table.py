"""
Generate a comparison table similar to Table II in the paper
"""
import numpy as np
import json
from tabulate import tabulate
import matplotlib.pyplot as plt

def generate_comparison_table():
    """Generate comparison table from results"""
    
    # Load results
    try:
        with open('results/results_summary.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: results_summary.json not found. Please run main.py first.")
        return
    
    # Prepare table data
    table_data = []
    headers = ['Scenario', 'Method', 'CT (ms)', 'PL (m)', 'Success']
    
    for result in results:
        scenario_name = result['scenario']
        
        # Paper's method
        paper = result['paper_method']
        table_data.append([
            scenario_name,
            "Paper's Method",
            f"{paper['comp_time']:.2f}",
            f"{paper['path_length']:.2f}",
            "✓" if paper['success'] else "✗"
        ])
        
        # Expected from paper (if available)
        if result['expected_paper']['time'] is not None:
            table_data.append([
                "",
                "  (Paper reports)",
                f"{result['expected_paper']['time']:.2f}",
                f"{result['expected_paper']['length']:.2f}",
                "✓"
            ])
        
        # Neural network
        neural = result['neural']
        table_data.append([
            "",
            "Neural Network",
            f"{neural['comp_time']:.2f}",
            f"{neural['path_length']:.2f}",
            "✓" if neural['success'] else "✗"
        ])
        
        # Improvement
        if paper['success'] and neural['success']:
            time_imp = (paper['comp_time'] - neural['comp_time']) / paper['comp_time'] * 100
            length_imp = (paper['path_length'] - neural['path_length']) / paper['path_length'] * 100
            table_data.append([
                "",
                "  Improvement",
                f"{time_imp:+.1f}%",
                f"{length_imp:+.1f}%",
                ""
            ])
        
        table_data.append(["", "", "", "", ""])  # Separator
    
    # Print table
    print("\n" + "="*80)
    print("COMPARISON TABLE (Similar to Paper's Table II)")
    print("="*80)
    print("\nCT = Computation Time (ms)")
    print("PL = Path Length (m)")
    print()
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Save to file
    with open('results/comparison_table.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPARISON TABLE (Similar to Paper's Table II)\n")
        f.write("="*80 + "\n\n")
        f.write("CT = Computation Time (ms)\n")
        f.write("PL = Path Length (m)\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print(f"\nTable saved to: results/comparison_table.txt")
    
    # Generate bar chart comparison
    generate_bar_chart(results)

def generate_bar_chart(results):
    """Generate bar chart comparing methods"""
    
    scenarios = []
    paper_times = []
    neural_times = []
    paper_lengths = []
    neural_lengths = []
    
    for result in results:
        if result['paper_method']['success'] and result['neural']['success']:
            scenarios.append(result['scenario'].replace(' (Paper Table II)', '').replace(' (SCS=', '\n(SCS='))
            paper_times.append(result['paper_method']['comp_time'])
            neural_times.append(result['neural']['comp_time'])
            paper_lengths.append(result['paper_method']['path_length'])
            neural_lengths.append(result['neural']['path_length'])
    
    if not scenarios:
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Computation time
    bars1 = ax1.bar(x - width/2, paper_times, width, label="Paper's Method", color='#3498db')
    bars2 = ax1.bar(x + width/2, neural_times, width, label='Neural Network', color='#e74c3c')
    
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Computation Time (ms)')
    ax1.set_title('Computation Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Path length
    bars3 = ax2.bar(x - width/2, paper_lengths, width, label="Paper's Method", color='#3498db')
    bars4 = ax2.bar(x + width/2, neural_lengths, width, label='Neural Network', color='#e74c3c')
    
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Path Length (m)')
    ax2.set_title('Path Length Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/comparison_bar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Bar chart saved to: results/comparison_bar_chart.png")

if __name__ == '__main__':
    generate_comparison_table()