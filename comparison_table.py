"""
Generate comparison table with node metrics
"""
import json
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_table():
    """Generate comparison table from results"""
    
    try:
        with open('results/results_summary.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: results_summary.json not found. Run main.py first.")
        return
    
    # Prepare table data
    table_data = []
    headers = ['Scenario', 'Method', 'CT (ms)', 'PL (m)', 'Nodes Gen', 'Nodes Exp', 'Success']
    
    for result in results:
        scenario_name = result['scenario']
        
        # Paper's method
        paper = result['paper_method']
        table_data.append([
            scenario_name,
            "Paper's Method",
            f"{paper['comp_time']:.2f}",
            f"{paper['path_length']:.2f}",
            f"{paper['nodes_generated']}",
            f"{paper['nodes_expanded']}",
            "✓" if paper['success'] else "✗"
        ])
        
        # Neural network
        neural = result['neural']
        table_data.append([
            "",
            "Neural Network",
            f"{neural['comp_time']:.2f}",
            f"{neural['path_length']:.2f}",
            f"{neural['nodes_generated']}",
            f"{neural['nodes_expanded']}",
            "✓" if neural['success'] else "✗"
        ])
        
        # Improvement
        if paper['success'] and neural['success']:
            time_imp = (paper['comp_time'] - neural['comp_time']) / paper['comp_time'] * 100
            length_imp = (paper['path_length'] - neural['path_length']) / paper['path_length'] * 100
            nodes_imp = (paper['nodes_generated'] - neural['nodes_generated']) / paper['nodes_generated'] * 100
            
            table_data.append([
                "",
                "  Improvement",
                f"{time_imp:+.1f}%",
                f"{length_imp:+.1f}%",
                f"{nodes_imp:+.1f}%",
                f"-",
                ""
            ])
        
        table_data.append(["", "", "", "", "", "", ""])  # Separator
    
    # Print table
    print("\n" + "="*100)
    print("COMPARISON TABLE (with Node Generation Metrics)")
    print("="*100)
    print("\nCT = Computation Time (ms)")
    print("PL = Path Length (m)")
    print("Nodes Gen = Nodes Generated")
    print("Nodes Exp = Nodes Expanded")
    print()
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Save to file
    with open('results/comparison_table.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPARISON TABLE (with Node Generation Metrics)\n")
        f.write("="*100 + "\n\n")
        f.write("CT = Computation Time (ms)\n")
        f.write("PL = Path Length (m)\n")
        f.write("Nodes Gen = Nodes Generated\n")
        f.write("Nodes Exp = Nodes Expanded\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print(f"\nTable saved to: results/comparison_table.txt")
    
    # Generate bar chart with node metrics
    generate_enhanced_bar_chart(results)

def generate_enhanced_bar_chart(results):
    """Generate bar chart including node metrics"""
    
    scenarios = []
    paper_times = []
    neural_times = []
    paper_nodes = []
    neural_nodes = []
    
    for result in results:
        if result['paper_method']['success'] and result['neural']['success']:
            scenarios.append(result['scenario'].replace(' (Paper Table II)', '').replace(' (SCS=', '\n(SCS='))
            paper_times.append(result['paper_method']['comp_time'])
            neural_times.append(result['neural']['comp_time'])
            paper_nodes.append(result['paper_method']['nodes_generated'])
            neural_nodes.append(result['neural']['nodes_generated'])
    
    if not scenarios:
        return
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # 1. Computation time
    bars1 = ax1.bar(x - width/2, paper_times, width, label="Paper's Method", color='#3498db')
    bars2 = ax1.bar(x + width/2, neural_times, width, label='Neural Network', color='#e74c3c')
    
    ax1.set_xlabel('Scenario', fontsize=11)
    ax1.set_ylabel('Computation Time (ms)', fontsize=11)
    ax1.set_title('Computation Time Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Path length
    paper_lengths = [result['paper_method']['path_length'] for result in results 
                     if result['paper_method']['success'] and result['neural']['success']]
    neural_lengths = [result['neural']['path_length'] for result in results 
                      if result['paper_method']['success'] and result['neural']['success']]
    
    bars3 = ax2.bar(x - width/2, paper_lengths, width, label="Paper's Method", color='#3498db')
    bars4 = ax2.bar(x + width/2, neural_lengths, width, label='Neural Network', color='#e74c3c')
    
    ax2.set_xlabel('Scenario', fontsize=11)
    ax2.set_ylabel('Path Length (m)', fontsize=11)
    ax2.set_title('Path Length Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars3 + bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. Nodes generated (NEW!)
    bars5 = ax3.bar(x - width/2, paper_nodes, width, label="Paper's Method", color='#3498db')
    bars6 = ax3.bar(x + width/2, neural_nodes, width, label='Neural Network', color='#e74c3c')
    
    ax3.set_xlabel('Scenario', fontsize=11)
    ax3.set_ylabel('Nodes Generated', fontsize=11)
    ax3.set_title('Search Efficiency (Fewer is Better)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars5 + bars6:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/comparison_with_nodes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced bar chart saved to: results/comparison_with_nodes.png")
    
    # Generate improvement summary chart
    generate_improvement_chart(results)

def generate_improvement_chart(results):
    """Generate chart showing percentage improvements"""
    
    time_improvements = []
    length_improvements = []
    node_improvements = []
    scenario_names = []
    
    for result in results:
        if result['paper_method']['success'] and result['neural']['success']:
            paper = result['paper_method']
            neural = result['neural']
            
            time_imp = (paper['comp_time'] - neural['comp_time']) / paper['comp_time'] * 100
            length_imp = (paper['path_length'] - neural['path_length']) / paper['path_length'] * 100
            node_imp = (paper['nodes_generated'] - neural['nodes_generated']) / paper['nodes_generated'] * 100
            
            time_improvements.append(time_imp)
            length_improvements.append(length_imp)
            node_improvements.append(node_imp)
            scenario_names.append(result['scenario'].replace(' (Paper Table II)', '').replace(' (SCS=', '\n(SCS='))
    
    if not scenario_names:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenario_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, time_improvements, width, label='Time Improvement', color='#2ecc71')
    bars2 = ax.bar(x, length_improvements, width, label='Path Length Improvement', color='#f39c12')
    bars3 = ax.bar(x + width, node_improvements, width, label='Nodes Generated Improvement', color='#9b59b6')
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Neural Network Performance Improvements', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/improvements_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Improvement summary saved to: results/improvements_summary.png")

if __name__ == '__main__':
    generate_comparison_table()