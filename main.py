"""
Main script to run experiments comparing:
1. Paper's original method (fixed heuristic with scenario-based planning)
2. Neural network-enhanced heuristic
"""
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from scenario_planner import ScenarioPlanner  # Paper's exact algorithm
from hybrid_astar import HybridAStar
from neural_heuristic import load_trained_model
from visualizer import Visualizer
from config import Config
import os
import json

def run_single_experiment(env, start, goal, is_parallel, scs, neural_model):
    """Run both paper's method and neural heuristic"""
    
    results = {
        'paper_method': {},
        'neural': {}
    }
    
    # Paper's Method (Scenario-based with fixed heuristic)
    print("  Running paper's scenario-based method (fixed heuristic)...")
    scenario_planner = ScenarioPlanner(env)
    path_paper, time_paper, length_paper, success_paper = scenario_planner.plan(
        start, goal, is_parallel, scs
    )
    
    results['paper_method'] = {
        'path': path_paper,
        'comp_time': time_paper,
        'path_length': length_paper,
        'success': success_paper
    }
    
    # Neural Network Heuristic (using standard Hybrid A*)
    print("  Running neural network heuristic...")
    planner_neural = HybridAStar(env, use_neural=True, neural_model=neural_model)
    path_neural, time_neural, length_neural, success_neural = planner_neural.plan(
        start, goal, is_parallel, scs
    )
    
    results['neural'] = {
        'path': path_neural,
        'comp_time': time_neural,
        'path_length': length_neural,
        'success': success_neural
    }
    
    return results

def run_experiments():
    """Run all experiments comparing paper's method vs neural network"""
    print("="*70)
    print("  Comparison: Paper's Method vs Neural Network-Enhanced Heuristic")
    print("="*70)
    
    # Load trained neural network
    neural_model = load_trained_model('models/neural_heuristic.pth')
    
    visualizer = Visualizer()
    all_results = []
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/paths', exist_ok=True)
    
    # Test scenarios matching Table II from paper
    test_scenarios = [
        {
            'name': 'Parallel Parking (Paper Table II)',
            'type': 'parallel',
            'scs': Config.SCS_para,  # 1.6
            'expected_time_paper': 196.23,  # ms from Table II
            'expected_length_paper': 16.33   # m from Table II
        },
        {
            'name': 'Perpendicular Parking (Paper Table II)',
            'type': 'perpendicular',
            'scs': Config.SCS_perp,  # 1.4
            'expected_time_paper': 217.83,
            'expected_length_paper': 15.03
        },
        {
            'name': 'Narrow Parallel (SCS=1.2)',
            'type': 'parallel',
            'scs': 1.2,
            'expected_time_paper': None,
            'expected_length_paper': None
        },
        {
            'name': 'Narrow Perpendicular (SCS=1.15)',
            'type': 'perpendicular',
            'scs': 1.15,
            'expected_time_paper': None,
            'expected_length_paper': None
        }
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*70}")
        print(f"Scenario {i+1}: {scenario['name']}")
        print(f"{'='*70}")
        
        # Create environment
        env = Environment()
        if scenario['type'] == 'parallel':
            start, goal, is_parallel = env.create_parallel_parking_scenario(scenario['scs'])
        else:
            start, goal, is_parallel = env.create_perpendicular_parking_scenario(scenario['scs'])
        
        # Run experiment
        results = run_single_experiment(env, start, goal, is_parallel, scenario['scs'], neural_model)
        results['scenario'] = scenario['name']
        results['expected_paper'] = {
            'time': scenario['expected_time_paper'],
            'length': scenario['expected_length_paper']
        }
        all_results.append(results)
        
        # Print results
        print("\nResults:")
        print(f"  Paper's Method (Scenario-based + Fixed Heuristic):")
        print(f"    Success: {results['paper_method']['success']}")
        print(f"    Computation Time: {results['paper_method']['comp_time']:.2f} ms", end='')
        if scenario['expected_time_paper']:
            print(f" (Paper reports: {scenario['expected_time_paper']:.2f} ms)")
        else:
            print()
        print(f"    Path Length: {results['paper_method']['path_length']:.2f} m", end='')
        if scenario['expected_length_paper']:
            print(f" (Paper reports: {scenario['expected_length_paper']:.2f} m)")
        else:
            print()
        
        print(f"\n  Neural Network-Enhanced Heuristic:")
        print(f"    Success: {results['neural']['success']}")
        print(f"    Computation Time: {results['neural']['comp_time']:.2f} ms")
        print(f"    Path Length: {results['neural']['path_length']:.2f} m")
        
        if results['paper_method']['success'] and results['neural']['success']:
            time_improvement = ((results['paper_method']['comp_time'] - results['neural']['comp_time']) / 
                              results['paper_method']['comp_time'] * 100)
            length_improvement = ((results['paper_method']['path_length'] - results['neural']['path_length']) / 
                                results['paper_method']['path_length'] * 100)
            print(f"\n  Neural Network Improvements over Paper's Method:")
            print(f"    Time: {time_improvement:+.2f}%")
            print(f"    Path Length: {length_improvement:+.2f}%")
        
        # Visualize paths
        if results['paper_method']['success']:
            fig, ax = visualizer.plot_environment(
                env, start, goal, results['paper_method']['path'],
                title=f"{scenario['name']} - Paper's Method"
            )
            plt.savefig(f"results/paths/scenario_{i+1}_paper.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        if results['neural']['success']:
            fig, ax = visualizer.plot_environment(
                env, start, goal, results['neural']['path'],
                title=f"{scenario['name']} - Neural Network"
            )
            plt.savefig(f"results/paths/scenario_{i+1}_neural.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Comparison plot
        if results['paper_method']['success'] and results['neural']['success']:
            fig = visualizer.plot_comparison(
                results['paper_method'], results['neural'], scenario['name']
            )
            plt.savefig(f"results/comparison_scenario_{i+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    # Generate summary
    print("\n" + "="*70)
    print("Generating Summary Statistics")
    print("="*70)
    
    visualizer.plot_summary_statistics(all_results, 
                                      keys=['paper_method', 'neural'])
    
    # Save results
    results_summary = []
    for result in all_results:
        summary = {
            'scenario': result['scenario'],
            'paper_method': {
                'success': result['paper_method']['success'],
                'comp_time': result['paper_method']['comp_time'],
                'path_length': result['paper_method']['path_length']
            },
            'expected_paper': result['expected_paper'],
            'neural': {
                'success': result['neural']['success'],
                'comp_time': result['neural']['comp_time'],
                'path_length': result['neural']['path_length']
            }
        }
        results_summary.append(summary)
    
    with open('results/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\nAll results saved to 'results/' directory")
    print("="*70)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: Paper's Method vs Neural Network")
    print("="*70)
    
    paper_success = sum([r['paper_method']['success'] for r in all_results])
    neural_success = sum([r['neural']['success'] for r in all_results])
    
    print(f"\nSuccess Rate:")
    print(f"  Paper's Method: {paper_success}/{len(all_results)} ({paper_success/len(all_results)*100:.1f}%)")
    print(f"  Neural Network: {neural_success}/{len(all_results)} ({neural_success/len(all_results)*100:.1f}%)")
    
    successful_both = [r for r in all_results if r['paper_method']['success'] and r['neural']['success']]
    
    if successful_both:
        avg_time_paper = np.mean([r['paper_method']['comp_time'] for r in successful_both])
        avg_time_neural = np.mean([r['neural']['comp_time'] for r in successful_both])
        avg_length_paper = np.mean([r['paper_method']['path_length'] for r in successful_both])
        avg_length_neural = np.mean([r['neural']['path_length'] for r in successful_both])
        
        print(f"\nAverage Performance (successful cases):")
        print(f"  Computation Time:")
        print(f"    Paper's Method: {avg_time_paper:.2f} ms")
        print(f"    Neural Network: {avg_time_neural:.2f} ms")
        print(f"    Improvement: {(avg_time_paper - avg_time_neural) / avg_time_paper * 100:+.2f}%")
        
        print(f"  Path Length:")
        print(f"    Paper's Method: {avg_length_paper:.2f} m")
        print(f"    Neural Network: {avg_length_neural:.2f} m")
        print(f"    Improvement: {(avg_length_paper - avg_length_neural) / avg_length_paper * 100:+.2f}%")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    if not os.path.exists('models/neural_heuristic.pth'):
        print("WARNING: Trained model not found!")
        print("Please run: python data_generator.py && python train_network.py")
        print("\nContinuing with untrained model...")
    
    run_experiments()