"""
Main comparison with node count metrics
"""
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from scenario_planner import ScenarioPlanner
from hybrid_astar import HybridAStar
from neural_heuristic import create_neural_heuristic
from visualizer import Visualizer
from config import Config
import os
import json

def run_single_experiment(env, start, goal, is_parallel, scs, neural_model):
    """Run comparison with node count tracking"""
    
    results = {
        'paper_method': {},
        'neural': {}
    }
    
    # === 论文方法 ===
    print("  Running paper's method...")
    scenario_planner = ScenarioPlanner(env)
    result_paper = scenario_planner.plan(start, goal, is_parallel, scs)
    
    if len(result_paper) == 6:
        path_paper, time_paper, length_paper, success_paper, nodes_gen_paper, nodes_exp_paper = result_paper
    else:
        path_paper, time_paper, length_paper, success_paper = result_paper
        nodes_gen_paper, nodes_exp_paper = 0, 0
    
    results['paper_method'] = {
        'path': path_paper,
        'comp_time': time_paper,
        'path_length': length_paper,
        'success': success_paper,
        'nodes_generated': nodes_gen_paper,
        'nodes_expanded': nodes_exp_paper
    }

    print("  Running neural network method...")
    neural_planner = HybridAStar(env, use_neural=True, neural_model=neural_model)
    result_neural = neural_planner.plan(start, goal, is_parallel, scs)
    
    if len(result_neural) == 6:
        path_neural, time_neural, length_neural, success_neural, nodes_gen_neural, nodes_exp_neural = result_neural
    else:
        path_neural, time_neural, length_neural, success_neural = result_neural
        nodes_gen_neural, nodes_exp_neural = 0, 0
    
    results['neural'] = {
        'path': path_neural,
        'comp_time': time_neural,
        'path_length': length_neural,
        'success': success_neural,
        'nodes_generated': nodes_gen_neural,
        'nodes_expanded': nodes_exp_neural
    }
    
    return results

def run_experiments():
    """Run all experiments"""
    print("="*70)
    print("  Comparison: Paper's Method vs Neural Network (with Node Metrics)")
    print("="*70)
    
    # Load neural network
    neural_model = create_neural_heuristic('models/neural_heuristic.pth')
    
    visualizer = Visualizer()
    all_results = []
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/paths', exist_ok=True)
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Parallel Parking (Paper Table II)',
            'type': 'parallel',
            'scs': Config.SCS_para
        },
        {
            'name': 'Perpendicular Parking (Paper Table II)',
            'type': 'perpendicular',
            'scs': Config.SCS_perp
        },
        {
            'name': 'Narrow Parallel (SCS=1.2)',
            'type': 'parallel',
            'scs': 1.2
        },
        {
            'name': 'Narrow Perpendicular (SCS=1.15)',
            'type': 'perpendicular',
            'scs': 1.15
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
        all_results.append(results)
        
        # Print results
        print("\nResults:")
        print(f"  Paper's Method:")
        print(f"    Success: {results['paper_method']['success']}")
        print(f"    Computation Time: {results['paper_method']['comp_time']:.2f} ms")
        print(f"    Path Length: {results['paper_method']['path_length']:.2f} m")
        print(f"    Nodes Generated: {results['paper_method']['nodes_generated']}")
        print(f"    Nodes Expanded: {results['paper_method']['nodes_expanded']}")
        
        print(f"\n  Neural Network:")
        print(f"    Success: {results['neural']['success']}")
        print(f"    Computation Time: {results['neural']['comp_time']:.2f} ms")
        print(f"    Path Length: {results['neural']['path_length']:.2f} m")
        print(f"    Nodes Generated: {results['neural']['nodes_generated']}")
        print(f"    Nodes Expanded: {results['neural']['nodes_expanded']}")
        
        if results['paper_method']['success'] and results['neural']['success']:
            time_imp = ((results['paper_method']['comp_time'] - results['neural']['comp_time']) / 
                       results['paper_method']['comp_time'] * 100)
            length_imp = ((results['paper_method']['path_length'] - results['neural']['path_length']) / 
                         results['paper_method']['path_length'] * 100)
            nodes_imp = ((results['paper_method']['nodes_generated'] - results['neural']['nodes_generated']) / 
                        results['paper_method']['nodes_generated'] * 100)
            
            print(f"\n  Neural Network Improvements:")
            print(f"    Time: {time_imp:+.2f}%")
            print(f"    Path Length: {length_imp:+.2f}%")
            print(f"    Nodes Generated: {nodes_imp:+.2f}%")
        
        # Visualizations
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
    
    # Summary
    visualizer.plot_summary_statistics(all_results, keys=['paper_method', 'neural'])
    
    # Save results
    results_summary = []
    for result in all_results:
        summary = {
            'scenario': result['scenario'],
            'paper_method': {
                'success': result['paper_method']['success'],
                'comp_time': result['paper_method']['comp_time'],
                'path_length': result['paper_method']['path_length'],
                'nodes_generated': result['paper_method']['nodes_generated'],
                'nodes_expanded': result['paper_method']['nodes_expanded']
            },
            'neural': {
                'success': result['neural']['success'],
                'comp_time': result['neural']['comp_time'],
                'path_length': result['neural']['path_length'],
                'nodes_generated': result['neural']['nodes_generated'],
                'nodes_expanded': result['neural']['nodes_expanded']
            }
        }
        results_summary.append(summary)
    
    with open('results/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*70)
    print("Generating Summary")
    print("="*70)
    
    # Final summary with node metrics
    paper_success = sum([r['paper_method']['success'] for r in all_results])
    neural_success = sum([r['neural']['success'] for r in all_results])
    
    print(f"\nSuccess Rate:")
    print(f"  Paper's Method: {paper_success}/{len(all_results)}")
    print(f"  Neural Network: {neural_success}/{len(all_results)}")
    
    successful_both = [r for r in all_results if r['paper_method']['success'] and r['neural']['success']]
    
    if successful_both:
        avg_time_paper = np.mean([r['paper_method']['comp_time'] for r in successful_both])
        avg_time_neural = np.mean([r['neural']['comp_time'] for r in successful_both])
        avg_length_paper = np.mean([r['paper_method']['path_length'] for r in successful_both])
        avg_length_neural = np.mean([r['neural']['path_length'] for r in successful_both])
        avg_nodes_paper = np.mean([r['paper_method']['nodes_generated'] for r in successful_both])
        avg_nodes_neural = np.mean([r['neural']['nodes_generated'] for r in successful_both])
        
        print(f"\nAverage Performance:")
        print(f"  Computation Time:")
        print(f"    Paper: {avg_time_paper:.2f} ms")
        print(f"    Neural: {avg_time_neural:.2f} ms")
        print(f"    Improvement: {(avg_time_paper - avg_time_neural) / avg_time_paper * 100:+.2f}%")
        
        print(f"  Path Length:")
        print(f"    Paper: {avg_length_paper:.2f} m")
        print(f"    Neural: {avg_length_neural:.2f} m")
        print(f"    Improvement: {(avg_length_paper - avg_length_neural) / avg_length_paper * 100:+.2f}%")
        
        print(f"  Nodes Generated:")
        print(f"    Paper: {avg_nodes_paper:.0f}")
        print(f"    Neural: {avg_nodes_neural:.0f}")
        print(f"    Improvement: {(avg_nodes_paper - avg_nodes_neural) / avg_nodes_paper * 100:+.2f}%")

if __name__ == '__main__':
    run_experiments()