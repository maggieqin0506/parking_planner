"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config import Config
import os

class Visualizer:
    def __init__(self):
        self.fig_size = (12, 10)
        
    def plot_environment(self, env, start, goal, path=None, title="Path Planning Result"):
        """Plot environment with path"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot occupancy grid
        ax.imshow(env.occupancy_grid.T, origin='lower', cmap='binary', 
                 extent=[0, env.width, 0, env.height], alpha=0.5)
        
        # Plot start
        self._draw_vehicle(ax, start[0], start[1], start[2], 'green', 'Start')
        
        # Plot goal
        self._draw_vehicle(ax, goal[0], goal[1], goal[2], 'red', 'Goal')
        
        # Plot path
        if path is not None and len(path) > 0:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Path')
            
            # Plot vehicle at intermediate positions
            step = max(1, len(path) // 10)
            for i in range(0, len(path), step):
                if i > 0 and i < len(path) - 1:
                    self._draw_vehicle(ax, path[i][0], path[i][1], path[i][2], 
                                     'blue', alpha=0.3)
        
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return fig, ax
    
    def _draw_vehicle(self, ax, x, y, theta, color='blue', label=None, alpha=1.0):
        """Draw vehicle rectangle"""
        # Vehicle corners in local frame
        corners = np.array([
            [-Config.E_w/2, 0],
            [Config.E_w/2, 0],
            [Config.E_w/2, Config.E_l],
            [-Config.E_w/2, Config.E_l]
        ])
        
        # Rotation matrix
        R = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        
        # Transform to world frame
        corners_world = corners @ R.T + np.array([x, y])
        
        # Draw rectangle
        vehicle_patch = patches.Polygon(corners_world, closed=True, 
                                       edgecolor=color, facecolor=color, 
                                       alpha=alpha, linewidth=2, label=label)
        ax.add_patch(vehicle_patch)
        
        # Draw heading arrow
        arrow_length = Config.E_l * 0.3
        arrow_dx = arrow_length * np.cos(theta)
        arrow_dy = arrow_length * np.sin(theta)
        ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.2, 
                head_length=0.15, fc=color, ec=color, alpha=alpha)
    
    def plot_comparison(self, results_fixed, results_neural, scenario_name):
        """Plot comparison between fixed and neural heuristics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Computation time comparison
        ax = axes[0, 0]
        methods = ['Paper\'s Method', 'Neural Network']
        times = [results_fixed['comp_time'], results_neural['comp_time']]
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(methods, times, color=colors)
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Computation Time Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f} ms', ha='center', va='bottom')
        
        # Path length comparison
        ax = axes[0, 1]
        lengths = [results_fixed['path_length'], results_neural['path_length']]
        bars = ax.bar(methods, lengths, color=colors)
        ax.set_ylabel('Path Length (m)')
        ax.set_title('Path Length Comparison')
        ax.grid(True, alpha=0.3)
        
        for bar, length in zip(bars, lengths):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{length:.2f} m', ha='center', va='bottom')
        
        # Success rate
        ax = axes[1, 0]
        success = [int(results_fixed['success']), int(results_neural['success'])]
        bars = ax.bar(methods, success, color=colors)
        ax.set_ylabel('Success (1=Success, 0=Failure)')
        ax.set_title('Planning Success')
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        
        # Performance improvement
        ax = axes[1, 1]
        if results_fixed['comp_time'] > 0:
            time_improvement = (results_fixed['comp_time'] - results_neural['comp_time']) / results_fixed['comp_time'] * 100
        else:
            time_improvement = 0
        
        if results_fixed['path_length'] > 0:
            length_improvement = (results_fixed['path_length'] - results_neural['path_length']) / results_fixed['path_length'] * 100
        else:
            length_improvement = 0
        
        metrics = ['Time Reduction', 'Path Improvement']
        improvements = [time_improvement, length_improvement]
        colors_imp = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax.bar(metrics, improvements, color=colors_imp)
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Performance Improvement (Neural vs Paper)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{imp:.1f}%', ha='center', va='bottom' if imp > 0 else 'top')
        
        plt.suptitle(f'Comparison: {scenario_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_summary_statistics(self, all_results, output_path='results/summary.png', 
                               keys=['paper_method', 'neural']):
        """Plot summary statistics across all scenarios"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        key1, key2 = keys
        label1 = "Paper's Method" if key1 == 'paper_method' else key1.replace('_', ' ').title()
        label2 = "Neural Network" if key2 == 'neural' else key2.replace('_', ' ').title()
        
        # Extract data
        times1 = [r[key1]['comp_time'] for r in all_results if r[key1]['success']]
        times2 = [r[key2]['comp_time'] for r in all_results if r[key2]['success']]
        
        lengths1 = [r[key1]['path_length'] for r in all_results if r[key1]['success']]
        lengths2 = [r[key2]['path_length'] for r in all_results if r[key2]['success']]
        
        # Computation time distribution
        ax = axes[0, 0]
        if times1:
            ax.hist(times1, alpha=0.5, label=label1, bins=15, color='#3498db')
        if times2:
            ax.hist(times2, alpha=0.5, label=label2, bins=15, color='#e74c3c')
        ax.set_xlabel('Computation Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Computation Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Path length distribution
        ax = axes[0, 1]
        if lengths1:
            ax.hist(lengths1, alpha=0.5, label=label1, bins=15, color='#3498db')
        if lengths2:
            ax.hist(lengths2, alpha=0.5, label=label2, bins=15, color='#e74c3c')
        ax.set_xlabel('Path Length (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Path Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[1, 0]
        if times1 and times2:
            bp = ax.boxplot([times1, times2], labels=[label1, label2], patch_artist=True)
            bp['boxes'][0].set_facecolor('#3498db')
            bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Computation Time Comparison')
        ax.grid(True, alpha=0.3)
        
        # Success rate
        ax = axes[1, 1]
        success1 = sum([r[key1]['success'] for r in all_results])
        success2 = sum([r[key2]['success'] for r in all_results])
        total = len(all_results)
        
        success_rates = [success1/total * 100, success2/total * 100]
        bars = ax.bar([label1, label2], success_rates, color=['#3498db', '#e74c3c'])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Overall Success Rate')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.suptitle(f'Summary: {label1} vs {label2}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs('results', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Summary statistics saved to {output_path}")