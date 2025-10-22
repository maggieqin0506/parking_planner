"""
Visualizer with node count display
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config import Config

class Visualizer:
    def plot_environment(self, env, start, goal, path, title="Path Planning"):
        """Plot environment with path"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Obstacles
        for obs in env.obstacles:
            rect = patches.Rectangle(
                (obs['x'] - obs['width']/2, obs['y'] - obs['height']/2),
                obs['width'], obs['height'],
                angle=np.degrees(obs.get('theta', 0)),
                linewidth=2, edgecolor='gray', facecolor='lightgray'
            )
            ax.add_patch(rect)
        
        # Start position
        self._draw_vehicle(ax, start[0], start[1], start[2], 'green', 'Start')
        
        # Goal position
        self._draw_vehicle(ax, goal[0], goal[1], goal[2], 'red', 'Goal')
        
        # Path
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Path')
            
            # Draw vehicle at intervals
            step = max(1, len(path) // 10)
            for i in range(0, len(path), step):
                self._draw_vehicle(ax, path[i][0], path[i][1], path[i][2], 'blue', alpha=0.3)
        
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        return fig, ax
    
    def _draw_vehicle(self, ax, x, y, theta, color, label=None, alpha=1.0):
        """Draw vehicle rectangle"""
        vehicle_length = Config.E_l
        vehicle_width = Config.E_w
        
        # Vehicle corners in local frame
        corners = np.array([
            [-vehicle_length/2, -vehicle_width/2],
            [vehicle_length/2, -vehicle_width/2],
            [vehicle_length/2, vehicle_width/2],
            [-vehicle_length/2, vehicle_width/2]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Transform to world frame
        corners_world = corners @ R.T + np.array([x, y])
        
        # Draw
        rect = patches.Polygon(corners_world, closed=True, 
                              edgecolor=color, facecolor=color, 
                              alpha=alpha, linewidth=2, label=label)
        ax.add_patch(rect)
        
        # Direction indicator
        front_x = x + vehicle_length/2 * np.cos(theta)
        front_y = y + vehicle_length/2 * np.sin(theta)
        ax.plot([x, front_x], [y, front_y], color=color, linewidth=2, alpha=alpha)
    
    def plot_comparison(self, paper_result, neural_result, scenario_name):
        """Plot comparison with node metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['comp_time', 'path_length', 'nodes_generated', 'nodes_expanded']
        titles = ['Computation Time (ms)', 'Path Length (m)', 
                 'Nodes Generated', 'Nodes Expanded']
        colors = ['#3498db', '#e74c3c']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            paper_val = paper_result[metric]
            neural_val = neural_result[metric]
            
            bars = ax.bar(['Paper', 'Neural'], [paper_val, neural_val], color=colors)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}' if idx < 2 else f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Show improvement
            if paper_val > 0:
                improvement = (paper_val - neural_val) / paper_val * 100
                ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%',
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10, fontweight='bold')
        
        fig.suptitle(f'Comparison: {scenario_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_summary_statistics(self, all_results, keys=['paper_method', 'neural']):
        """Plot overall summary with node metrics"""
        successful_results = [r for r in all_results 
                             if all(r[k]['success'] for k in keys)]
        
        if not successful_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            ('comp_time', 'Computation Time (ms)'),
            ('path_length', 'Path Length (m)'),
            ('nodes_generated', 'Nodes Generated'),
            ('nodes_expanded', 'Nodes Expanded')
        ]
        
        for idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            for i, key in enumerate(keys):
                values = [r[key][metric] for r in successful_results]
                scenarios = [r['scenario'].replace(' (Paper Table II)', '')[:20] 
                           for r in successful_results]
                
                x = np.arange(len(scenarios))
                width = 0.35
                offset = (i - 0.5) * width
                
                label = "Paper's Method" if key == 'paper_method' else 'Neural Network'
                color = '#3498db' if key == 'paper_method' else '#e74c3c'
                
                bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}' if idx < 2 else f'{int(height)}',
                           ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Overall Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Summary statistics saved to: results/summary.png")