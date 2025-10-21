"""
Test and visualize environment setup
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from environment import Environment
from vehicle import Vehicle
from config import Config

def visualize_scenario(scenario_type='perpendicular', scs=1.4):
    """Visualize the parking scenario"""
    env = Environment()
    vehicle = Vehicle()
    
    if scenario_type == 'perpendicular':
        start, goal, is_parallel = env.create_perpendicular_parking_scenario(scs)
        title = f'Perpendicular Parking (SCS={scs})'
    else:
        start, goal, is_parallel = env.create_parallel_parking_scenario(scs)
        title = f'Parallel Parking (SCS={scs})'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot occupancy grid
    ax.imshow(env.occupancy_grid.T, origin='lower', cmap='binary',
             extent=[0, env.width, 0, env.height], alpha=0.5)
    
    # Draw start vehicle
    draw_vehicle(ax, start[0], start[1], start[2], 'green', 'Start')
    
    # Draw goal vehicle
    draw_vehicle(ax, goal[0], goal[1], goal[2], 'red', 'Goal')
    
    # Check collisions
    start_collision = vehicle.check_collision(start[0], start[1], start[2], env.occupancy_grid)
    goal_collision = vehicle.check_collision(goal[0], goal[1], goal[2], env.occupancy_grid)
    
    # Add text info
    info_text = f'Start: ({start[0]:.2f}, {start[1]:.2f}, {np.rad2deg(start[2]):.1f}°)\n'
    info_text += f'Goal: ({goal[0]:.2f}, {goal[1]:.2f}, {np.rad2deg(goal[2]):.1f}°)\n'
    info_text += f'Distance: {np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2):.2f}m\n'
    info_text += f'Start collision: {start_collision}\n'
    info_text += f'Goal collision: {goal_collision}'
    
    ax.text(0.5, env.height - 1, info_text, fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{scenario_type}_test.png', dpi=150)
    print(f'Saved visualization to {scenario_type}_test.png')
    plt.show()

def draw_vehicle(ax, x, y, theta, color='blue', label=None):
    """Draw vehicle rectangle"""
    corners = np.array([
        [-Config.E_w/2, 0],
        [Config.E_w/2, 0],
        [Config.E_w/2, Config.E_l],
        [-Config.E_w/2, Config.E_l]
    ])
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
    
    corners_world = corners @ R.T + np.array([x, y])
    
    vehicle_patch = patches.Polygon(corners_world, closed=True,
                                   edgecolor=color, facecolor=color,
                                   alpha=0.6, linewidth=2, label=label)
    ax.add_patch(vehicle_patch)
    
    # Draw heading arrow
    arrow_length = Config.E_l * 0.4
    arrow_dx = arrow_length * np.cos(theta)
    arrow_dy = arrow_length * np.sin(theta)
    ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.3,
            head_length=0.2, fc=color, ec=color, linewidth=2)

if __name__ == '__main__':
    print("Testing Perpendicular Parking...")
    visualize_scenario('perpendicular', 1.4)
    
    print("\nTesting Parallel Parking...")
    visualize_scenario('parallel', 1.6)