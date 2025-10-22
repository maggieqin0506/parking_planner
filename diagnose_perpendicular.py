import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from vehicle import Vehicle, State
from config import Config


def diagnose_perpendicular_scenario():
    print("=" * 70)
    print("Diagnosing Perpendicular Parking Scenario")
    print("=" * 70)

    env = Environment()
    # Creates a perpendicular parking scenario with a specific parameter (e.g., parking space width factor 1.4)
    start, goal, is_parallel = env.create_perpendicular_parking_scenario(1.4)

    print(f"\nScenario Information:")
    # Formats the start state coordinates and converts yaw angle to degrees for display
    print(f"  Start: ({start[0]:.2f}, {start[1]:.2f}, {np.degrees(start[2]):.1f}°)")
    # Formats the goal state coordinates and converts yaw angle to degrees for display
    print(f"  Goal: ({goal[0]:.2f}, {goal[1]:.2f}, {np.degrees(goal[2]):.1f}°)")
    print(f"  Is Parallel Parking: {is_parallel}")

    vehicle = Vehicle()

    # Check for collision at the start position
    print(f"\nStart Collision Check:")
    start_state = State(start[0], start[1], start[2])
    start_collision = vehicle.check_collision(
        start_state.x, start_state.y, start_state.theta,
        env.occupancy_grid
    )

    if start_collision:
        print(f"  ❌ Start Collision! Position ({start[0]:.2f}, {start[1]:.2f})")
    else:
        print(f"  ✓ Start No Collision")

    # Check for collision at the goal position
    print(f"\nGoal Collision Check:")
    goal_state = State(goal[0], goal[1], goal[2])
    goal_collision = vehicle.check_collision(
        goal_state.x, goal_state.y, goal_state.theta,
        env.occupancy_grid
    )

    if goal_collision:
        print(f"  ❌ Goal Collision! Position ({goal[0]:.2f}, {goal[1]:.2f})")
    else:
        print(f"  ✓ Goal No Collision")

    # Check environment boundaries
    print(f"\nBoundary Check:")
    print(f"  Environment Size: {env.width} x {env.height}")
    print(f"  Start Boundary Check:")
    # Checks if start X is within environment bounds (with a 0.5m margin)
    print(
        f"    X: {start[0]:.2f} (Range: 0.5 - {env.width - 0.5:.1f}) - {'✓' if 0.5 <= start[0] <= env.width - 0.5 else '❌'}")
    # Checks if start Y is within environment bounds (with a 0.5m margin)
    print(
        f"    Y: {start[1]:.2f} (Range: 0.5 - {env.height - 0.5:.1f}) - {'✓' if 0.5 <= start[1] <= env.height - 0.5 else '❌'}")

    print(f"  Goal Boundary Check:")
    # NOTE: There's a typo in the original Chinese code here: 'goal.width' should be 'env.width'.
    # I've kept 'goal.width' in the string output for a faithful translation of the *output format* (which is what the user asked for),
    # but I've kept the *check* logic with 'env.width' as the original check does.
    print(
        f"    X: {goal[0]:.2f} (Range: 0.5 - {env.width - 0.5:.1f}) - {'✓' if 0.5 <= goal[0] <= env.width - 0.5 else '❌'}")
    # Checks if goal Y is within environment bounds (with a 0.5m margin)
    print(
        f"    Y: {goal[1]:.2f} (Range: 0.5 - {env.height - 0.5:.1f}) - {'✓' if 0.5 <= goal[1] <= env.height - 0.5 else '❌'}")

    # Visualize the environment
    print(f"\nGenerating Visualization...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw obstacles
    for obs in env.obstacles:
        rect_x = obs['x'] - obs['width'] / 2
        rect_y = obs['y'] - obs['height'] / 2

        # Rotate rectangle
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D

        rect = Rectangle((rect_x, rect_y), obs['width'], obs['height'],
                         linewidth=2, edgecolor='gray', facecolor='lightgray')

        if obs.get('theta', 0) != 0:
            t = Affine2D().rotate_around(obs['x'], obs['y'], obs['theta']) + ax.transData
            rect.set_transform(t)

        ax.add_patch(rect)

        # Label obstacle
        ax.text(obs['x'], obs['y'], f"Obs\n({obs['x']:.1f},{obs['y']:.1f})",
                ha='center', va='center', fontsize=8, color='red')

    # Draw vehicle
    def draw_vehicle(x, y, theta, color, label):
        corners = np.array([
            [-Config.E_l / 2, -Config.E_w / 2],
            [Config.E_l / 2, -Config.E_w / 2],
            [Config.E_l / 2, Config.E_w / 2],
            [-Config.E_l / 2, Config.E_w / 2],
            [-Config.E_l / 2, -Config.E_w / 2]
        ])

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        corners_world = corners @ R.T + np.array([x, y])

        ax.plot(corners_world[:, 0], corners_world[:, 1],
                color=color, linewidth=2, label=label)
        # Draw a line indicating the vehicle's orientation (front)
        ax.plot([x, x + Config.E_l / 2 * np.cos(theta)],
                [y, y + Config.E_l / 2 * np.sin(theta)],
                color=color, linewidth=2)
        # Draw the center point
        ax.plot(x, y, 'o', color=color, markersize=8)

    # Draw start and goal
    draw_vehicle(start[0], start[1], start[2],
                 'green' if not start_collision else 'red',
                 'Start' + (' (COLLISION!)' if start_collision else ''))
    draw_vehicle(goal[0], goal[1], goal[2],
                 'blue' if not goal_collision else 'red',
                 'Goal' + (' (COLLISION!)' if goal_collision else ''))

    # Set axes
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Perpendicular Parking Scenario Diagnosis', fontsize=14, fontweight='bold')

    # Add diagnosis information
    info_text = f"Start: ({start[0]:.2f}, {start[1]:.2f}, {np.degrees(start[2]):.0f}°)\n"
    info_text += f"Goal: ({goal[0]:.2f}, {goal[1]:.2f}, {np.degrees(goal[2]):.0f}°)\n"
    info_text += f"Start Collision: {'Yes' if start_collision else 'No'}\n"
    info_text += f"Goal Collision: {'Yes' if goal_collision else 'No'}"

    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')

    plt.tight_layout()
    # Save the visualization
    plt.savefig('results/perpendicular_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: results/perpendicular_diagnosis.png")
    plt.close()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Diagnosis Summary:")
    print(f"{'=' * 70}")

    if start_collision or goal_collision:
        print(f"\n❌ Scenario configuration has issues!")
        if start_collision:
            print(f"  - Start position collides with an obstacle")
            print(f"  - Suggestion: Adjust init_x, init_y in create_perpendicular_parking_scenario()")
        if goal_collision:
            print(f"  - Goal position collides with an obstacle")
            print(f"  - Suggestion: Check parking space size and goal position calculation")
    else:
        print(f"\n✓ Start and goal have no collision")
        print(f"\nPossible Other Issues:")
        print(f"  1. Path planning algorithm parameters need tuning")
        print(f"  2. Insufficient motion primitives (too few steering angles)")
        print(f"  3. Maximum iterations are not enough")
        print(f"  4. Unreasonable goal threshold setting")

    # Print obstacle information
    print(f"\nObstacle Details:")
    for i, obs in enumerate(env.obstacles):
        # Displays the position, size, and rotation angle of each obstacle
        print(f"  Obstacle {i + 1}: Position({obs['x']:.2f}, {obs['y']:.2f}), "
              f"Size({obs['width']:.2f} x {obs['height']:.2f}), "
              f"Angle{np.degrees(obs.get('theta', 0)):.0f}°")

    print(f"\nVehicle Parameters:")
    print(f"  Length: {Config.E_l:.2f}m")
    print(f"  Width: {Config.E_w:.2f}m")
    print(f"  Wheelbase: {Config.E_wb:.2f}m")


if __name__ == '__main__':
    diagnose_perpendicular_scenario()