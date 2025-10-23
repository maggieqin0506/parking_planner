import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from scenario_planner import ScenarioPlanner  # Assuming this class exists
from vehicle import Vehicle, State
from config import Config


def test_perpendicular():
    print("=" * 70)
    print("Testing Perpendicular Parking Scenario")
    print("=" * 70)

    env = Environment()
    # Create the scenario configuration (start, goal, is_parallel)
    start, goal, is_parallel = env.create_perpendicular_parking_scenario(1.4)

    print(f"\nConfiguration Verification:")
    print(f"  Start: ({start[0]:.2f}, {start[1]:.2f}, {np.degrees(start[2]):.0f}°)")
    print(f"  Goal: ({goal[0]:.2f}, {goal[1]:.2f}, {np.degrees(goal[2]):.0f}°)")

    vehicle = Vehicle()

    # Check start position collision
    start_state = State(start[0], start[1], start[2])
    start_collision = vehicle.check_collision(
        start_state.x, start_state.y, start_state.theta,
        env.occupancy_grid
    )
    print(f"\n  Start Collision: {'❌ Yes' if start_collision else '✅ No'}")

    # Check goal position collision
    goal_state = State(goal[0], goal[1], goal[2])
    goal_collision = vehicle.check_collision(
        goal_state.x, goal_state.y, goal_state.theta,
        env.occupancy_grid
    )
    print(f"  Goal Collision: {'❌ Yes' if goal_collision else '✅ No'}")

    if start_collision or goal_collision:
        print(f"\n❌ Scenario configuration is flawed, planning cannot proceed")
        visualize_failed_scenario(env, start, goal, start_collision, goal_collision)
        return False

    print(f"\n✅ Start and goal are collision-free, starting planning...")

    # Attempt planning
    # Assuming ScenarioPlanner initializes and uses Hybrid A* internally
    planner = ScenarioPlanner(env)
    # The 'scs' parameter 1.4 is passed to the planner, likely for heuristic configuration
    result = planner.plan(start, goal, is_parallel, 1.4)

    # Unpack the result (handling different possible return formats)
    if len(result) == 6:
        path, comp_time, path_length, success, nodes_gen, nodes_exp = result
    else:
        path, comp_time, path_length, success = result
        nodes_gen, nodes_exp = 0, 0

    print(f"\nPlanning Results:")
    print(f"  Success: {'✅ Yes' if success else '❌ No'}")
    print(f"  Computation Time: {comp_time:.2f} ms")
    print(f"  Path Length: {path_length:.2f} m")
    print(f"  Nodes Generated: {nodes_gen}")
    print(f"  Nodes Expanded: {nodes_exp}")

    # Visualization
    visualize_result(env, start, goal, path, success)

    return success


def visualize_result(env, start, goal, path, success):
    """Visualize the planning result"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw obstacles
    for obs in env.obstacles:
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D

        # Calculate lower left corner for the rectangle
        rect = Rectangle(
            (obs['x'] - obs['width'] / 2, obs['y'] - obs['height'] / 2),
            obs['width'], obs['height'],
            linewidth=2, edgecolor='gray', facecolor='lightgray'
        )

        # Apply rotation if specified
        if obs.get('theta', 0) != 0:
            t = Affine2D().rotate_around(obs['x'], obs['y'], obs['theta']) + ax.transData
            rect.set_transform(t)

        ax.add_patch(rect)

    # Draw vehicle (outline and front indicator)
    def draw_vehicle(x, y, theta, color, label, alpha=1.0):
        # Define vehicle corners relative to its center (Config.E_l/Config.E_w are length/width)
        corners = np.array([
            [-Config.E_l / 2, -Config.E_w / 2],
            [Config.E_l / 2, -Config.E_w / 2],
            [Config.E_l / 2, Config.E_w / 2],
            [-Config.E_l / 2, Config.E_w / 2],
            [-Config.E_l / 2, -Config.E_w / 2]
        ])

        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Transform to world coordinates
        corners_world = corners @ R.T + np.array([x, y])

        # Plot outline
        ax.plot(corners_world[:, 0], corners_world[:, 1],
                color=color, linewidth=2, label=label, alpha=alpha)
        # Plot front indicator
        ax.plot([x, x + Config.E_l / 2 * np.cos(theta)],
                [y, y + Config.E_l / 2 * np.sin(theta)],
                color=color, linewidth=2, alpha=alpha)

    # Start and Goal vehicles
    draw_vehicle(start[0], start[1], start[2], 'green', 'Start', alpha=0.8)
    draw_vehicle(goal[0], goal[1], goal[2], 'blue', 'Goal', alpha=0.8)

    # Path
    if success and path:
        path_array = np.array(path)
        # Plot the path line
        ax.plot(path_array[:, 0], path_array[:, 1],
                'r-', linewidth=2, label='Path', alpha=0.6)

        # Draw vehicle snapshots along the path
        step = max(1, len(path) // 10)
        for i in range(0, len(path), step):
            # Pass None for label to avoid multiple legend entries
            draw_vehicle(path[i][0], path[i][1], path[i][2],
                         'orange', None, alpha=0.3)

    # Set plot properties
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"Perpendicular Parking Test - {'Success' if success else 'Failed'}",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/test_perpendicular.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: results/test_perpendicular.png")
    plt.close()


def visualize_failed_scenario(env, start, goal, start_collision, goal_collision):
    """Visualize a failed scenario configuration (collision at start/goal)"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw obstacles
    for obs in env.obstacles:
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D

        rect = Rectangle(
            (obs['x'] - obs['width'] / 2, obs['y'] - obs['height'] / 2),
            obs['width'], obs['height'],
            linewidth=2, edgecolor='gray', facecolor='lightgray'
        )

        if obs.get('theta', 0) != 0:
            t = Affine2D().rotate_around(obs['x'], obs['y'], obs['theta']) + ax.transData
            rect.set_transform(t)

        ax.add_patch(rect)

    # Draw vehicle (filled for clarity of collision)
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

        # Fill the vehicle shape
        ax.fill(corners_world[:, 0], corners_world[:, 1],
                color=color, alpha=0.5, label=label)
        # Draw the outline
        ax.plot(corners_world[:, 0], corners_world[:, 1],
                color=color, linewidth=2)

    # Start (Red = Collision, Green = OK)
    draw_vehicle(start[0], start[1], start[2],
                 'red' if start_collision else 'green',
                 'Start' + (' (COLLISION!)' if start_collision else ''))

    # Goal (Red = Collision, Blue = OK)
    draw_vehicle(goal[0], goal[1], goal[2],
                 'red' if goal_collision else 'blue',
                 'Goal' + (' (COLLISION!)' if goal_collision else ''))

    # Set plot properties
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Perpendicular Parking Scenario - Configuration Error', fontsize=14, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig('results/perpendicular_failed.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Failed scenario visualization saved to: results/perpendicular_failed.png")
    plt.close()


if __name__ == '__main__':
    test_perpendicular()