"""
单独测试垂直泊车场景
"""
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from scenario_planner import ScenarioPlanner
from vehicle import Vehicle, State
from config import Config

def test_perpendicular():
    """测试垂直泊车"""
    
    print("="*70)
    print("测试垂直泊车场景")
    print("="*70)
    
    # 创建环境
    env = Environment()
    start, goal, is_parallel = env.create_perpendicular_parking_scenario(1.4)
    
    print(f"\n配置验证:")
    print(f"  起点: ({start[0]:.2f}, {start[1]:.2f}, {np.degrees(start[2]):.0f}°)")
    print(f"  目标: ({goal[0]:.2f}, {goal[1]:.2f}, {np.degrees(goal[2]):.0f}°)")
    
    # 创建车辆检查碰撞
    vehicle = Vehicle()
    
    # 检查起点
    start_state = State(start[0], start[1], start[2])
    start_collision = vehicle.check_collision(
        start_state.x, start_state.y, start_state.theta,
        env.occupancy_grid
    )
    print(f"\n  起点碰撞: {'❌ 是' if start_collision else '✅ 否'}")
    
    # 检查目标
    goal_state = State(goal[0], goal[1], goal[2])
    goal_collision = vehicle.check_collision(
        goal_state.x, goal_state.y, goal_state.theta,
        env.occupancy_grid
    )
    print(f"  目标碰撞: {'❌ 是' if goal_collision else '✅ 否'}")
    
    if start_collision or goal_collision:
        print(f"\n❌ 场景配置有问题，无法进行规划")
        visualize_failed_scenario(env, start, goal, start_collision, goal_collision)
        return
    
    print(f"\n✅ 起点和目标都无碰撞，开始规划...")
    
    # 尝试规划
    planner = ScenarioPlanner(env)
    result = planner.plan(start, goal, is_parallel, 1.4)
    
    if len(result) == 6:
        path, comp_time, path_length, success, nodes_gen, nodes_exp = result
    else:
        path, comp_time, path_length, success = result
        nodes_gen, nodes_exp = 0, 0
    
    print(f"\n规划结果:")
    print(f"  成功: {'✅ 是' if success else '❌ 否'}")
    print(f"  计算时间: {comp_time:.2f} ms")
    print(f"  路径长度: {path_length:.2f} m")
    print(f"  生成节点: {nodes_gen}")
    print(f"  扩展节点: {nodes_exp}")
    
    # 可视化
    visualize_result(env, start, goal, path, success)
    
    return success

def visualize_result(env, start, goal, path, success):
    """可视化规划结果"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制障碍物
    for obs in env.obstacles:
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D
        
        rect = Rectangle(
            (obs['x'] - obs['width']/2, obs['y'] - obs['height']/2),
            obs['width'], obs['height'],
            linewidth=2, edgecolor='gray', facecolor='lightgray'
        )
        
        if obs.get('theta', 0) != 0:
            t = Affine2D().rotate_around(obs['x'], obs['y'], obs['theta']) + ax.transData
            rect.set_transform(t)
        
        ax.add_patch(rect)
    
    # 绘制车辆
    def draw_vehicle(x, y, theta, color, label, alpha=1.0):
        corners = np.array([
            [-Config.E_l/2, -Config.E_w/2],
            [Config.E_l/2, -Config.E_w/2],
            [Config.E_l/2, Config.E_w/2],
            [-Config.E_l/2, Config.E_w/2],
            [-Config.E_l/2, -Config.E_w/2]
        ])
        
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        corners_world = corners @ R.T + np.array([x, y])
        
        ax.plot(corners_world[:, 0], corners_world[:, 1],
               color=color, linewidth=2, label=label, alpha=alpha)
        ax.plot([x, x + Config.E_l/2 * np.cos(theta)],
               [y, y + Config.E_l/2 * np.sin(theta)],
               color=color, linewidth=2, alpha=alpha)
    
    # 起点和目标
    draw_vehicle(start[0], start[1], start[2], 'green', 'Start', alpha=0.8)
    draw_vehicle(goal[0], goal[1], goal[2], 'blue', 'Goal', alpha=0.8)
    
    # 路径
    if success and path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 
               'r-', linewidth=2, label='Path', alpha=0.6)
        
        # 沿路径绘制车辆
        step = max(1, len(path) // 10)
        for i in range(0, len(path), step):
            draw_vehicle(path[i][0], path[i][1], path[i][2], 
                        'orange', None, alpha=0.3)
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"垂直泊车测试 - {'成功' if success else '失败'}", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/test_perpendicular.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化保存到: results/test_perpendicular.png")
    plt.close()

def visualize_failed_scenario(env, start, goal, start_collision, goal_collision):
    """可视化失败的场景"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制障碍物
    for obs in env.obstacles:
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D
        
        rect = Rectangle(
            (obs['x'] - obs['width']/2, obs['y'] - obs['height']/2),
            obs['width'], obs['height'],
            linewidth=2, edgecolor='gray', facecolor='lightgray'
        )
        
        if obs.get('theta', 0) != 0:
            t = Affine2D().rotate_around(obs['x'], obs['y'], obs['theta']) + ax.transData
            rect.set_transform(t)
        
        ax.add_patch(rect)
    
    # 绘制车辆
    def draw_vehicle(x, y, theta, color, label):
        corners = np.array([
            [-Config.E_l/2, -Config.E_w/2],
            [Config.E_l/2, -Config.E_w/2],
            [Config.E_l/2, Config.E_w/2],
            [-Config.E_l/2, Config.E_w/2],
            [-Config.E_l/2, -Config.E_w/2]
        ])
        
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        corners_world = corners @ R.T + np.array([x, y])
        
        ax.fill(corners_world[:, 0], corners_world[:, 1],
               color=color, alpha=0.5, label=label)
        ax.plot(corners_world[:, 0], corners_world[:, 1],
               color=color, linewidth=2)
    
    # 起点（红色=碰撞，绿色=正常）
    draw_vehicle(start[0], start[1], start[2],
                'red' if start_collision else 'green',
                'Start' + (' (COLLISION!)' if start_collision else ''))
    
    # 目标（红色=碰撞，蓝色=正常）
    draw_vehicle(goal[0], goal[1], goal[2],
                'red' if goal_collision else 'blue',
                'Goal' + (' (COLLISION!)' if goal_collision else ''))
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('垂直泊车场景 - 配置错误', fontsize=14, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('results/perpendicular_failed.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 失败场景可视化保存到: results/perpendicular_failed.png")
    plt.close()

if __name__ == '__main__':
    test_perpendicular()