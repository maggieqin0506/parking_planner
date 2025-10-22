"""
诊断垂直泊车场景为什么失败
"""
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from vehicle import Vehicle, State
from config import Config

def diagnose_perpendicular_scenario():
    """诊断垂直泊车场景"""
    
    print("="*70)
    print("诊断垂直泊车场景")
    print("="*70)
    
    # 创建环境
    env = Environment()
    start, goal, is_parallel = env.create_perpendicular_parking_scenario(1.4)
    
    print(f"\n场景信息:")
    print(f"  起点: ({start[0]:.2f}, {start[1]:.2f}, {np.degrees(start[2]):.1f}°)")
    print(f"  目标: ({goal[0]:.2f}, {goal[1]:.2f}, {np.degrees(goal[2]):.1f}°)")
    print(f"  是否平行泊车: {is_parallel}")
    
    # 创建车辆
    vehicle = Vehicle()
    
    # 检查起点碰撞
    print(f"\n起点碰撞检查:")
    start_state = State(start[0], start[1], start[2])
    start_collision = vehicle.check_collision(
        start_state.x, start_state.y, start_state.theta, 
        env.occupancy_grid
    )
    
    if start_collision:
        print(f"  ❌ 起点碰撞！位置 ({start[0]:.2f}, {start[1]:.2f})")
    else:
        print(f"  ✓ 起点无碰撞")
    
    # 检查目标碰撞
    print(f"\n目标碰撞检查:")
    goal_state = State(goal[0], goal[1], goal[2])
    goal_collision = vehicle.check_collision(
        goal_state.x, goal_state.y, goal_state.theta,
        env.occupancy_grid
    )
    
    if goal_collision:
        print(f"  ❌ 目标碰撞！位置 ({goal[0]:.2f}, {goal[1]:.2f})")
    else:
        print(f"  ✓ 目标无碰撞")
    
    # 检查环境边界
    print(f"\n边界检查:")
    print(f"  环境大小: {env.width} x {env.height}")
    print(f"  起点边界检查:")
    print(f"    X: {start[0]:.2f} (范围: 0.5 - {env.width-0.5:.1f}) - {'✓' if 0.5 <= start[0] <= env.width-0.5 else '❌'}")
    print(f"    Y: {start[1]:.2f} (范围: 0.5 - {env.height-0.5:.1f}) - {'✓' if 0.5 <= start[1] <= env.height-0.5 else '❌'}")
    
    print(f"  目标边界检查:")
    print(f"    X: {goal[0]:.2f} (范围: 0.5 - {env.width-0.5:.1f}) - {'✓' if 0.5 <= goal[0] <= goal.width-0.5 else '❌'}")
    print(f"    Y: {goal[1]:.2f} (范围: 0.5 - {env.height-0.5:.1f}) - {'✓' if 0.5 <= goal[1] <= env.height-0.5 else '❌'}")
    
    # 可视化环境
    print(f"\n生成可视化...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制障碍物
    for obs in env.obstacles:
        rect_x = obs['x'] - obs['width']/2
        rect_y = obs['y'] - obs['height']/2
        
        # 旋转矩形
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D
        
        rect = Rectangle((rect_x, rect_y), obs['width'], obs['height'],
                        linewidth=2, edgecolor='gray', facecolor='lightgray')
        
        if obs.get('theta', 0) != 0:
            t = Affine2D().rotate_around(obs['x'], obs['y'], obs['theta']) + ax.transData
            rect.set_transform(t)
        
        ax.add_patch(rect)
        
        # 标注障碍物
        ax.text(obs['x'], obs['y'], f"Obs\n({obs['x']:.1f},{obs['y']:.1f})",
               ha='center', va='center', fontsize=8, color='red')
    
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
        
        ax.plot(corners_world[:, 0], corners_world[:, 1], 
               color=color, linewidth=2, label=label)
        ax.plot([x, x + Config.E_l/2 * np.cos(theta)],
               [y, y + Config.E_l/2 * np.sin(theta)],
               color=color, linewidth=2)
        ax.plot(x, y, 'o', color=color, markersize=8)
    
    # 绘制起点和目标
    draw_vehicle(start[0], start[1], start[2], 
                'green' if not start_collision else 'red', 
                'Start' + (' (COLLISION!)' if start_collision else ''))
    draw_vehicle(goal[0], goal[1], goal[2], 
                'blue' if not goal_collision else 'red',
                'Goal' + (' (COLLISION!)' if goal_collision else ''))
    
    # 设置坐标轴
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('垂直泊车场景诊断', fontsize=14, fontweight='bold')
    
    # 添加诊断信息
    info_text = f"起点: ({start[0]:.2f}, {start[1]:.2f}, {np.degrees(start[2]):.0f}°)\n"
    info_text += f"目标: ({goal[0]:.2f}, {goal[1]:.2f}, {np.degrees(goal[2]):.0f}°)\n"
    info_text += f"起点碰撞: {'是' if start_collision else '否'}\n"
    info_text += f"目标碰撞: {'是' if goal_collision else '否'}"
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig('results/perpendicular_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"✓ 可视化保存到: results/perpendicular_diagnosis.png")
    plt.close()
    
    # 总结
    print(f"\n{'='*70}")
    print(f"诊断总结:")
    print(f"{'='*70}")
    
    if start_collision or goal_collision:
        print(f"\n❌ 场景配置有问题！")
        if start_collision:
            print(f"  - 起点位置与障碍物碰撞")
            print(f"  - 建议: 调整 create_perpendicular_parking_scenario() 中的 init_x, init_y")
        if goal_collision:
            print(f"  - 目标位置与障碍物碰撞")
            print(f"  - 建议: 检查停车位大小和目标位置计算")
    else:
        print(f"\n✓ 起点和目标都无碰撞")
        print(f"\n可能的其他问题:")
        print(f"  1. 路径规划算法参数需要调整")
        print(f"  2. 运动原语不够（steering angles太少）")
        print(f"  3. 最大迭代次数不够")
        print(f"  4. 目标阈值设置不合理")
    
    # 打印障碍物信息
    print(f"\n障碍物详情:")
    for i, obs in enumerate(env.obstacles):
        print(f"  障碍物 {i+1}: 位置({obs['x']:.2f}, {obs['y']:.2f}), "
              f"大小({obs['width']:.2f} x {obs['height']:.2f}), "
              f"角度{np.degrees(obs.get('theta', 0)):.0f}°")
    
    print(f"\n车辆参数:")
    print(f"  长度: {Config.E_l:.2f}m")
    print(f"  宽度: {Config.E_w:.2f}m")
    print(f"  轴距: {Config.E_wb:.2f}m")

if __name__ == '__main__':
    diagnose_perpendicular_scenario()