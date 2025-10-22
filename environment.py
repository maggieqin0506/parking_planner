"""
Environment - FINAL WORKING VERSION
修复: 目标位置需要向后移动，避免与侧边车辆碰撞
"""
import numpy as np
from config import Config

class Environment:
    def __init__(self, width=15.0, height=15.0, resolution=0.2):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))
        
        self.obstacles = []
        
    def add_obstacle(self, x, y, width, height, theta=0):
        """Add rectangular obstacle"""
        self.obstacles.append({
            'x': x, 'y': y, 'width': width, 'height': height, 'theta': theta
        })
        
        # Update occupancy grid
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                gx = i * self.resolution
                gy = j * self.resolution
                
                dx = gx - x
                dy = gy - y
                rx = dx * np.cos(-theta) - dy * np.sin(-theta)
                ry = dx * np.sin(-theta) + dy * np.cos(-theta)
                
                if abs(rx) <= width/2 and abs(ry) <= height/2:
                    self.occupancy_grid[i, j] = 1
    
    def create_parallel_parking_scenario(self, scs=1.6):
        """平行泊车 - 已验证工作"""
        self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))
        self.obstacles = []
        
        vehicle_y = 5.0
        parking_length = scs * Config.E_l
        
        # 前车
        front_x = 6.0
        self.add_obstacle(front_x, vehicle_y, Config.E_l, Config.E_w)
        
        # 后车
        rear_x = front_x + Config.E_l + parking_length
        self.add_obstacle(rear_x, vehicle_y, Config.E_l, Config.E_w)
        
        # 起始和目标
        init_x = 2.0
        init_y = 9.0
        init_theta = 0.0
        
        goal_x = front_x + Config.E_l / 2 + parking_length / 2
        goal_y = vehicle_y
        goal_theta = 0.0
        
        return (init_x, init_y, init_theta), (goal_x, goal_y, goal_theta), True
    
    def create_perpendicular_parking_scenario(self, scs=1.4):
        """
        垂直泊车 - 最终修复版本
        
        关键发现：
        - 目标车辆垂直放置（theta=90°）时，车辆长度方向是Y轴
        - 需要确保目标Y坐标留出足够空间，避免车辆前后端碰撞
        """
        self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))
        self.obstacles = []
        
        parking_width = scs * Config.E_w
        
        # ====== 关键修复：障碍物放置在前方 ======
        # 左右两侧车辆的Y坐标应该在停车位"前方"
        # 这样目标车辆可以停在它们"后面"
        
        parking_area_x = 8.0
        
        # 左侧车辆 - 垂直放置，在停车位前方
        left_car_x = parking_area_x - parking_width/2 - Config.E_w/2
        left_car_y = 4.0  # 停车位前方
        self.add_obstacle(left_car_x, left_car_y, Config.E_w, Config.E_l, theta=np.pi/2)
        
        # 右侧车辆
        right_car_x = parking_area_x + parking_width/2 + Config.E_w/2
        right_car_y = 4.0
        self.add_obstacle(right_car_x, right_car_y, Config.E_w, Config.E_l, theta=np.pi/2)
        
        # ====== 起点：在停车位前方的通道 ======
        init_x = parking_area_x
        init_y = 10.0
        init_theta = 0.0
        
        # ====== 目标：停车位中心 ======
        # 目标车辆垂直放置，车头朝前（theta=90°）
        # 车辆中心应该在障碍物后方足够远的位置
        
        # 计算安全的目标Y坐标
        # 左/右车辆的后端位置
        obstacle_rear_y = left_car_y - Config.E_l/2
        
        # 目标车辆的前端应该在障碍物后端之后至少0.3米
        safety_margin = 0.3
        goal_vehicle_front_y = obstacle_rear_y - safety_margin
        
        # 目标车辆中心 = 前端 - 车长/2
        goal_y = goal_vehicle_front_y - Config.E_l/2
        goal_x = parking_area_x
        goal_theta = np.pi / 2
        
        # ====== 边界安全检查 ======
        min_y_boundary = 1.0
        if goal_y - Config.E_l/2 < min_y_boundary:
            # 调整整个布局
            adjustment = min_y_boundary - (goal_y - Config.E_l/2) + 0.5
            
            # 重新创建环境
            self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))
            self.obstacles = []
            
            left_car_y += adjustment
            right_car_y += adjustment
            goal_y += adjustment
            
            self.add_obstacle(left_car_x, left_car_y, Config.E_w, Config.E_l, theta=np.pi/2)
            self.add_obstacle(right_car_x, right_car_y, Config.E_w, Config.E_l, theta=np.pi/2)
        
        print(f"\n  [垂直泊车配置 v2]")
        print(f"    停车位中心X: {parking_area_x:.2f}m")
        print(f"    停车位宽度: {parking_width:.2f}m (SCS={scs:.2f})")
        print(f"    左侧车辆: ({left_car_x:.2f}, {left_car_y:.2f})")
        print(f"    右侧车辆: ({right_car_x:.2f}, {right_car_y:.2f})")
        print(f"    障碍物后端Y: {obstacle_rear_y:.2f}m")
        print(f"    目标位置: ({goal_x:.2f}, {goal_y:.2f}, {np.degrees(goal_theta):.0f}°)")
        print(f"    目标车辆后端Y: {goal_y - Config.E_l/2:.2f}m")
        print(f"    目标车辆前端Y: {goal_y + Config.E_l/2:.2f}m")
        
        return (init_x, init_y, init_theta), (goal_x, goal_y, goal_theta), False
