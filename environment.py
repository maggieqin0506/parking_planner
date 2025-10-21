"""
Environment - FIXED to ensure valid goals
"""
import numpy as np
from config import Config

class Environment:
    def __init__(self, width=Config.env_width, height=Config.env_height):
        self.width = width
        self.height = height
        self.resolution = Config.grid_resolution
        
        self.grid_width = int(width / self.resolution)
        self.grid_height = int(height / self.resolution)
        
        self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))
        
    def add_obstacle(self, x, y, width, height):
        """Add rectangular obstacle"""
        x_start = int(x / self.resolution)
        y_start = int(y / self.resolution)
        x_end = int((x + width) / self.resolution)
        y_end = int((y + height) / self.resolution)
        
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(self.grid_width, x_end)
        y_end = min(self.grid_height, y_end)
        
        self.occupancy_grid[x_start:x_end, y_start:y_end] = 1.0
    
    def create_parallel_parking_scenario(self, scs=Config.SCS_para):
        """Create parallel parking scenario"""
        self.occupancy_grid.fill(0)
        
        parking_length = scs * Config.E_l
        curb_y = 4.0
        vehicle_y = curb_y + 0.3
        front_x = 6.0
        
        # Front parked vehicle
        self.add_obstacle(front_x, vehicle_y, Config.E_l, Config.E_w)
        
        # Rear parked vehicle
        rear_x = front_x + Config.E_l + parking_length
        self.add_obstacle(rear_x, vehicle_y, Config.E_l, Config.E_w)
        
        # Curb
        self.add_obstacle(0, 0, self.width, curb_y)
        
        # Far boundary
        self.add_obstacle(0, 12, self.width, self.height - 12)
        
        # Goal: center of parking space
        goal_x = front_x + Config.E_l + parking_length / 2
        goal_y = vehicle_y + Config.E_w / 2
        goal_theta = 0.0
        
        # Start: in driving lane
        init_x = 2.0
        init_y = 9.0
        init_theta = 0.0
        
        return (init_x, init_y, init_theta), (goal_x, goal_y, goal_theta), True
    
    def create_perpendicular_parking_scenario(self, scs=Config.SCS_perp):
        """Create perpendicular parking scenario"""
        self.occupancy_grid.fill(0)
        
        parking_width = scs * Config.E_w
        
        # Simple clear layout
        left_y = 6.0
        right_y = left_y + Config.E_w + parking_width
        parking_x = 10.0
        
        # Left parked vehicle
        self.add_obstacle(parking_x, left_y, Config.E_l, Config.E_w)
        
        # Right parked vehicle
        self.add_obstacle(parking_x, right_y, Config.E_l, Config.E_w)
        
        # Back wall - far away
        self.add_obstacle(13.5, 0, 0.5, self.height)
        
        # Goal: clear space in the middle, away from obstacles
        goal_x = parking_x - Config.E_l / 2 - 0.8  # Well before obstacles
        goal_y = left_y + Config.E_w + parking_width / 2
        goal_theta = 0.0
        
        # Start: in aisle
        init_x = 3.0
        init_y = goal_y
        init_theta = 0.0
        
        return (init_x, init_y, init_theta), (goal_x, goal_y, goal_theta), False
    
    def compute_euclidean_distance_map(self):
        """Compute Euclidean distance transform"""
        from scipy.ndimage import distance_transform_edt
        edm = distance_transform_edt(1 - self.occupancy_grid) * self.resolution
        return edm