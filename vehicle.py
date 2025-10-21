"""
Vehicle model and related utilities
"""
import numpy as np
from config import Config

class Vehicle:
    def __init__(self):
        self.length = Config.E_l
        self.width = Config.E_w
        self.wheelbase = Config.E_wb
        self.max_steer = Config.phi_max
        self.max_curvature = np.tan(self.max_steer) / self.wheelbase
        
    def get_vehicle_box(self, x, y, theta):
        """
        Get vehicle bounding box corners
        Returns: 4x2 array of corner coordinates
        """
        # Define corners in vehicle frame
        corners_local = np.array([
            [-self.width/2, 0],  # rear left
            [self.width/2, 0],   # rear right
            [self.width/2, self.length],  # front right
            [-self.width/2, self.length]  # front left
        ])
        
        # Rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
        
        # Transform to world frame
        corners_world = corners_local @ R.T + np.array([x, y])
        return corners_world
    
    def check_collision(self, x, y, theta, occupancy_grid):
        """
        Check if vehicle collides with obstacles
        """
        corners = self.get_vehicle_box(x, y, theta)
        
        # Sample points along vehicle perimeter
        n_samples = 20
        points = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i+1) % 4]
            for t in np.linspace(0, 1, n_samples):
                point = p1 + t * (p2 - p1)
                points.append(point)
        
        points = np.array(points)
        
        # Check occupancy grid
        grid_coords = (points / Config.grid_resolution).astype(int)
        
        for coord in grid_coords:
            if (0 <= coord[0] < occupancy_grid.shape[0] and 
                0 <= coord[1] < occupancy_grid.shape[1]):
                if occupancy_grid[coord[0], coord[1]] > 0.5:
                    return True
            else:
                return True  # Out of bounds
        
        return False

class State:
    """Represents a vehicle state"""
    def __init__(self, x, y, theta, direction=1):
        self.x = x
        self.y = y
        self.theta = theta
        self.direction = direction  # 1: forward, -1: backward
        
    def __eq__(self, other):
        return (abs(self.x - other.x) < 0.1 and 
                abs(self.y - other.y) < 0.1 and 
                abs(self.theta - other.theta) < 0.1)
    
    def __hash__(self):
        return hash((round(self.x, 1), round(self.y, 1), round(self.theta, 1)))
    
    def to_array(self):
        return np.array([self.x, self.y, self.theta])