"""
Reed-Shepp curve implementation for path planning
"""
import numpy as np
from config import Config

class ReedsShepp:
    def __init__(self):
        self.max_curvature = np.tan(Config.phi_max) / Config.E_wb
        self.min_radius = 1.0 / self.max_curvature
        
    def distance(self, start, goal):
        """
        Compute Reed-Shepp distance between two poses
        Simplified implementation using straight-line + rotation approximation
        """
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dtheta = self._normalize_angle(goal[2] - start[2])
        
        # Euclidean distance
        linear_dist = np.sqrt(dx**2 + dy**2)
        
        # Rotational distance (converted to arc length)
        angular_dist = abs(dtheta) * self.min_radius
        
        # Combined distance
        return linear_dist + angular_dist
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def generate_path(self, start, goal, step_size=0.1):
        """
        Generate simple path (simplified RS curve)
        Returns list of states
        """
        # Simplified: straight line interpolation
        n_steps = int(self.distance(start, goal) / step_size)
        n_steps = max(n_steps, 2)
        
        path = []
        for i in range(n_steps + 1):
            t = i / n_steps
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            theta = start[2] + t * self._normalize_angle(goal[2] - start[2])
            path.append((x, y, theta))
        
        return path