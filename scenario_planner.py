"""
Scenario planner using Hybrid A*
"""
import numpy as np
from vehicle import Vehicle
from config import Config
import time

class ScenarioPlanner:
    def __init__(self, environment):
        self.env = environment
        self.vehicle = Vehicle()
        
    def plan(self, start, goal, scenario_type=None, scs=None):
        """Planning using Hybrid A* with fixed heuristic"""
        start_time = time.time()
        
        is_parallel = scenario_type if scenario_type is not None else False
        
        # Use Hybrid A* with FIXED heuristic (paper's method)
        from hybrid_astar import HybridAStar
        planner = HybridAStar(self.env, use_neural=False)
        path, _, _, success = planner.plan(start, goal, is_parallel, scs or 1.5)
        
        comp_time = (time.time() - start_time) * 1000
        
        if success and path:
            path_length = self._compute_path_length(path)
            return path, comp_time, path_length, True
        
        return None, comp_time, 0, False
    
    def _compute_path_length(self, path):
        """Compute total path length"""
        if len(path) < 2:
            return 0
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx**2 + dy**2)
        return length