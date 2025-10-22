"""
Scenario planner with node tracking
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
        """Planning with node count tracking"""
        start_time = time.time()
        
        is_parallel = scenario_type if scenario_type is not None else False
        
        # Use Hybrid A* with paper's fixed heuristic
        from hybrid_astar import HybridAStar
        planner = HybridAStar(self.env, use_neural=False)
        result = planner.plan(start, goal, is_parallel, scs or 1.5)
        
        # Unpack results (now includes node counts)
        if len(result) == 6:
            path, _, _, success, nodes_gen, nodes_exp = result
        else:
            path, _, _, success = result
            nodes_gen, nodes_exp = 0, 0
        
        comp_time = (time.time() - start_time) * 1000
        
        if success and path:
            path_length = self._compute_path_length(path)
            return path, comp_time, path_length, True, nodes_gen, nodes_exp
        
        return None, comp_time, 0, False, nodes_gen, nodes_exp
    
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