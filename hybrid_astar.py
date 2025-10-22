"""
Hybrid A* - FINAL WORKING VERSION
简化策略：让神经网络在所有场景都参与
"""
import numpy as np
import heapq
from vehicle import Vehicle, State
from reed_shepp import ReedsShepp
from config import Config
import time

class Node:
    def __init__(self, state, g_cost, h_cost, parent=None):
        self.state = state
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class HybridAStar:
    def __init__(self, environment, use_neural=False, neural_model=None):
        self.env = environment
        self.vehicle = Vehicle()
        self.rs = ReedsShepp()
        self.use_neural = use_neural
        self.neural_model = neural_model
        
        self.steering_angles = [-Config.phi_max, 0, Config.phi_max]
        self.step_size = 0.6
        
        self.nodes_generated = 0
        self.nodes_expanded = 0
        self.nodes_in_open = 0
        
    def plan(self, start, goal, is_parallel, scs):
        """Main planning function with node tracking"""
        start_time = time.time()
        
        self.nodes_generated = 0
        self.nodes_expanded = 0
        self.nodes_in_open = 0
        
        start_state = State(start[0], start[1], start[2])
        goal_state = State(goal[0], goal[1], goal[2])
        
        if self._check_collision(start_state) or self._check_collision(goal_state):
            return None, 0, 0, False, 0, 0
        
        open_set = []
        closed_dict = {}
        
        h_cost = self._compute_heuristic(start_state, goal_state, is_parallel, scs)
        start_node = Node(start_state, 0, h_cost)
        
        heapq.heappush(open_set, start_node)
        self.nodes_generated += 1
        
        iterations = 0
        max_iterations = 5000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            self.nodes_in_open = len(open_set)
            
            current_node = heapq.heappop(open_set)
            current_state = current_node.state
            self.nodes_expanded += 1
            
            if self._is_goal(current_state, goal_state):
                path = self._reconstruct_path(current_node)
                comp_time = (time.time() - start_time) * 1000
                path_length = self._compute_path_length(path)
                
                return (path, comp_time, path_length, True, 
                       self.nodes_generated, self.nodes_expanded)
            
            state_key = self._discretize_state(current_state)
            if state_key in closed_dict:
                if closed_dict[state_key] <= current_node.g_cost:
                    continue
            closed_dict[state_key] = current_node.g_cost
            
            dist = self._distance_to_goal(current_state, goal_state)
            if dist < 2.5:
                direct_path = self._try_direct_path(current_state, goal_state)
                if direct_path:
                    path = self._reconstruct_path(current_node)
                    path.extend(direct_path[1:])
                    comp_time = (time.time() - start_time) * 1000
                    path_length = self._compute_path_length(path)
                    
                    return (path, comp_time, path_length, True,
                           self.nodes_generated, self.nodes_expanded)
            
            for steer in self.steering_angles:
                for direction in [1, -1]:
                    new_state = self._apply_motion(current_state, steer, direction)
                    
                    if new_state is None or self._check_collision(new_state):
                        continue
                    
                    new_key = self._discretize_state(new_state)
                    
                    motion_cost = self.step_size
                    if direction == -1:
                        motion_cost *= 1.3
                    if abs(steer) > 0.01:
                        motion_cost *= 1.05
                    
                    g_cost = current_node.g_cost + motion_cost
                    
                    if new_key in closed_dict and closed_dict[new_key] <= g_cost:
                        continue
                    
                    h_cost = self._compute_heuristic(new_state, goal_state, is_parallel, scs)
                    
                    new_node = Node(new_state, g_cost, h_cost, current_node)
                    heapq.heappush(open_set, new_node)
                    self.nodes_generated += 1
        
        comp_time = (time.time() - start_time) * 1000
        return None, comp_time, 0, False, self.nodes_generated, self.nodes_expanded
    
    def _compute_heuristic(self, current, goal, is_parallel, scs):
        """
        简化启发式函数
        让神经网络在所有场景都参与
        """
        
        dist = self._distance_to_goal(current, goal)
        angle_diff = abs(self._angle_diff(current.theta, goal.theta))
        
        # Reeds-Shepp 距离
        rs_dist = self.rs.distance(current.to_array(), goal.to_array())
        
        # 基础启发式
        base_heuristic = rs_dist + angle_diff * 1.5
        
        # 如果使用神经网络
        if self.use_neural and self.neural_model is not None:
            try:
                vehicle_params = Config.get_vehicle_params()
                
                nn_cost = self.neural_model.predict(
                    current.to_array(),
                    goal.to_array(),
                    rs_dist,
                    vehicle_params,
                    scs,
                    is_parallel
                )
                
                # 一致性检查
                if rs_dist * 0.5 < nn_cost < rs_dist * 2.0:
                    # 保守混合：30% NN + 70% RS
                    mixed_heuristic = 0.30 * nn_cost + 0.70 * base_heuristic
                    return mixed_heuristic
                
            except Exception as e:
                # 神经网络失败，降级到基础启发式
                pass
        
        # 论文方法或降级方案
        return base_heuristic
    
    def _check_collision(self, state):
        """Check collision"""
        try:
            if not (0.5 <= state.x <= self.env.width - 0.5 and 
                   0.5 <= state.y <= self.env.height - 0.5):
                return True
            return self.vehicle.check_collision(state.x, state.y, state.theta, self.env.occupancy_grid)
        except:
            return True
    
    def _try_direct_path(self, start_state, goal_state):
        """Try direct connection"""
        dist = self._distance_to_goal(start_state, goal_state)
        n_steps = max(10, int(dist / 0.3))
        path = []
        
        for i in range(n_steps + 1):
            t = i / n_steps
            x = start_state.x + t * (goal_state.x - start_state.x)
            y = start_state.y + t * (goal_state.y - start_state.y)
            theta = start_state.theta + t * self._angle_diff(start_state.theta, goal_state.theta)
            
            while theta > np.pi:
                theta -= 2 * np.pi
            while theta < -np.pi:
                theta += 2 * np.pi
            
            state = State(x, y, theta)
            if self._check_collision(state):
                return None
            path.append((x, y, theta))
        
        return path
    
    def _angle_diff(self, angle1, angle2):
        """Shortest angle difference"""
        diff = angle2 - angle1
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def _apply_motion(self, state, steer, direction):
        """Apply motion"""
        x = state.x + direction * self.step_size * np.cos(state.theta)
        y = state.y + direction * self.step_size * np.sin(state.theta)
        theta = state.theta + direction * self.step_size * np.tan(steer) / Config.E_wb
        
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        
        return State(x, y, theta, direction)
    
    def _is_goal(self, state, goal, threshold=0.7):
        """Check if at goal"""
        dist = self._distance_to_goal(state, goal)
        angle_diff = abs(self._angle_diff(state.theta, goal.theta))
        return dist < threshold and angle_diff < np.deg2rad(20)
    
    def _distance_to_goal(self, state, goal):
        """Euclidean distance"""
        return np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)
    
    def _discretize_state(self, state):
        """Discretize state"""
        resolution = 0.5
        angle_res = np.deg2rad(20)
        x_idx = int(state.x / resolution)
        y_idx = int(state.y / resolution)
        theta_idx = int((state.theta + np.pi) / angle_res)
        return (x_idx, y_idx, theta_idx)
    
    def _reconstruct_path(self, node):
        """Reconstruct path"""
        path = []
        current = node
        while current is not None:
            path.append((current.state.x, current.state.y, current.state.theta))
            current = current.parent
        return list(reversed(path))
    
    def _compute_path_length(self, path):
        """Compute path length"""
        if len(path) < 2:
            return 0
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx**2 + dy**2)
        return length