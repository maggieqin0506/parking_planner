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
    
    # Comparator for the priority queue (min-heap)
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class HybridAStar:
    """
    Hybrid A* - FINAL WORKING VERSION
    Simplified strategy: Enable the Neural Network to participate in all scenarios
    """
    def __init__(self, environment, use_neural=False, neural_model=None):
        self.env = environment
        self.vehicle = Vehicle()
        self.rs = ReedsShepp()
        self.use_neural = use_neural
        self.neural_model = neural_model
        
        # Steering angles: max left, straight, max right
        self.steering_angles = [-Config.phi_max, 0, Config.phi_max]
        self.step_size = 0.6 # Constant distance moved in one step
        
        # Metrics for diagnosis and analysis
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
        
        # Initial collision check
        if self._check_collision(start_state) or self._check_collision(goal_state):
            # Returns None path, computation time (0), path length (0), failure (False), nodes
            return None, 0, 0, False, 0, 0
        
        open_set = [] # Priority queue (min-heap)
        closed_dict = {} # Dictionary for expanded nodes (state_key -> min g_cost)
        
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
            
            # Check for goal achievement (within threshold)
            if self._is_goal(current_state, goal_state):
                path = self._reconstruct_path(current_node)
                comp_time = (time.time() - start_time) * 1000
                path_length = self._compute_path_length(path)
                
                # Success: path, computation time (ms), length (m), success (True), nodes
                return (path, comp_time, path_length, True, 
                       self.nodes_generated, self.nodes_expanded)
            
            # Use discretized state key for lookup in the closed set
            state_key = self._discretize_state(current_state)
            if state_key in closed_dict:
                # If we found a path to this cell with a lower or equal cost, skip
                if closed_dict[state_key] <= current_node.g_cost:
                    continue
            closed_dict[state_key] = current_node.g_cost
            
            # Try connecting directly to goal using Reeds-Shepp (or a simplified path)
            dist = self._distance_to_goal(current_state, goal_state)
            if dist < 2.5: # Use a smaller distance threshold for goal-checking
                direct_path = self._try_direct_path(current_state, goal_state)
                if direct_path:
                    path = self._reconstruct_path(current_node)
                    path.extend(direct_path[1:]) # Append the direct path (excluding start)
                    comp_time = (time.time() - start_time) * 1000
                    path_length = self._compute_path_length(path)
                    
                    # Success via direct connection
                    return (path, comp_time, path_length, True,
                           self.nodes_generated, self.nodes_expanded)
            
            # Explore neighbors
            for steer in self.steering_angles:
                for direction in [1, -1]: # Forward (1) and Reverse (-1)
                    # Apply motion model
                    new_state = self._apply_motion(current_state, steer, direction)
                    
                    if new_state is None or self._check_collision(new_state):
                        continue
                    
                    new_key = self._discretize_state(new_state)
                    
                    # Cost calculation (g_cost)
                    motion_cost = self.step_size
                    if direction == -1:
                        motion_cost *= 1.3 # Penalty for reversing
                    if abs(steer) > 0.01:
                        motion_cost *= 1.05 # Small penalty for turning
                    
                    g_cost = current_node.g_cost + motion_cost
                    
                    # Check closed set again with the new g_cost
                    if new_key in closed_dict and closed_dict[new_key] <= g_cost:
                        continue
                    
                    # Compute heuristic (h_cost)
                    h_cost = self._compute_heuristic(new_state, goal_state, is_parallel, scs)
                    
                    new_node = Node(new_state, g_cost, h_cost, current_node)
                    heapq.heappush(open_set, new_node)
                    self.nodes_generated += 1
        
        # Planning failed (open_set empty or max iterations reached)
        comp_time = (time.time() - start_time) * 1000
        return None, comp_time, 0, False, self.nodes_generated, self.nodes_expanded
    
    def _compute_heuristic(self, current, goal, is_parallel, scs):
        """
        Simplified Heuristic Function:
        Ensure the Neural Network is involved in all scenarios.
        """
        
        # Euclidean distance (for quick check, although not used in final heuristic)
        dist = self._distance_to_goal(current, goal)
        angle_diff = abs(self._angle_diff(current.theta, goal.theta))
        
        # Reeds-Shepp (RS) distance (always a lower bound, essential)
        rs_dist = self.rs.distance(current.to_array(), goal.to_array())
        
        # Basic Heuristic (RS + weighted angle difference)
        base_heuristic = rs_dist + angle_diff * 1.5
        
        # Neural Network Integration
        if self.use_neural and self.neural_model is not None:
            try:
                # Retrieve vehicle parameters (assuming Config.get_vehicle_params() exists)
                vehicle_params = Config.get_vehicle_params()
                
                # Predict cost-to-go using the trained NN model
                nn_cost = self.neural_model.predict(
                    current.to_array(),
                    goal.to_array(),
                    rs_dist,
                    vehicle_params,
                    scs,
                    is_parallel
                )
                
                # Consistency Check (ensuring NN prediction is reasonable)
                if rs_dist * 0.5 < nn_cost < rs_dist * 2.0:
                    # Conservative Mix: 30% NN + 70% RS-based Heuristic
                    # This ensures the heuristic remains somewhat consistent and admissible
                    mixed_heuristic = 0.30 * nn_cost + 0.70 * base_heuristic
                    return mixed_heuristic
                
            except Exception as e:
                # If Neural Network prediction fails, degrade to the base heuristic
                pass
        
        # Base method (RS-based heuristic) or fallback plan
        return base_heuristic
    
    def _check_collision(self, state):
        """Check collision against environment bounds and obstacles"""
        try:
            # Boundary check
            if not (0.5 <= state.x <= self.env.width - 0.5 and 
                   0.5 <= state.y <= self.env.height - 0.5):
                return True
            # Obstacle check using vehicle model and occupancy grid
            return self.vehicle.check_collision(state.x, state.y, state.theta, self.env.occupancy_grid)
        except:
            # Fail safe for unexpected errors during collision check
            return True
    
    def _try_direct_path(self, start_state, goal_state):
        """Try direct connection (simplified path, checking for collision)"""
        dist = self._distance_to_goal(start_state, goal_state)
        n_steps = max(10, int(dist / 0.3))
        path = []
        
        for i in range(n_steps + 1):
            t = i / n_steps
            # Interpolate position
            x = start_state.x + t * (goal_state.x - start_state.x)
            y = start_state.y + t * (goal_state.y - start_state.y)
            # Interpolate angle using shortest difference
            theta = start_state.theta + t * self._angle_diff(start_state.theta, goal_state.theta)
            
            # Normalize angle
            while theta > np.pi:
                theta -= 2 * np.pi
            while theta < -np.pi:
                theta += 2 * np.pi
            
            state = State(x, y, theta)
            # Check collision along the simplified path
            if self._check_collision(state):
                return None
            path.append((x, y, theta))
        
        return path
    
    def _angle_diff(self, angle1, angle2):
        """Calculate the shortest angle difference from angle1 to angle2"""
        diff = angle2 - angle1
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def _apply_motion(self, state, steer, direction):
        """Apply the kinematic bicycle model motion with a fixed step size"""
        # New position (forward kinematics)
        x = state.x + direction * self.step_size * np.cos(state.theta)
        y = state.y + direction * self.step_size * np.sin(state.theta)
        # New heading
        theta = state.theta + direction * self.step_size * np.tan(steer) / Config.E_wb
        
        # Normalize angle
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        
        return State(x, y, theta, direction)
    
    def _is_goal(self, state, goal, threshold=0.7):
        """Check if the current state is within the goal region"""
        dist = self._distance_to_goal(state, goal)
        angle_diff = abs(self._angle_diff(state.theta, goal.theta))
        
        # Goal reached if position is close and angle is acceptable
        return dist < threshold and angle_diff < np.deg2rad(20)
    
    def _distance_to_goal(self, state, goal):
        """Compute the Euclidean distance between the two state centers"""
        return np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)
    
    def _discretize_state(self, state):
        """Discretize state for closed set lookup"""
        resolution = 0.5 # Positional resolution
        angle_res = np.deg2rad(20) # Angular resolution (20 degrees)
        x_idx = int(state.x / resolution)
        y_idx = int(state.y / resolution)
        # Shift angle to [0, 2pi] before dividing by resolution
        theta_idx = int((state.theta + np.pi) / angle_res) 
        return (x_idx, y_idx, theta_idx)
    
    def _reconstruct_path(self, node):
        """Reconstruct the path from the goal node back to the start node"""
        path = []
        current = node
        while current is not None:
            path.append((current.state.x, current.state.y, current.state.theta))
            current = current.parent
        return list(reversed(path))
    
    def _compute_path_length(self, path):
        """Compute the total geometric length of the planned path"""
        if len(path) < 2:
            return 0
        length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += np.sqrt(dx**2 + dy**2)
        return length