"""
Configuration parameters for the parking planner
"""
import numpy as np

class Config:
    # Vehicle parameters (from paper Table I)
    E_l = 2.55  # Vehicle length (m)
    E_w = 1.55  # Vehicle width (m)
    E_wb = 1.9  # Wheel base length (m)
    phi_max = 0.47  # Maximum steering angle (rad)
    
    # Grid parameters
    grid_resolution = 0.2  # Grid cell size (m)
    angle_resolution = np.deg2rad(15)  # Angle discretization
    
    # Hybrid A* parameters
    w_F = 1.0  # Forward weight
    w_B = 2.0  # Backward weight
    b_h = 8.0  # Constraint distance to intermediate state
    
    # Environment size
    env_width = 15.0  # meters
    env_height = 15.0  # meters
    
    # Parking space parameters
    SCS_perp = 1.4  # SCS for perpendicular parking
    SCS_para = 1.6  # SCS for parallel parking
    
    dist_perp = 1.5 * E_l  # Extension distance for perpendicular parking
    dist_para = 1.0  # Maximum distance per round for parallel parking
    
    # Neural network parameters
    nn_hidden_size = 128
    nn_learning_rate = 0.001
    nn_batch_size = 64
    nn_epochs = 100
    
    # Simulation parameters
    n_scenarios = 100  # Number of test scenarios
    
    @staticmethod
    def get_vehicle_params():
        """Return vehicle parameters as a numpy array"""
        return np.array([Config.E_l, Config.E_w, Config.E_wb, Config.phi_max])