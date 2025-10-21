"""
Neural network heuristic function - FIXED VERSION
"""
import torch
import torch.nn as nn
import numpy as np
from config import Config
import os

class NeuralHeuristic(nn.Module):
    def __init__(self, input_size=11, hidden_size=Config.nn_hidden_size):  # ← 改为11
        super(NeuralHeuristic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()  # 确保非负输出
        )
        
    def forward(self, x):
        return self.network(x)
    
    def predict(self, current_state, goal_state, rs_distance, vehicle_params, scs, is_parallel):
        """
        Predict cost-to-go
        Input features (11 total):
          1-2:  delta_x, delta_y
          3-4:  sin(delta_theta), cos(delta_theta)
          5:    reed_shepp_distance
          6-9:  E_l, E_w, E_wb, phi_max (vehicle params)
          10:   SCS (environment constraint)
          11:   is_parallel (scenario type)
        """
        dx = goal_state[0] - current_state[0]
        dy = goal_state[1] - current_state[1]
        dtheta = goal_state[2] - current_state[2]
        
        # Normalize angle difference to [-π, π]
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta < -np.pi:
            dtheta += 2 * np.pi
        
        # Construct feature vector (11 features)
        features = np.array([
            dx,
            dy,
            np.sin(dtheta),
            np.cos(dtheta),
            rs_distance,
            vehicle_params[0],  # E_l
            vehicle_params[1],  # E_w
            vehicle_params[2],  # E_wb
            vehicle_params[3],  # phi_max
            scs,
            float(is_parallel)
        ], dtype=np.float32)
        
        # Normalize features
        features_normalized = self._normalize_features(features)
        
        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(features_normalized).unsqueeze(0)
            cost = self.network(x).item()
        
        return max(0.1, cost)  # Ensure positive and non-zero
    
    def _normalize_features(self, features):
        """Normalize features to similar scales"""
        normalized = features.copy()
        normalized[0:2] /= 15.0   # Position: normalize by environment size
        normalized[4] /= 20.0     # RS distance
        normalized[5:9] /= 5.0    # Vehicle parameters
        normalized[9] /= 2.0      # SCS
        # features[10] (is_parallel) is already 0 or 1, no normalization needed
        return normalized

def load_trained_model(model_path='models/neural_heuristic.pth'):
    """Load trained neural network model with detailed error reporting"""
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print(f"   Current directory: {os.getcwd()}")
        if os.path.exists('models'):
            print(f"   Files in models/: {os.listdir('models')}")
        else:
            print(f"   models/ directory does not exist")
        print(f"\n   Solution: Run 'python train_nn.py' to train the model")
        print(f"   Using untrained model for now...")
        return NeuralHeuristic()
    
    # Load model
    model = NeuralHeuristic()
    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        # Get file size
        file_size = os.path.getsize(model_path) / 1024  # KB
        print(f"✓ Loaded trained model from {model_path} ({file_size:.1f} KB)")
        return model
        
    except RuntimeError as e:
        print(f"❌ Error loading model: {e}")
        print(f"   This usually means the model architecture has changed")
        print(f"\n   Solution: Retrain the model with 'python train_nn.py'")
        print(f"   Using untrained model for now...")
        return NeuralHeuristic()
        
    except Exception as e:
        print(f"❌ Unexpected error loading model: {e}")
        print(f"   Using untrained model for now...")
        return NeuralHeuristic()