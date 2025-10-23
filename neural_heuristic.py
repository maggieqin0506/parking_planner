import torch
import torch.nn as nn
import numpy as np


class NeuralHeuristic(nn.Module):
    """Neural network for learned heuristic"""

    def __init__(self):
        super().__init__()
        # Define the network architecture
        self.network = nn.Sequential(
            nn.Linear(11, 128),  # Input dimension is 11 features
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)  # Output dimension is 1 (cost-to-go)
        )

    def forward(self, x):
        return self.network(x)


def load_trained_model(model_path='models/neural_heuristic.pth'):
    """
    Load trained model (returns raw PyTorch model)
    For backward compatibility - use create_neural_heuristic() instead
    """
    model = NeuralHeuristic()
    # Load the state dictionary to the model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()  # Set the model to evaluation mode

    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    print(f"✓ Loaded trained model from {model_path} ({size_mb:.1f} KB)")

    return model


class NeuralHeuristicWrapper:
    """
    Wrapper for neural heuristic with proper normalization

    KEY FIX: Ensure the same normalization used during training is applied during inference
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()  # Ensure model is in evaluation mode

    def predict(self, current_state, goal_state, rs_distance, vehicle_params, scs, is_parallel):
        """
        Predict cost-to-go

        Parameters must exactly match those used during training!
        """
        # Calculate features
        dx = goal_state[0] - current_state[0]
        dy = goal_state[1] - current_state[1]
        dtheta = goal_state[2] - current_state[2]

        # Angle normalization
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta < -np.pi:
            dtheta += 2 * np.pi

        # Handle vehicle_params (supports dict or list/tuple)
        if isinstance(vehicle_params, dict):
            E_l = vehicle_params['E_l']
            E_w = vehicle_params['E_w']
            E_wb = vehicle_params['E_wb']
            phi_max = vehicle_params['phi_max']
        elif isinstance(vehicle_params, (list, tuple)):
            E_l = vehicle_params[0]
            E_w = vehicle_params[1]
            E_wb = vehicle_params[2]
            phi_max = vehicle_params[3]
        else:
            # Fallback to Config if type is unknown or not provided
            from config import Config
            E_l = Config.E_l
            E_w = Config.E_w
            E_wb = Config.E_wb
            phi_max = Config.phi_max

        # Build feature vector (must be in the exact order used during training)
        features = np.array([
            dx, dy,
            np.sin(dtheta), np.cos(dtheta),
            rs_distance,
            E_l, E_w, E_wb, phi_max,
            scs,
            1.0 if is_parallel else 0.0
        ], dtype=np.float32)

        # ====== KEY FIX: Apply the same normalization as during training ======
        # This MUST exactly match ParkingDataset.__init__ in train_nn.py!
        features[0:2] /= 15.0  # dx, dy normalized by max coordinate range (e.g., max environment size)
        features[4] /= 20.0  # rs_distance normalized by max RS distance
        features[5:9] /= 5.0  # vehicle params normalized by max expected parameter value
        features[9] /= 2.0  # scs normalized by max SCS value

        # Convert to tensor and predict
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
            prediction = self.model(features_tensor).item()

        # Ensure prediction is non-negative
        prediction = max(0.0, prediction)

        # Enforce admissibility (prediction must not be significantly less than RS distance)
        # Using 0.8 * rs_distance as a lower bound ensures it's *close* to admissible 
        # (assuming RS is the true lower bound)
        prediction = max(prediction, rs_distance * 0.8)

        return prediction


def create_neural_heuristic(model_path='models/neural_heuristic.pth'):
    """
    Create the neural network heuristic function

    Returns a wrapper that can be used directly in Hybrid A*
    """
    model = load_trained_model(model_path)
    return NeuralHeuristicWrapper(model)