"""
Neural heuristic - FIXED VERSION
关键修复：推理时应用与训练相同的归一化
"""
import torch
import torch.nn as nn
import numpy as np

class NeuralHeuristic(nn.Module):
    """Neural network for learned heuristic"""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(11, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def load_trained_model(model_path='models/neural_heuristic.pth'):
    """
    Load trained model (returns raw PyTorch model)
    For backward compatibility - use create_neural_heuristic() instead
    """
    model = NeuralHeuristic()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 计算模型大小
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    print(f"✓ Loaded trained model from {model_path} ({size_mb:.1f} KB)")
    
    return model

class NeuralHeuristicWrapper:
    """
    Wrapper for neural heuristic with proper normalization
    
    关键修复：确保推理时使用与训练相同的归一化
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def predict(self, current_state, goal_state, rs_distance, vehicle_params, scs, is_parallel):
        """
        预测cost-to-go
        
        参数与训练时必须完全一致！
        """
        # 计算特征
        dx = goal_state[0] - current_state[0]
        dy = goal_state[1] - current_state[1]
        dtheta = goal_state[2] - current_state[2]
        
        # 角度归一化
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta < -np.pi:
            dtheta += 2 * np.pi
        
        # 处理vehicle_params（支持字典和列表）
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
            from config import Config
            E_l = Config.E_l
            E_w = Config.E_w
            E_wb = Config.E_wb
            phi_max = Config.phi_max
        
        # 构建特征向量（与训练时顺序完全一致）
        features = np.array([
            dx, dy,
            np.sin(dtheta), np.cos(dtheta),
            rs_distance,
            E_l, E_w, E_wb, phi_max,
            scs,
            1.0 if is_parallel else 0.0
        ], dtype=np.float32)
        
        # ====== 关键修复：应用与训练相同的归一化 ======
        # 这必须与 train_nn.py 的 ParkingDataset.__init__ 完全一致！
        features[0:2] /= 15.0      # dx, dy
        features[4] /= 20.0         # rs_distance
        features[5:9] /= 5.0        # vehicle params
        features[9] /= 2.0          # scs
        
        # 转换为tensor并预测
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            prediction = self.model(features_tensor).item()
        
        # 确保预测值为正
        prediction = max(0.0, prediction)
        
        # 确保预测值不小于RS距离（admissibility）
        prediction = max(prediction, rs_distance * 0.8)
        
        return prediction

def create_neural_heuristic(model_path='models/neural_heuristic.pth'):
    """
    创建神经网络启发式函数
    
    返回一个可以直接用于Hybrid A*的wrapper
    """
    model = load_trained_model(model_path)
    return NeuralHeuristicWrapper(model)