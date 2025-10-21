"""
Generate training data - simplified version
"""
import numpy as np
import pickle
from tqdm import tqdm
from environment import Environment
from hybrid_astar import HybridAStar
from reed_shepp import ReedsShepp
from config import Config
import os

class DataGenerator:
    def __init__(self):
        self.rs = ReedsShepp()
        
    def generate_dataset(self, n_samples=500, output_file='data/training_data.pkl'):
        """Generate training dataset"""
        print("Generating training data...")
        
        dataset = {
            'features': [],
            'labels': []
        }
        
        successful_samples = 0
        attempts = 0
        max_attempts = n_samples * 5
        
        pbar = tqdm(total=n_samples, desc="Collecting samples")
        
        while successful_samples < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Random scenario
            is_parallel = np.random.choice([True, False])
            
            try:
                if is_parallel:
                    scs = np.random.uniform(1.3, 1.8)
                    env = Environment()
                    start, goal, _ = env.create_parallel_parking_scenario(scs)
                else:
                    scs = np.random.uniform(1.2, 1.6)
                    env = Environment()
                    start, goal, _ = env.create_perpendicular_parking_scenario(scs)
                
                # Use Hybrid A* to get path
                # Around line 45-50, ensure this:


                # Then in the loop:
                planner = HybridAStar(env, use_neural=False)
                path, _, _, success = planner.plan(start, goal, is_parallel, scs)
                
                if not success or path is None or len(path) < 5:
                    continue
                
                # Extract features from path
                vehicle_params = Config.get_vehicle_params()
                
                # Sample points from path (not all, to avoid too much data)
                sample_indices = np.linspace(0, len(path)-1, min(10, len(path)), dtype=int)
                
                for idx in sample_indices:
                    if idx >= len(path) - 1:
                        continue
                        
                    current = path[idx]
                    
                    # Features
                    dx = goal[0] - current[0]
                    dy = goal[1] - current[1]
                    dtheta = goal[2] - current[2]
                    
                    # Normalize angle
                    while dtheta > np.pi:
                        dtheta -= 2 * np.pi
                    while dtheta < -np.pi:
                        dtheta += 2 * np.pi
                    
                    rs_dist = self.rs.distance(current, goal)
                    
                    features = np.array([
                        dx, dy,
                        np.sin(dtheta), np.cos(dtheta),
                        rs_dist,
                        vehicle_params[0], vehicle_params[1],
                        vehicle_params[2], vehicle_params[3],
                        scs,
                        float(is_parallel)
                    ])
                    
                    # Label: remaining path length
                    remaining_length = 0
                    for k in range(idx, len(path) - 1):
                        dx_seg = path[k+1][0] - path[k][0]
                        dy_seg = path[k+1][1] - path[k][1]
                        remaining_length += np.sqrt(dx_seg**2 + dy_seg**2)
                    
                    dataset['features'].append(features)
                    dataset['labels'].append(remaining_length)
                
                successful_samples += 1
                pbar.update(1)
                
            except Exception as e:
                # Skip problematic scenarios
                continue
        
        pbar.close()
        
        if len(dataset['features']) == 0:
            print("\nWARNING: No training samples generated!")
            print("Creating dummy dataset for testing...")
            # Create some dummy data so training doesn't crash
            for i in range(100):
                dummy_features = np.random.randn(11)
                dummy_label = np.random.rand() * 20
                dataset['features'].append(dummy_features)
                dataset['labels'].append(dummy_label)
        
        # Save dataset
        os.makedirs('data', exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\nGenerated {len(dataset['features'])} training samples from {successful_samples} successful paths")
        print(f"Saved to {output_file}")
        
        return dataset

if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_dataset(n_samples=100)  # Start with fewer samples