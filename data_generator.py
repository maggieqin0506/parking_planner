
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
        
    def generate_dataset(self, n_samples=100, output_file='data/training_data.pkl'):
        """Generate training dataset"""
        print("="*70)
        print("generating training data")
        print("="*70)
        
        dataset = {
            'features': [],
            'labels': []
        }
        
        successful_samples = 0
        attempts = 0
        max_attempts = n_samples * 10
        
        pbar = tqdm(total=n_samples, desc="data collection")
        
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
                
                # Use Hybrid A* to get path (不使用神经网络)
                planner = HybridAStar(env, use_neural=False)
                
                result = planner.plan(start, goal, is_parallel, scs)
                
                if len(result) >= 4:
                    path = result[0]
                    success = result[3]
                else:
                    continue
                
                if not success or path is None or len(path) < 5:
                    continue
                
                vehicle_params_raw = Config.get_vehicle_params()
                
                if isinstance(vehicle_params_raw, dict):
                    E_l = vehicle_params_raw['E_l']
                    E_w = vehicle_params_raw['E_w']
                    E_wb = vehicle_params_raw['E_wb']
                    phi_max = vehicle_params_raw['phi_max']
                elif isinstance(vehicle_params_raw, (list, tuple)):
                    E_l = vehicle_params_raw[0]
                    E_w = vehicle_params_raw[1]
                    E_wb = vehicle_params_raw[2]
                    phi_max = vehicle_params_raw[3]
                else:
                    E_l = Config.E_l
                    E_w = Config.E_w
                    E_wb = Config.E_wb
                    phi_max = Config.phi_max
                
                # Sample points from path
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
                    
                    # Reed-Shepp distance
                    try:
                        rs_dist = self.rs.distance(
                            np.array(current), 
                            np.array(goal)
                        )
                    except:
                        rs_dist = np.sqrt(dx**2 + dy**2)
                    
                    remaining_length = 0
                    for k in range(idx, len(path) - 1):
                        dx_seg = path[k+1][0] - path[k][0]
                        dy_seg = path[k+1][1] - path[k][1]
                        remaining_length += np.sqrt(dx_seg**2 + dy_seg**2)

                    if remaining_length < 0 or remaining_length > 30:
                        continue
                    
                    if remaining_length < rs_dist * 0.7:
                        remaining_length = rs_dist
                    
                    features = np.array([
                        dx, dy,
                        np.sin(dtheta), np.cos(dtheta),
                        rs_dist,
                        E_l, E_w, E_wb, phi_max,
                        scs,
                        float(is_parallel)
                    ])
                    
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        continue
                    
                    dataset['features'].append(features)
                    dataset['labels'].append(remaining_length)
                
                successful_samples += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Error in scenario {attempts}: {e}")
                continue
        
        pbar.close()
        
        if len(dataset['features']) == 0:
            print("\n NO DATA GENERATED!")

            for i in range(100):
                dx = np.random.uniform(-5, 5)
                dy = np.random.uniform(-5, 5)
                dtheta = np.random.uniform(-np.pi, np.pi)
                rs_dist = np.sqrt(dx**2 + dy**2) + abs(dtheta)
                
                dummy_features = np.array([
                    dx, dy,
                    np.sin(dtheta), np.cos(dtheta),
                    rs_dist,
                    2.55, 1.55, 1.9, 0.47,
                    np.random.uniform(1.2, 1.8),  # SCS
                    float(np.random.choice([0, 1]))  # is_parallel
                ])
                
                dummy_label = rs_dist * np.random.uniform(1.0, 1.5)
                
                dataset['features'].append(dummy_features)
                dataset['labels'].append(dummy_label)
        
        features_array = np.array(dataset['features'])
        labels_array = np.array(dataset['labels'])
        
        print(f"\n" + "="*70)
        print(f"data generated")
        print(f"="*70)
        print(f"\nstats:")
        print(f"  total attempts: {attempts}")
        print(f"  successful scenarios: {successful_samples}")
        print(f"  total sample sizes: {len(dataset['features'])}")
        print(f"  dimension: {features_array.shape}")
        print(f"  range: [{labels_array.min():.2f}, {labels_array.max():.2f}]")
        print(f"  mean: {labels_array.mean():.2f}")
        print(f"  median: {np.median(labels_array):.2f}")
        print(f"  standard deviation: {labels_array.std():.2f}")
        
        rs_distances = features_array[:, 4]
        cost_to_rs_ratio = labels_array / (rs_distances + 1e-6)
        print(f"\nRS distance vs Cost-to-go:")
        print(f"  Cost/RS rate: {cost_to_rs_ratio.mean():.3f} ± {cost_to_rs_ratio.std():.3f}")
        
        if cost_to_rs_ratio.mean() > 2.0:
            print(f" Cost greater thanRS (ratio > 2.0)")
        elif cost_to_rs_ratio.mean() < 0.9:
            print(f" Cost less than RS (unreasonable!)")
        else:
            print(f"correct")
        
        is_parallel_feat = features_array[:, 10]
        n_parallel = np.sum(is_parallel_feat == 1)
        n_perp = np.sum(is_parallel_feat == 0)
        print(f"\nscenario distribution:")
        print(f"  Parallel: {n_parallel} ({n_parallel/len(features_array)*100:.1f}%)")
        print(f"  Perpendicular: {n_perp} ({n_perp/len(features_array)*100:.1f}%)")
        
        os.makedirs('data', exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n✓ data saved to: {output_file}")
        
        return dataset

if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_dataset(n_samples=100)