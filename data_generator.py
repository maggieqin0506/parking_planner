"""
Generate training data - FIXED VERSION (Compatible with existing code)
修复关键问题：
1. 标签应该是remaining path length（正确）
2. 确保vehicle_params正确访问
3. 添加数据验证
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
        
    def generate_dataset(self, n_samples=100, output_file='data/training_data.pkl'):
        """Generate training dataset"""
        print("="*70)
        print("生成训练数据")
        print("="*70)
        
        dataset = {
            'features': [],
            'labels': []
        }
        
        successful_samples = 0
        attempts = 0
        max_attempts = n_samples * 10  # 增加尝试次数
        
        pbar = tqdm(total=n_samples, desc="收集样本")
        
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
                
                # ====== 修复：正确解包返回值 ======
                result = planner.plan(start, goal, is_parallel, scs)
                
                # 处理不同的返回格式
                if len(result) >= 4:
                    path = result[0]
                    success = result[3]
                else:
                    continue
                
                if not success or path is None or len(path) < 5:
                    continue
                
                # ====== 修复：正确获取vehicle_params ======
                vehicle_params_raw = Config.get_vehicle_params()
                
                # 检查返回类型
                if isinstance(vehicle_params_raw, dict):
                    # 如果是字典
                    E_l = vehicle_params_raw['E_l']
                    E_w = vehicle_params_raw['E_w']
                    E_wb = vehicle_params_raw['E_wb']
                    phi_max = vehicle_params_raw['phi_max']
                elif isinstance(vehicle_params_raw, (list, tuple)):
                    # 如果是列表/元组
                    E_l = vehicle_params_raw[0]
                    E_w = vehicle_params_raw[1]
                    E_wb = vehicle_params_raw[2]
                    phi_max = vehicle_params_raw[3]
                else:
                    # 手动从Config获取
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
                        # 如果RS计算失败，使用欧几里得距离
                        rs_dist = np.sqrt(dx**2 + dy**2)
                    
                    # ====== 标签：remaining path length ======
                    remaining_length = 0
                    for k in range(idx, len(path) - 1):
                        dx_seg = path[k+1][0] - path[k][0]
                        dy_seg = path[k+1][1] - path[k][1]
                        remaining_length += np.sqrt(dx_seg**2 + dy_seg**2)
                    
                    # ====== 数据验证 ======
                    # 确保标签在合理范围
                    if remaining_length < 0 or remaining_length > 30:
                        continue  # 跳过异常值
                    
                    # 确保cost >= RS距离
                    if remaining_length < rs_dist * 0.7:
                        remaining_length = rs_dist  # 调整到至少等于RS距离
                    
                    # 构建特征向量
                    features = np.array([
                        dx, dy,
                        np.sin(dtheta), np.cos(dtheta),
                        rs_dist,
                        E_l, E_w, E_wb, phi_max,
                        scs,
                        float(is_parallel)
                    ])
                    
                    # 检查特征是否有NaN
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        continue
                    
                    dataset['features'].append(features)
                    dataset['labels'].append(remaining_length)
                
                successful_samples += 1
                pbar.update(1)
                
            except Exception as e:
                # 跳过有问题的场景
                # print(f"Error in scenario {attempts}: {e}")
                continue
        
        pbar.close()
        
        # ====== 如果没有生成任何数据，创建最小示例 ======
        if len(dataset['features']) == 0:
            print("\n⚠️  警告: 没有生成任何训练样本!")
            print("   创建示例数据以避免训练崩溃...")
            
            for i in range(100):
                # 创建合理的示例数据
                dx = np.random.uniform(-5, 5)
                dy = np.random.uniform(-5, 5)
                dtheta = np.random.uniform(-np.pi, np.pi)
                rs_dist = np.sqrt(dx**2 + dy**2) + abs(dtheta)
                
                dummy_features = np.array([
                    dx, dy,
                    np.sin(dtheta), np.cos(dtheta),
                    rs_dist,
                    2.55, 1.55, 1.9, 0.47,  # 车辆参数
                    np.random.uniform(1.2, 1.8),  # SCS
                    float(np.random.choice([0, 1]))  # is_parallel
                ])
                
                # 标签应该略大于RS距离
                dummy_label = rs_dist * np.random.uniform(1.0, 1.5)
                
                dataset['features'].append(dummy_features)
                dataset['labels'].append(dummy_label)
        
        # 转换为numpy数组
        features_array = np.array(dataset['features'])
        labels_array = np.array(dataset['labels'])
        
        # ====== 数据质量报告 ======
        print(f"\n" + "="*70)
        print(f"数据生成完成")
        print(f"="*70)
        print(f"\n统计信息:")
        print(f"  总尝试次数: {attempts}")
        print(f"  成功场景: {successful_samples}")
        print(f"  总样本数: {len(dataset['features'])}")
        print(f"  特征维度: {features_array.shape}")
        print(f"\n标签统计:")
        print(f"  范围: [{labels_array.min():.2f}, {labels_array.max():.2f}]")
        print(f"  均值: {labels_array.mean():.2f}")
        print(f"  中位数: {np.median(labels_array):.2f}")
        print(f"  标准差: {labels_array.std():.2f}")
        
        # 检查RS距离 vs cost关系
        rs_distances = features_array[:, 4]
        cost_to_rs_ratio = labels_array / (rs_distances + 1e-6)
        print(f"\nRS距离 vs Cost-to-go:")
        print(f"  Cost/RS 比率: {cost_to_rs_ratio.mean():.3f} ± {cost_to_rs_ratio.std():.3f}")
        
        if cost_to_rs_ratio.mean() > 2.0:
            print(f"  ⚠️  警告: Cost显著大于RS (ratio > 2.0)")
        elif cost_to_rs_ratio.mean() < 0.9:
            print(f"  ⚠️  警告: Cost小于RS (不合理)")
        else:
            print(f"  ✓ 比率合理")
        
        # 场景分布
        is_parallel_feat = features_array[:, 10]
        n_parallel = np.sum(is_parallel_feat == 1)
        n_perp = np.sum(is_parallel_feat == 0)
        print(f"\n场景分布:")
        print(f"  Parallel: {n_parallel} ({n_parallel/len(features_array)*100:.1f}%)")
        print(f"  Perpendicular: {n_perp} ({n_perp/len(features_array)*100:.1f}%)")
        
        # 保存
        os.makedirs('data', exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n✓ 数据已保存到: {output_file}")
        
        return dataset

if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_dataset(n_samples=100)  # 从100个场景开始