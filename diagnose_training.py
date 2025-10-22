"""
诊断训练数据和神经网络质量
找出为什么NN没有提升性能
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from neural_heuristic import NeuralHeuristic, load_trained_model
from reed_shepp import ReedsShepp

def diagnose_training_data():
    """检查训练数据质量"""
    
    print("="*70)
    print("🔍 训练数据质量诊断")
    print("="*70)
    
    # 1. 加载训练数据
    try:
        with open('data/training_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"\n✓ 成功加载训练数据")
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到 data/training_data.pkl")
        print(f"   请先运行: python data_generator.py")
        return
    
    X = np.array(data['features'])
    y = np.array(data['labels'])
    
    print(f"\n📊 数据集统计:")
    print(f"  样本数量: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  标签维度: {y.shape}")
    
    # 2. 检查特征分布
    print(f"\n📈 特征统计:")
    feature_names = ['Δx', 'Δy', 'sin(Δθ)', 'cos(Δθ)', 'RS_dist', 
                     'E_l', 'E_w', 'E_wb', 'φ_max', 'SCS', 'is_parallel']
    
    for i, name in enumerate(feature_names):
        values = X[:, i]
        print(f"  {name:12s}: mean={np.mean(values):8.4f}, "
              f"std={np.std(values):8.4f}, "
              f"range=[{np.min(values):8.4f}, {np.max(values):8.4f}]")
    
    # 3. 检查标签分布
    print(f"\n🎯 标签统计 (真实cost-to-go):")
    print(f"  Mean: {np.mean(y):.4f}")
    print(f"  Std:  {np.std(y):.4f}")
    print(f"  Min:  {np.min(y):.4f}")
    print(f"  Max:  {np.max(y):.4f}")
    print(f"  Median: {np.median(y):.4f}")
    
    # 4. 检查RS距离 vs 真实cost的关系
    rs_distances = X[:, 4]  # RS distance是第5个特征
    
    print(f"\n🔗 RS距离 vs 真实Cost关系:")
    ratio = y / (rs_distances + 1e-6)
    print(f"  y/RS_dist ratio mean: {np.mean(ratio):.4f}")
    print(f"  y/RS_dist ratio std:  {np.std(ratio):.4f}")
    
    if np.mean(ratio) > 1.5:
        print(f"  ⚠️  警告: 真实cost远大于RS距离 (ratio > 1.5)")
        print(f"      这可能表示训练数据来自很复杂的场景")
    elif np.mean(ratio) < 0.8:
        print(f"  ⚠️  警告: 真实cost小于RS距离 (ratio < 0.8)")
        print(f"      这不太可能,RS应该是下界")
    else:
        print(f"  ✓ 比率合理 (0.8 < ratio < 1.5)")
    
    # 5. 检查场景分布
    is_parallel = X[:, 10]
    n_parallel = np.sum(is_parallel == 1)
    n_perpendicular = np.sum(is_parallel == 0)
    
    print(f"\n🅿️  场景类型分布:")
    print(f"  Parallel parking:       {n_parallel:4d} ({n_parallel/len(X)*100:.1f}%)")
    print(f"  Perpendicular parking:  {n_perpendicular:4d} ({n_perpendicular/len(X)*100:.1f}%)")
    
    if abs(n_parallel - n_perpendicular) > len(X) * 0.3:
        print(f"  ⚠️  警告: 场景分布不平衡")
        print(f"      建议: 增加少数类别的样本")
    
    # 6. 检查SCS分布
    scs_values = X[:, 9]
    print(f"\n📏 SCS (空间约束) 分布:")
    print(f"  Mean SCS: {np.mean(scs_values):.4f}")
    print(f"  Range: [{np.min(scs_values):.4f}, {np.max(scs_values):.4f}]")
    
    unique_scs = np.unique(scs_values)
    print(f"  Unique SCS values: {unique_scs}")
    
    if len(unique_scs) < 5:
        print(f"  ⚠️  警告: SCS值种类太少 ({len(unique_scs)})")
        print(f"      建议: 生成更多不同SCS的场景")
    
    # 7. 可视化
    visualize_training_data(X, y, rs_distances)
    
    return X, y

def diagnose_neural_network():
    """检查神经网络预测质量"""
    
    print("\n" + "="*70)
    print("🧠 神经网络预测质量诊断")
    print("="*70)
    
    # 加载模型（使用新的wrapper）
    try:
        from neural_heuristic import create_neural_heuristic
        neural_heuristic = create_neural_heuristic('models/neural_heuristic.pth')
        print(f"\n✓ 成功加载神经网络模型")
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到训练好的模型")
        print(f"   请先运行: python train_nn.py")
        return
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return
    
    # 加载测试数据
    with open('data/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = np.array(data['features'])
    y = np.array(data['labels'])
    
    # 使用最后20%作为测试
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # 预测（使用wrapper的predict方法）
    predictions = []
    for i in range(len(X_test)):
        features = X_test[i]
        # 特征顺序: [dx, dy, sin(dtheta), cos(dtheta), rs_dist, E_l, E_w, E_wb, phi_max, scs, is_parallel]
        current_state = np.array([0.0, 0.0, 0.0])  # dummy
        goal_state = np.array([features[0], features[1], np.arctan2(features[2], features[3])])
        rs_dist = features[4]
        vehicle_params = [features[5], features[6], features[7], features[8]]
        scs = features[9]
        is_parallel = bool(features[10] > 0.5)
        
        pred = neural_heuristic.predict(current_state, goal_state, rs_dist, vehicle_params, scs, is_parallel)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    # 分析预测质量
    print(f"\n📊 测试集预测统计:")
    print(f"  样本数: {len(y_test)}")
    
    errors = predictions - y_test
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / (y_test + 1e-6) * 100
    
    print(f"\n📐 预测误差:")
    print(f"  Mean Absolute Error: {np.mean(abs_errors):.4f}")
    print(f"  Std of errors:       {np.std(errors):.4f}")
    print(f"  Max error:           {np.max(abs_errors):.4f}")
    print(f"  Median error:        {np.median(abs_errors):.4f}")
    
    print(f"\n📊 相对误差 (%):")
    print(f"  Mean: {np.mean(rel_errors):.2f}%")
    print(f"  Std:  {np.std(rel_errors):.2f}%")
    print(f"  Max:  {np.max(rel_errors):.2f}%")
    
    # 检查是否系统性高估或低估
    print(f"\n🎯 预测偏差:")
    mean_error = np.mean(errors)
    if mean_error > 0.5:
        print(f"  ⚠️  系统性高估: {mean_error:.4f}")
        print(f"      NN预测的cost高于真实值")
        print(f"      → 会导致过度谨慎,生成更多节点")
    elif mean_error < -0.5:
        print(f"  ⚠️  系统性低估: {mean_error:.4f}")
        print(f"      NN预测的cost低于真实值")
        print(f"      → 违反admissibility,可能找不到最优解")
    else:
        print(f"  ✓ 预测基本无偏: {mean_error:.4f}")
    
    # 按场景类型分析
    print(f"\n🔍 按场景类型分析:")
    
    is_parallel_test = X_test[:, 10]
    
    # Parallel parking
    parallel_mask = is_parallel_test == 1
    if np.sum(parallel_mask) > 0:
        parallel_errors = abs_errors[parallel_mask]
        print(f"  Parallel parking:")
        print(f"    MAE: {np.mean(parallel_errors):.4f}")
        print(f"    相对误差: {np.mean(rel_errors[parallel_mask]):.2f}%")
    
    # Perpendicular parking
    perp_mask = is_parallel_test == 0
    if np.sum(perp_mask) > 0:
        perp_errors = abs_errors[perp_mask]
        print(f"  Perpendicular parking:")
        print(f"    MAE: {np.mean(perp_errors):.4f}")
        print(f"    相对误差: {np.mean(rel_errors[perp_mask]):.2f}%")
        
        # 关键发现
        if np.mean(perp_errors) > np.mean(parallel_errors) * 1.5:
            print(f"  ⚠️  垂直泊车误差明显更大!")
            print(f"      这解释了为什么perpendicular场景性能更差")
    
    # 可视化预测质量
    visualize_predictions(y_test, predictions, X_test)
    
    return predictions, y_test

def visualize_training_data(X, y, rs_distances):
    """可视化训练数据"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 标签分布
    ax = axes[0, 0]
    ax.hist(y, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('True Cost-to-go', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Training Labels', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. RS距离 vs 真实Cost
    ax = axes[0, 1]
    ax.scatter(rs_distances, y, alpha=0.5, s=20)
    ax.plot([0, np.max(rs_distances)], [0, np.max(rs_distances)], 
            'r--', label='y = RS_dist', linewidth=2)
    ax.set_xlabel('Reed-Shepp Distance', fontsize=12)
    ax.set_ylabel('True Cost-to-go', fontsize=12)
    ax.set_title('RS Distance vs True Cost', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 场景类型对比
    ax = axes[1, 0]
    is_parallel = X[:, 10]
    parallel_costs = y[is_parallel == 1]
    perp_costs = y[is_parallel == 0]
    
    data_to_plot = [parallel_costs, perp_costs]
    ax.boxplot(data_to_plot, labels=['Parallel', 'Perpendicular'])
    ax.set_ylabel('Cost-to-go', fontsize=12)
    ax.set_title('Cost Distribution by Scenario Type', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. SCS vs Cost
    ax = axes[1, 1]
    scs_values = X[:, 9]
    ax.scatter(scs_values, y, alpha=0.5, s=20, c=is_parallel, cmap='viridis')
    ax.set_xlabel('SCS (Space Constraint Scale)', fontsize=12)
    ax.set_ylabel('True Cost-to-go', fontsize=12)
    ax.set_title('SCS vs Cost (color=scenario type)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_data_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 训练数据可视化保存到: results/training_data_diagnosis.png")
    plt.close()

def visualize_predictions(y_true, y_pred, X_test):
    """可视化预测质量"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 预测 vs 真实值
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=30)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('True Cost', fontsize=12)
    ax.set_ylabel('Predicted Cost', fontsize=12)
    ax.set_title('Prediction vs Ground Truth', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 误差分布
    ax = axes[0, 1]
    errors = y_pred - y_true
    ax.hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors), color='blue', linestyle='--', linewidth=2, 
              label=f'Mean Error: {np.mean(errors):.3f}')
    ax.set_xlabel('Prediction Error (pred - true)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 按场景类型的误差
    ax = axes[1, 0]
    is_parallel = X_test[:, 10]
    parallel_errors = np.abs(errors[is_parallel == 1])
    perp_errors = np.abs(errors[is_parallel == 0])
    
    data = [parallel_errors, perp_errors]
    ax.boxplot(data, labels=['Parallel', 'Perpendicular'])
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Scenario Type', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 相对误差
    ax = axes[1, 1]
    rel_errors = np.abs(errors) / (y_true + 1e-6) * 100
    ax.hist(rel_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Relative Error (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Relative Error Distribution (Mean: {np.mean(rel_errors):.1f}%)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/prediction_quality_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"✓ 预测质量可视化保存到: results/prediction_quality_diagnosis.png")
    plt.close()

def check_inference_speed():
    """检查推理速度"""
    
    print("\n" + "="*70)
    print("⚡ 推理速度测试")
    print("="*70)
    
    import time
    from reed_shepp import ReedsShepp
    
    # 加载模型
    model = load_trained_model('models/neural_heuristic.pth')
    model.eval()
    
    rs = ReedsShepp()
    
    # 创建测试输入
    test_input = np.random.rand(1, 11).astype(np.float32)
    test_tensor = torch.FloatTensor(test_input)
    
    # 测试NN推理速度
    n_iterations = 1000
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(test_tensor)
    nn_time = (time.time() - start_time) / n_iterations * 1000
    
    # 测试RS距离计算速度
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([5.0, 5.0, 1.57])
    
    start_time = time.time()
    for _ in range(n_iterations):
        _ = rs.distance(start, goal)
    rs_time = (time.time() - start_time) / n_iterations * 1000
    
    print(f"\n⏱️  每次调用时间:")
    print(f"  神经网络推理: {nn_time:.4f} ms")
    print(f"  RS距离计算:   {rs_time:.4f} ms")
    print(f"  速度比: NN is {nn_time/rs_time:.1f}x slower than RS")
    
    # 估算整体影响
    avg_nodes_per_search = 1500
    total_overhead = nn_time * avg_nodes_per_search
    
    print(f"\n📊 对整体搜索的影响:")
    print(f"  假设平均搜索{avg_nodes_per_search}个节点")
    print(f"  NN总开销: {total_overhead:.1f} ms")
    
    if total_overhead > 100:
        print(f"  ⚠️  警告: NN开销 ({total_overhead:.0f}ms) 很大!")
        print(f"      这可能完全抵消搜索效率提升")

def main():
    """运行所有诊断"""
    
    print("\n" + "🔬"*35)
    print("神经网络泊车规划 - 完整诊断")
    print("🔬"*35 + "\n")
    
    # 1. 训练数据诊断
    X, y = diagnose_training_data()
    
    # 2. 神经网络诊断
    predictions, y_test = diagnose_neural_network()
    
    # 3. 推理速度测试
    check_inference_speed()
    
    # 总结
    print("\n" + "="*70)
    print("📝 诊断总结")
    print("="*70)
    print("\n查看以下文件获取详细分析:")
    print("  1. results/training_data_diagnosis.png - 训练数据质量")
    print("  2. results/prediction_quality_diagnosis.png - 预测质量")
    print("\n常见问题:")
    print("  ✓ 训练数据不够多 → 增加样本")
    print("  ✓ 场景分布不平衡 → 平衡parallel/perpendicular")
    print("  ✓ SCS值种类太少 → 生成更多不同SCS")
    print("  ✓ 预测误差太大 → 改进模型架构/训练")
    print("  ✓ 推理太慢 → 模型优化/量化")

if __name__ == '__main__':
    main()