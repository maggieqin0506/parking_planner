"""
Diagnose Training Data and Neural Network Quality
Find out why the NN does not improve performance
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from neural_heuristic import NeuralHeuristic, load_trained_model
from reed_shepp import ReedsShepp # Assuming this is used for comparison, although not directly in the functions.

def diagnose_training_data():
    """Check the quality of the training data"""

    print("="*70)
    print("🔍 Training Data Quality Diagnosis")
    print("="*70)

    # 1. Load training data
    try:
        with open('data/training_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"\n✓ Successfully loaded training data")
    except FileNotFoundError:
        print(f"\n❌ ERROR: training_data.pkl not found at data/")
        print(f"   Please run: python data_generator.py first")
        return None, None

    X = np.array(data['features'])
    y = np.array(data['labels'])

    print(f"\n📊 Dataset Statistics:")
    print(f"  Number of Samples: {len(X)}")
    print(f"  Feature Dimension: {X.shape[1]}")
    print(f"  Label Dimension: {y.shape}")

    # 2. Check feature distribution
    print(f"\n📈 Feature Statistics:")
    feature_names = ['Δx', 'Δy', 'sin(Δθ)', 'cos(Δθ)', 'RS_dist',
                     'E_l', 'E_w', 'E_wb', 'φ_max', 'SCS', 'is_parallel']

    for i, name in enumerate(feature_names):
        values = X[:, i]
        print(f"  {name:12s}: mean={np.mean(values):8.4f}, "
              f"std={np.std(values):8.4f}, "
              f"range=[{np.min(values):8.4f}, {np.max(values):8.4f}]")

    # 3. Check label distribution
    print(f"\n🎯 Label Statistics (True cost-to-go):")
    print(f"  Mean: {np.mean(y):.4f}")
    print(f"  Std:  {np.std(y):.4f}")
    print(f"  Min:  {np.min(y):.4f}")
    print(f"  Max:  {np.max(y):.4f}")
    print(f"  Median: {np.median(y):.4f}")

    # 4. Check relationship between RS distance and true cost
    rs_distances = X[:, 4]  # RS distance is the 5th feature (index 4)

    print(f"\n🔗 RS Distance vs True Cost Relationship:")
    ratio = y / (rs_distances + 1e-6)
    print(f"  y/RS_dist ratio mean: {np.mean(ratio):.4f}")
    print(f"  y/RS_dist ratio std:  {np.std(ratio):.4f}")

    if np.mean(ratio) > 1.5:
        print(f"  ⚠️  WARNING: True cost is much larger than RS distance (ratio > 1.5)")
        print(f"      This might indicate the training data comes from very complex scenarios")
    elif np.mean(ratio) < 0.8:
        print(f"  ⚠️  WARNING: True cost is less than RS distance (ratio < 0.8)")
        print(f"      This is unlikely, as RS should be a lower bound (admissible)")
    else:
        print(f"  ✓ Ratio is reasonable (0.8 < ratio < 1.5)")

    # 5. Check scenario distribution
    is_parallel = X[:, 10]
    n_parallel = np.sum(is_parallel == 1)
    n_perpendicular = np.sum(is_parallel == 0)

    print(f"\n🅿️  Scenario Type Distribution:")
    print(f"  Parallel parking:       {n_parallel:4d} ({n_parallel/len(X)*100:.1f}%)")
    print(f"  Perpendicular parking:  {n_perpendicular:4d} ({n_perpendicular/len(X)*100:.1f}%)")

    if abs(n_parallel - n_perpendicular) > len(X) * 0.3:
        print(f"  ⚠️  WARNING: Scenario distribution is imbalanced")
        print(f"      Suggestion: Increase samples for the minority class")

    # 6. Check SCS distribution
    scs_values = X[:, 9]
    print(f"\n📏 SCS (Space Constraint Scale) Distribution:")
    print(f"  Mean SCS: {np.mean(scs_values):.4f}")
    print(f"  Range: [{np.min(scs_values):.4f}, {np.max(scs_values):.4f}]")

    unique_scs = np.unique(scs_values)
    print(f"  Unique SCS values: {unique_scs}")

    if len(unique_scs) < 5:
        print(f"  ⚠️  WARNING: Too few unique SCS values ({len(unique_scs)})")
        print(f"      Suggestion: Generate more scenarios with different SCS")

    # 7. Visualization
    visualize_training_data(X, y, rs_distances)

    return X, y

def diagnose_neural_network():
    """Check the quality of neural network prediction"""

    print("\n" + "="*70)
    print("🧠 Neural Network Prediction Quality Diagnosis")
    print("="*70)

    # Load model (using the new wrapper)
    try:
        from neural_heuristic import create_neural_heuristic
        # Assuming the create_neural_heuristic function handles loading/initialization
        # Based on the original code, this line seems to be a wrapper to load the model for use
        neural_heuristic = create_neural_heuristic('models/neural_heuristic.pth')
        print(f"\n✓ Successfully loaded neural network model")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Trained model not found")
        print(f"   Please run: python train_nn.py first")
        return None, None
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None, None

    # Load test data (using training data for demonstration/consistency with original code)
    try:
        with open('data/training_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        # This case is handled in diagnose_training_data, but good practice to handle here too
        print(f"\n❌ ERROR: training_data.pkl not found.")
        return None, None

    X = np.array(data['features'])
    y = np.array(data['labels'])

    # Use the last 20% as test set
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # Predict (using the wrapper's predict method)
    predictions = []
    for i in range(len(X_test)):
        features = X_test[i]
        # Feature order: [dx, dy, sin(dtheta), cos(dtheta), rs_dist, E_l, E_w, E_wb, phi_max, scs, is_parallel]
        current_state = np.array([0.0, 0.0, 0.0])  # dummy state for predict method
        goal_state = np.array([features[0], features[1], np.arctan2(features[2], features[3])])
        rs_dist = features[4]
        vehicle_params = [features[5], features[6], features[7], features[8]]
        scs = features[9]
        is_parallel = bool(features[10] > 0.5)

        # NOTE: Assuming neural_heuristic.predict takes these arguments as formatted here
        pred = neural_heuristic.predict(current_state, goal_state, rs_dist, vehicle_params, scs, is_parallel)
        predictions.append(pred)

    predictions = np.array(predictions)
    # Analyze prediction quality
    print(f"\n📊 Test Set Prediction Statistics:")
    print(f"  Number of Samples: {len(y_test)}")

    errors = predictions - y_test
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / (y_test + 1e-6) * 100

    print(f"\n📐 Prediction Error:")
    print(f"  Mean Absolute Error (MAE): {np.mean(abs_errors):.4f}")
    print(f"  Std of errors:             {np.std(errors):.4f}")
    print(f"  Max error:                 {np.max(abs_errors):.4f}")
    print(f"  Median error:              {np.median(abs_errors):.4f}")

    print(f"\n📊 Relative Error (%):")
    print(f"  Mean: {np.mean(rel_errors):.2f}%")
    print(f"  Std:  {np.std(rel_errors):.2f}%")
    print(f"  Max:  {np.max(rel_errors):.2f}%")

    # Check for systematic over- or underestimation
    print(f"\n🎯 Prediction Bias:")
    mean_error = np.mean(errors)
    if mean_error > 0.5:
        print(f"  ⚠️  Systematic Overestimation: {mean_error:.4f}")
        print(f"      NN predicted cost is higher than true value")
        print(f"      → Leads to excessive caution, generating more nodes")
    elif mean_error < -0.5:
        print(f"  ⚠️  Systematic Underestimation: {mean_error:.4f}")
        print(f"      NN predicted cost is lower than true value")
        print(f"      → Violates admissibility, may fail to find the optimal solution")
    else:
        print(f"  ✓ Prediction is largely unbiased: {mean_error:.4f}")

    # Analyze by scenario type
    print(f"\n🔍 Analysis by Scenario Type:")

    is_parallel_test = X_test[:, 10]

    # Parallel parking
    parallel_mask = is_parallel_test == 1
    parallel_errors = []
    if np.sum(parallel_mask) > 0:
        parallel_errors = abs_errors[parallel_mask]
        print(f"  Parallel parking:")
        print(f"    MAE: {np.mean(parallel_errors):.4f}")
        print(f"    Relative Error: {np.mean(rel_errors[parallel_mask]):.2f}%")

    # Perpendicular parking
    perp_mask = is_parallel_test == 0
    perp_errors = []
    if np.sum(perp_mask) > 0:
        perp_errors = abs_errors[perp_mask]
        print(f"  Perpendicular parking:")
        print(f"    MAE: {np.mean(perp_errors):.4f}")
        print(f"    Relative Error: {np.mean(rel_errors[perp_mask]):.2f}%")

        # Key finding
        if np.sum(parallel_mask) > 0 and np.mean(perp_errors) > np.mean(parallel_errors) * 1.5:
            print(f"  ⚠️  Perpendicular parking error is significantly larger!")
            print(f"      This explains why performance in perpendicular scenarios is worse")

    # Visualize prediction quality
    visualize_predictions(y_test, predictions, X_test)

    return predictions, y_test

def visualize_training_data(X, y, rs_distances):
    """Visualize training data"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Label distribution
    ax = axes[0, 0]
    ax.hist(y, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('True Cost-to-go', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Training Labels', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. RS distance vs True Cost
    ax = axes[0, 1]
    ax.scatter(rs_distances, y, alpha=0.5, s=20)
    ax.plot([0, np.max(rs_distances)], [0, np.max(rs_distances)],
            'r--', label='y = RS_dist', linewidth=2)
    ax.set_xlabel('Reed-Shepp Distance', fontsize=12)
    ax.set_ylabel('True Cost-to-go', fontsize=12)
    ax.set_title('RS Distance vs True Cost', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Scenario type comparison
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
    print(f"\n✓ Training data visualization saved to: results/training_data_diagnosis.png")
    plt.close()

def visualize_predictions(y_true, y_pred, X_test):
    """Visualize prediction quality"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Prediction vs True Value
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=30)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('True Cost', fontsize=12)
    ax.set_ylabel('Predicted Cost', fontsize=12)
    ax.set_title('Prediction vs Ground Truth', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Error Distribution
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

    # 3. Error by Scenario Type
    ax = axes[1, 0]
    is_parallel = X_test[:, 10]
    parallel_errors = np.abs(errors[is_parallel == 1])
    perp_errors = np.abs(errors[is_parallel == 0])

    data = [parallel_errors, perp_errors]
    ax.boxplot(data, labels=['Parallel', 'Perpendicular'])
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Scenario Type', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Relative Error
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
    print(f"✓ Prediction quality visualization saved to: results/prediction_quality_diagnosis.png")
    plt.close()

def check_inference_speed():
    """Check inference speed"""

    print("\n" + "="*70)
    print("⚡ Inference Speed Test")
    print("="*70)

    import time
    from reed_shepp import ReedsShepp

    # Load model
    try:
        model = load_trained_model('models/neural_heuristic.pth')
        model.eval()
    except FileNotFoundError:
        print(f"\n❌ ERROR: Trained model not found for speed test.")
        return

    rs = ReedsShepp()

    # Create test input
    test_input = np.random.rand(1, 11).astype(np.float32)
    test_tensor = torch.FloatTensor(test_input)

    # Test NN inference speed
    n_iterations = 1000

    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(test_tensor)
    nn_time = (time.time() - start_time) / n_iterations * 1000

    # Test RS distance calculation speed
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([5.0, 5.0, 1.57])

    start_time = time.time()
    for _ in range(n_iterations):
        _ = rs.distance(start, goal)
    rs_time = (time.time() - start_time) / n_iterations * 1000

    print(f"\n⏱️  Time per call:")
    print(f"  Neural Network Inference: {nn_time:.4f} ms")
    print(f"  RS Distance Calculation:  {rs_time:.4f} ms")
    print(f"  Speed Ratio: NN is {nn_time/rs_time:.1f}x slower than RS")

    # Estimate overall impact
    avg_nodes_per_search = 1500
    total_overhead = nn_time * avg_nodes_per_search

    print(f"\n📊 Impact on Overall Search:")
    print(f"  Assuming an average search of {avg_nodes_per_search} nodes")
    print(f"  Total NN Overhead: {total_overhead:.1f} ms")

    if total_overhead > 100:
        print(f"  ⚠️  WARNING: NN overhead ({total_overhead:.0f}ms) is significant!")
        print(f"      This might completely offset the search efficiency gain")

def main():
    """Run all diagnostics"""

    print("\n" + "🔬"*35)
    print("Neural Network Parking Planning - Full Diagnosis")
    print("🔬"*35 + "\n")

    # 1. Training Data Diagnosis
    X, y = diagnose_training_data()

    # Check if data was loaded successfully before proceeding
    if X is None:
        print("\n❌ Cannot proceed to NN diagnosis due to data loading error.")
        return

    # 2. Neural Network Diagnosis
    predictions, y_test = diagnose_neural_network()

    # 3. Inference Speed Test
    check_inference_speed()

    # Summary
    print("\n" + "="*70)
    print("📝 Diagnosis Summary")
    print("="*70)
    print("\nCheck the following files for detailed analysis:")
    print("  1. results/training_data_diagnosis.png - Training Data Quality")
    print("  2. results/prediction_quality_diagnosis.png - Prediction Quality")
    print("\nCommon Issues:")
    print("  ✓ Insufficient training data → Increase sample size")
    print("  ✓ Imbalanced scenario distribution → Balance parallel/perpendicular samples")
    print("  ✓ Too few unique SCS values → Generate more diverse SCS scenarios")
    print("  ✓ High prediction error → Improve model architecture/training")
    print("  ✓ Slow inference speed → Model optimization/quantization")

if __name__ == '__main__':
    main()