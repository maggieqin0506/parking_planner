
"""
对比两次运行的配置差异
"""

print("="*70)
print("对比两次运行结果")
print("="*70)

# 第一次运行（成功的）
run1 = {
    'parallel': {
        'paper': {'nodes': 378, 'time': 146.46, 'success': True},
        'neural': {'nodes': 909, 'time': 391.70, 'success': True}
    },
    'perpendicular': {
        'paper': {'nodes': 23, 'time': 7.42, 'success': True},
        'neural': {'nodes': 34, 'time': 12.89, 'success': True}
    }
}

# 第二次运行（现在的）
run2 = {
    'parallel': {
        'paper': {'nodes': 472, 'time': 170.94, 'success': True},
        'neural': {'nodes': 451, 'time': 194.92, 'success': True}
    },
    'perpendicular': {
        'paper': {'nodes': 0, 'time': 0.38, 'success': False},
        'neural': {'nodes': 0, 'time': 0.0, 'success': False}
    }
}

print("\n【平行泊车对比】")
print("-"*70)
print(f"                 | 第1次运行 | 第2次运行 | 变化")
print("-"*70)
print(f"Paper - Nodes    | {run1['parallel']['paper']['nodes']:>9} | {run2['parallel']['paper']['nodes']:>9} | {run2['parallel']['paper']['nodes']-run1['parallel']['paper']['nodes']:+.0f}")
print(f"Paper - Time(ms) | {run1['parallel']['paper']['time']:>9.2f} | {run2['parallel']['paper']['time']:>9.2f} | {run2['parallel']['paper']['time']-run1['parallel']['paper']['time']:+.2f}")
print(f"Neural - Nodes   | {run1['parallel']['neural']['nodes']:>9} | {run2['parallel']['neural']['nodes']:>9} | {run2['parallel']['neural']['nodes']-run1['parallel']['neural']['nodes']:+.0f}")
print(f"Neural - Time(ms)| {run1['parallel']['neural']['time']:>9.2f} | {run2['parallel']['neural']['time']:>9.2f} | {run2['parallel']['neural']['time']-run1['parallel']['neural']['time']:+.2f}")

print("\n✅ 改善点:")
print(f"  - Neural节点数: 909 → 451 (改善 {(909-451)/909*100:.1f}%)")
print(f"  - Neural vs Paper: +140% → +4.5% (大幅改善!)")

print("\n⚠️  问题点:")
print(f"  - Paper节点数增加: 378 → 472 (+{(472-378)/378*100:.1f}%)")
print(f"  - Neural仍比Paper慢: {194.92/170.94:.2f}x")

print("\n【垂直泊车对比】")
print("-"*70)
print(f"                 | 第1次运行 | 第2次运行 | 状态")
print("-"*70)
print(f"Paper - Success  | {run1['perpendicular']['paper']['success']!s:>9} | {run2['perpendicular']['paper']['success']!s:>9} | {'❌ 失败' if not run2['perpendicular']['paper']['success'] else '✓'}")
print(f"Paper - Nodes    | {run1['perpendicular']['paper']['nodes']:>9} | {run2['perpendicular']['paper']['nodes']:>9} | {'❌' if run2['perpendicular']['paper']['nodes'] == 0 else '✓'}")
print(f"Neural - Success | {run1['perpendicular']['neural']['success']!s:>9} | {run2['perpendicular']['neural']['success']!s:>9} | {'❌ 失败' if not run2['perpendicular']['neural']['success'] else '✓'}")
print(f"Neural - Nodes   | {run1['perpendicular']['neural']['nodes']:>9} | {run2['perpendicular']['neural']['nodes']:>9} | {'❌' if run2['perpendicular']['neural']['nodes'] == 0 else '✓'}")

print("\n❌ 严重问题:")
print(f"  - 垂直泊车完全失败 (0 nodes, 0ms)")
print(f"  - 说明起点或目标立即碰撞")

print("\n" + "="*70)
print("可能的原因:")
print("="*70)
print("\n1. environment.py 被修改了")
print("   → 检查 create_perpendicular_parking_scenario() 函数")
print("   → 对比 git diff environment.py")

print("\n2. config.py 中车辆参数改变了")
print("   → 检查 E_l, E_w, E_wb 等参数")

print("\n3. hybrid_astar.py 碰撞检查逻辑改变")
print("   → 检查 _check_collision() 函数")

print("\n" + "="*70)
print("建议操作:")
print("="*70)
print("\n1. 运行诊断脚本:")
print("   python diagnose_perpendicular.py")

print("\n2. 如果是环境问题，恢复原始配置:")
print("   git checkout environment.py")
print("   或手动检查 create_perpendicular_parking_scenario()")

print("\n3. 如果环境没问题，调整参数:")
print("   - 增加 max_iterations")
print("   - 减小 step_size")
print("   - 增加 steering_angles 数量")