
print("="*70)
print("compare two results")
print("="*70)

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
