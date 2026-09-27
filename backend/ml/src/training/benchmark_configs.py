"""
Benchmark script to test memory-safe and performance-optimal tree parameters
across the 8.97M training dataset.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score

train_data = np.load('e:/ML Claude/lulc-project/backend/ml/data/processed/train.npz')
val_data = np.load('e:/ML Claude/lulc-project/backend/ml/data/processed/val.npz')
X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']

print(f"Loaded Train: {len(X_train):,} px, Val: {len(X_val):,} px")

configs = [
    {
        "name": "Config A: max_depth=30, min_leaf=2, max_samples=0.4",
        "max_depth": 30,
        "min_samples_leaf": 2,
        "max_samples": 0.4
    },
    {
        "name": "Config B: max_depth=None, min_leaf=3, max_samples=0.35",
        "max_depth": None,
        "min_samples_leaf": 3,
        "max_samples": 0.35
    },
    {
        "name": "Config C: max_depth=30, min_leaf=1, max_samples=0.35",
        "max_depth": 30,
        "min_samples_leaf": 1,
        "max_samples": 0.35
    },
    {
        "name": "Config D: max_depth=25, min_leaf=2, max_samples=0.35 (Old baseline)",
        "max_depth": 25,
        "min_samples_leaf": 2,
        "max_samples": 0.35
    }
]

for cfg in configs:
    print("-" * 70)
    print(f"Testing: {cfg['name']}")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=cfg['max_depth'],
        min_samples_leaf=cfg['min_samples_leaf'],
        max_samples=cfg['max_samples'],
        class_weight='balanced',
        n_jobs=4,
        random_state=42
    )
    rf.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    val_preds = rf.predict(X_val)
    acc = np.mean(val_preds == y_val)
    macro_f1 = f1_score(y_val, val_preds, average='macro')
    weighted_f1 = f1_score(y_val, val_preds, average='weighted')
    kappa = cohen_kappa_score(y_val, val_preds)
    
    print(f"  Fit Time     : {fit_time:.1f}s")
    print(f"  Val Accuracy : {acc * 100:.2f}%")
    print(f"  Macro F1     : {macro_f1 * 100:.2f}%")
    print(f"  Weighted F1  : {weighted_f1 * 100:.2f}%")
    print(f"  Cohen Kappa  : {kappa:.4f}")
