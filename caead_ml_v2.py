# CAEAD - Improved ML Model v2
# Dr. B. Pritam Singh
# 5000 designs + Multi-output prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("CAEAD - ML Model v2 (5000 designs)")
print("Dr. B. Pritam Singh")
print("=" * 50)

# ── Load 5000 dataset ─────────────────────────────
print("\n[1/4] Loading 5000-design dataset...")
df = pd.read_csv('caead_5000_designs.csv')
print(f"   Loaded: {len(df)} designs, {len(df.columns)} parameters")

# ── Features & Targets ────────────────────────────
X = df[['patch_length_mm','patch_width_mm',
        'substrate_er','substrate_h_mm',
        'feed_position','ground_ratio']].values

# Multi-output — predict 3 things at once!
y_freq = df['resonant_freq_GHz'].values
y_bw   = df['bandwidth_MHz'].values
y_gain = df['gain_dBi'].values

# ── Train/Test Split ──────────────────────────────
X_train, X_test, \
yf_train, yf_test, \
yb_train, yb_test, \
yg_train, yg_test = train_test_split(
    X, y_freq, y_bw, y_gain,
    test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train 3 Models ────────────────────────────────
print("\n[2/4] Training models...")

def train_model(X_tr, y_tr, name):
    print(f"   Training {name} model...", end=" ")
    m = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=1000,
        random_state=42
    )
    m.fit(X_tr, y_tr)
    print("Done!")
    return m

model_freq = train_model(X_train_sc, yf_train, "Frequency")
model_bw   = train_model(X_train_sc, yb_train, "Bandwidth")
model_gain = train_model(X_train_sc, yg_train, "Gain")

# ── Evaluate ──────────────────────────────────────
print("\n[3/4] Evaluating models...")

yf_pred = model_freq.predict(X_test_sc)
yb_pred = model_bw.predict(X_test_sc)
yg_pred = model_gain.predict(X_test_sc)

r2_freq = r2_score(yf_test, yf_pred)
r2_bw   = r2_score(yb_test, yb_pred)
r2_gain = r2_score(yg_test, yg_pred)

mae_freq = mean_absolute_error(yf_test, yf_pred)
mae_bw   = mean_absolute_error(yb_test, yb_pred)
mae_gain = mean_absolute_error(yg_test, yg_pred)

print(f"\n   Frequency : R²={r2_freq:.4f} | MAE={mae_freq:.4f} GHz")
print(f"   Bandwidth : R²={r2_bw:.4f} | MAE={mae_bw:.4f} MHz")
print(f"   Gain      : R²={r2_gain:.4f} | MAE={mae_gain:.4f} dBi")

# ── Plots ──────────────────────────────────────────
print("\n[4/4] Plotting results...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CAEAD ML v2 — Actual vs Predicted (5000 designs)\nDr. B. Pritam Singh', fontsize=13)

pairs = [
    (yf_test, yf_pred, r2_freq, 'Frequency (GHz)', 'steelblue'),
    (yb_test, yb_pred, r2_bw,   'Bandwidth (MHz)', 'green'),
    (yg_test, yg_pred, r2_gain, 'Gain (dBi)',      'orange'),
]

for ax, (y_true, y_pred, r2, label, color) in zip(axes, pairs):
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=color)
    mn, mx = y_true.min(), y_true.max()
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel(f'Actual {label}')
    ax.set_ylabel(f'Predicted {label}')
    ax.set_title(f'{label}\nR²={r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('caead_ml_v2_results.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("CAEAD ML v2 — Summary")
print("=" * 50)
print(f"Dataset    : 5000 designs (vs 500 before)")
print(f"Features   : 6 (vs 4 before)")
print(f"Outputs    : 3 (Frequency, Bandwidth, Gain)")
print(f"Freq R²    : {r2_freq:.4f}")
print(f"BW R²      : {r2_bw:.4f}")
print(f"Gain R²    : {r2_gain:.4f}")
print("=" * 50)
print("File saved: caead_ml_v2_results.png")