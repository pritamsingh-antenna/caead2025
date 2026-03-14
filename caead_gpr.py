# CAEAD - Gaussian Process Regression (GPR)
# Dr. B. Pritam Singh
# Uncertainty quantification — prediction ke saath confidence bhi!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

print("CAEAD - Gaussian Process Regression (GPR)")
print("Dr. B. Pritam Singh")
print("=" * 55)
print("GPR advantage: Prediction + Uncertainty estimate!")
print("=" * 55)

# ── Load data ─────────────────────────────────────────
print("\n[1/4] Loading dataset...")
df = pd.read_csv('caead_5000_designs.csv')

X = df[['patch_length_mm','patch_width_mm',
        'substrate_er','substrate_h_mm',
        'feed_position','ground_ratio']].values
y = df['resonant_freq_GHz'].values

# GPR ke liye 500 samples use karenge — GPR slow hota hai large data par
np.random.seed(42)
idx = np.random.choice(len(X), 500, replace=False)
X_gpr = X[idx]
y_gpr = y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_gpr, y_gpr, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   GPR Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train GPR ─────────────────────────────────────────
print("\n[2/4] Training GPR model...")
print("   (Matern kernel — robust for engineering data)")

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=3
)

gpr.fit(X_train_sc, y_train)
print(f"   Done! Optimized kernel: {gpr.kernel_}")

# ── Predict with uncertainty ──────────────────────────
print("\n[3/4] Predicting with uncertainty bounds...")
y_pred, y_std = gpr.predict(X_test_sc, return_std=True)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"   R²  : {r2:.4f}")
print(f"   MAE : {mae:.4f} GHz")
print(f"   Avg uncertainty (1σ): {y_std.mean():.4f} GHz")

# ── Live demo — 3 designs ─────────────────────────────
print("\n[4/4] LIVE DEMO — Uncertainty-aware predictions:")
print("=" * 55)

test_designs = [
    [20.0, 20.0, 4.4, 1.6, 0.2, 1.5],  # Standard FR4
    [15.0, 18.0, 2.2, 0.8, 0.3, 1.8],  # Low Er substrate
    [30.0, 28.0, 9.8, 2.4, 0.15, 1.6], # High Er substrate
]
names = ["Standard FR4 patch", "Low Er substrate", "High Er substrate"]

for name, design in zip(names, test_designs):
    d_sc = scaler.transform([design])
    pred, std = gpr.predict(d_sc, return_std=True)
    print(f"\n  {name}")
    print(f"  Predicted : {pred[0]:.3f} GHz")
    print(f"  Confidence: {pred[0]-2*std[0]:.3f} — {pred[0]+2*std[0]:.3f} GHz (95%)")
    print(f"  Uncertainty: ±{std[0]:.3f} GHz")

# ── Plots ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CAEAD — Gaussian Process Regression (GPR)\nDr. B. Pritam Singh', fontsize=13)

# Sort by actual for clean plot
sort_idx = np.argsort(y_test)
y_test_s  = y_test[sort_idx]
y_pred_s  = y_pred[sort_idx]
y_std_s   = y_std[sort_idx]

# Plot 1: Actual vs Predicted with uncertainty
axes[0].scatter(y_test, y_pred, alpha=0.6, s=20, color='steelblue', zorder=3)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn,mx],[mn,mx],'r--',linewidth=2,label='Perfect')
axes[0].set_xlabel('Actual Frequency (GHz)')
axes[0].set_ylabel('Predicted Frequency (GHz)')
axes[0].set_title(f'Actual vs Predicted\nR²={r2:.4f}')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Plot 2: Prediction with confidence bands
x_plot = np.arange(len(y_test_s))
axes[1].plot(x_plot, y_test_s, 'b-', linewidth=1.5, label='Actual', alpha=0.8)
axes[1].plot(x_plot, y_pred_s, 'r-', linewidth=1.5, label='GPR Predicted')
axes[1].fill_between(x_plot,
    y_pred_s - 2*y_std_s,
    y_pred_s + 2*y_std_s,
    alpha=0.3, color='red', label='95% confidence')
axes[1].set_xlabel('Test Sample Index')
axes[1].set_ylabel('Frequency (GHz)')
axes[1].set_title('GPR Predictions with\n95% Confidence Bands')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# Plot 3: Uncertainty distribution
axes[2].hist(y_std, bins=30, color='purple', edgecolor='black', alpha=0.7)
axes[2].axvline(y_std.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean σ={y_std.mean():.3f}')
axes[2].set_xlabel('Prediction Uncertainty (GHz)')
axes[2].set_ylabel('Count')
axes[2].set_title('Uncertainty Distribution\n(Lower = More Confident)')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('caead_gpr_results.png', dpi=150)
plt.show()

print("\n" + "=" * 55)
print("CAEAD GPR — Summary")
print("=" * 55)
print(f"Kernel      : Matern (nu=2.5)")
print(f"Training    : 400 designs")
print(f"R²          : {r2:.4f}")
print(f"MAE         : {mae:.4f} GHz")
print(f"Avg σ       : {y_std.mean():.4f} GHz")
print("=" * 55)
print("Key feature : Uncertainty quantification!")
print("Use case    : High-stakes designs needing")
print("              confidence intervals")
print("=" * 55)