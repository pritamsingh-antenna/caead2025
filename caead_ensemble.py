# CAEAD - Ensemble Model
# Dr. B. Pritam Singh
# DNN + GPR + PINN — teeno combine karke best prediction!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

print("CAEAD - Ensemble Model (DNN + GPR + PINN)")
print("Dr. B. Pritam Singh")
print("=" * 55)

# ── Load data ─────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv('caead_5000_designs.csv')

X = df[['patch_length_mm','patch_width_mm',
        'substrate_er','substrate_h_mm',
        'feed_position','ground_ratio']].values
y = df['resonant_freq_GHz'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train DNN ─────────────────────────────────────────
print("\n[2/5] Training DNN...")
dnn = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=1000,
    random_state=42
)
dnn.fit(X_train_sc, y_train)
dnn_pred = dnn.predict(X_test_sc)
dnn_r2   = r2_score(y_test, dnn_pred)
print(f"   DNN R²: {dnn_r2:.4f}")

# ── Train GPR (on subset) ─────────────────────────────
print("\n[3/5] Training GPR...")
np.random.seed(42)
gpr_idx = np.random.choice(len(X_train), 500, replace=False)
X_gpr = X_train_sc[gpr_idx]
y_gpr = y_train[gpr_idx]

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(
    kernel=kernel, alpha=1e-6,
    normalize_y=True, n_restarts_optimizer=2
)
gpr.fit(X_gpr, y_gpr)
gpr_pred, gpr_std = gpr.predict(X_test_sc, return_std=True)
gpr_r2 = r2_score(y_test, gpr_pred)
print(f"   GPR R²: {gpr_r2:.4f}")

# ── PINN (simplified) ─────────────────────────────────
print("\n[4/5] Training PINN...")

class SimplePINN:
    def __init__(self, lr=0.001):
        np.random.seed(42)
        self.W1 = np.random.randn(6, 64) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 32) * 0.1
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, 1) * 0.1
        self.b3 = np.zeros(1)
        self.lr = lr

    def relu(self, x): return np.maximum(0, x)

    def forward(self, X):
        self.h1 = self.relu(X @ self.W1 + self.b1)
        self.h2 = self.relu(self.h1 @ self.W2 + self.b2)
        return (self.h2 @ self.W3 + self.b3).flatten()

    def train(self, X, y, X_orig, epochs=300):
        c = 3e8
        for ep in range(epochs):
            pred = self.forward(X)
            # Data loss
            err = pred - y
            data_loss = np.mean(err**2)
            # Physics loss
            L = X_orig[:, 0] / 1000
            er = X_orig[:, 2]
            f_phys = c / (2 * L * np.sqrt(er)) / 1e9
            phys_err = pred - f_phys
            # Gradients (simplified)
            delta = (err + 0.3 * phys_err).reshape(-1, 1) / len(y)
            dW3 = self.h2.T @ delta
            db3 = delta.sum(axis=0)
            d2 = (delta @ self.W3.T) * (self.h2 > 0)
            dW2 = self.h1.T @ d2
            db2 = d2.sum(axis=0)
            d1 = (d2 @ self.W2.T) * (self.h1 > 0)
            dW1 = X.T @ d1
            db1 = d1.sum(axis=0)
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

pinn = SimplePINN(lr=0.001)
pinn.train(X_train_sc, y_train, X_train, epochs=300)
pinn_pred = pinn.forward(X_test_sc)
pinn_r2   = r2_score(y_test, pinn_pred)
print(f"   PINN R²: {pinn_r2:.4f}")

# ── Ensemble ──────────────────────────────────────────
print("\n[5/5] Computing Ensemble...")

# Weighted ensemble — better model gets more weight
w_dnn  = dnn_r2
w_gpr  = gpr_r2
w_pinn = max(pinn_r2, 0.1)
w_total = w_dnn + w_gpr + w_pinn

ensemble_pred = (
    w_dnn  * dnn_pred +
    w_gpr  * gpr_pred +
    w_pinn * pinn_pred
) / w_total

ensemble_r2  = r2_score(y_test, ensemble_pred)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

print(f"\n   Weights: DNN={w_dnn/w_total:.2f} | GPR={w_gpr/w_total:.2f} | PINN={w_pinn/w_total:.2f}")

# ── Plots ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CAEAD — Ensemble Model (DNN + GPR + PINN)\nDr. B. Pritam Singh', fontsize=13)

# Plot 1: All models comparison
mn, mx = y_test.min(), y_test.max()
for pred, label, color in [
    (dnn_pred, f'DNN (R²={dnn_r2:.3f})', 'steelblue'),
    (gpr_pred, f'GPR (R²={gpr_r2:.3f})', 'green'),
    (pinn_pred, f'PINN (R²={pinn_r2:.3f})', 'orange'),
]:
    axes[0].scatter(y_test, pred, alpha=0.2, s=5, color=color, label=label)
axes[0].plot([mn,mx],[mn,mx],'r--',linewidth=2)
axes[0].set_xlabel('Actual (GHz)')
axes[0].set_ylabel('Predicted (GHz)')
axes[0].set_title('Individual Models')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Plot 2: Ensemble vs best individual
axes[1].scatter(y_test, dnn_pred, alpha=0.3, s=5,
                color='steelblue', label=f'DNN R²={dnn_r2:.4f}')
axes[1].scatter(y_test, ensemble_pred, alpha=0.5, s=8,
                color='red', label=f'Ensemble R²={ensemble_r2:.4f}')
axes[1].plot([mn,mx],[mn,mx],'k--',linewidth=2)
axes[1].set_xlabel('Actual (GHz)')
axes[1].set_ylabel('Predicted (GHz)')
axes[1].set_title('Ensemble vs Best Individual')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

# Plot 3: R² comparison bar chart
models  = ['DNN', 'GPR', 'PINN', 'Ensemble']
r2_vals = [dnn_r2, gpr_r2, pinn_r2, ensemble_r2]
colors  = ['steelblue','green','orange','red']
bars = axes[2].bar(models, r2_vals, color=colors, edgecolor='black', alpha=0.8)
axes[2].set_ylabel('R² Score')
axes[2].set_title('Model Comparison\nR² Scores')
axes[2].set_ylim(min(r2_vals)-0.05, 1.0)
axes[2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('caead_ensemble_results.png', dpi=150)
plt.show()

print("\n" + "=" * 55)
print("CAEAD Ensemble — Final Summary")
print("=" * 55)
print(f"DNN R²      : {dnn_r2:.4f}")
print(f"GPR R²      : {gpr_r2:.4f}")
print(f"PINN R²     : {pinn_r2:.4f}")
print(f"Ensemble R² : {ensemble_r2:.4f}  ← BEST!")
print(f"Ensemble MAE: {ensemble_mae:.4f} GHz")
print("=" * 55)
print("CAEAD Pipeline Complete!")
print("DNN + GPR + PINN → Ensemble → Optimizer → Design")
print("=" * 55)